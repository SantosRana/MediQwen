"""
src/multimodal/visual_embedder.py

Visual Embedder — encodes images and text into a shared embedding space
using CLIP (or a CLIP-compatible model) so that image queries can be
fused with text-based RAG retrieval.

Design goals:
  • Drop-in replacement / complement for src/models/embeddings.py
  • Returns normalized float32 vectors compatible with ChromaDB
  • Supports text-only, image-only, and fused (text + image) embeddings
  • Graceful fallback to text-only if image encoding fails
"""

import io
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default model config
# ---------------------------------------------------------------------------

DEFAULT_CLIP_MODEL    = "openai/clip-vit-base-patch32"   # ~600 MB, widely cached
FALLBACK_TEXT_MODEL   = "BAAI/bge-large-en-v1.5"         # same as settings.py default

FUSION_WEIGHTS = {
    "text":  0.6,   # text carries more semantic weight in medical domain
    "image": 0.4,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VisualEmbedder:
    """
    Multimodal embedder that encodes:
      - text  → 512-d CLIP text embedding
      - image → 512-d CLIP image embedding
      - fused → weighted average of text + image embeddings

    Falls back to text-only BGE embeddings when CLIP is unavailable.

    Usage:
        embedder = VisualEmbedder()
        vec = embedder.embed_text("patient has chest pain")
        vec = embedder.embed_image(pil_image)
        vec = embedder.embed_multimodal("rash on arm", pil_image)
    """

    def __init__(
        self,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        text_model_name: str = FALLBACK_TEXT_MODEL,
        device: Optional[str] = None,
        text_weight: float = FUSION_WEIGHTS["text"],
        image_weight: float = FUSION_WEIGHTS["image"],
        normalize: bool = True,
    ):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize     = normalize
        self.text_weight   = text_weight
        self.image_weight  = image_weight

        self._clip_available  = False
        self._text_available  = False

        # -- Try loading CLIP -------------------------------------------
        try:
            from transformers import CLIPProcessor, CLIPModel
            logger.info(f"🔧 Loading CLIP model: {clip_model_name}")
            self._clip_model     = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self._clip_model.eval()
            self._clip_available = True
            self.embedding_dim   = self._clip_model.config.projection_dim  # usually 512
            logger.info(f"✅ CLIP loaded | dim={self.embedding_dim} | device={self.device}")
        except Exception as e:
            logger.warning(
                f"⚠️ CLIP unavailable ({e}). "
                f"Install: pip install transformers torch"
            )

        # -- Fallback text embedder (BGE) --------------------------------
        if not self._clip_available:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info(f"🔧 Loading fallback text embedder: {text_model_name}")
                self._text_embedder  = HuggingFaceEmbeddings(
                    model_name=text_model_name,
                    model_kwargs={"device": self.device},
                )
                self._text_available = True
                self.embedding_dim   = 1024  # bge-large-en-v1.5
                logger.info(f"✅ Fallback text embedder loaded | dim={self.embedding_dim}")
            except Exception as e:
                raise RuntimeError(
                    f"Neither CLIP nor fallback text embedder could be loaded. "
                    f"Last error: {e}"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Encode a text string to a 1-D float32 embedding vector.
        """
        if not text or not text.strip():
            logger.warning("⚠️ embed_text called with empty string")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self._clip_available:
            return self._clip_encode_text(text)
        else:
            return self._bge_encode_text(text)

    def embed_image(self, image: Union[Image.Image, bytes, str]) -> np.ndarray:
        """
        Encode a PIL image (or path / bytes) to a 1-D float32 embedding vector.
        Returns zero vector with a warning if CLIP is unavailable.
        """
        img = self._to_pil(image)

        if not self._clip_available:
            logger.warning(
                "⚠️ CLIP not available — returning zero vector for image. "
                "Install transformers to enable image embeddings."
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)

        return self._clip_encode_image(img)

    def embed_multimodal(
        self,
        text: str,
        image: Union[Image.Image, bytes, str],
        text_weight: Optional[float] = None,
        image_weight: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fuse text and image embeddings via weighted average.

        Args:
            text:         Clinical query or image caption.
            image:        PIL image, bytes, or file path.
            text_weight:  Override default text fusion weight.
            image_weight: Override default image fusion weight.

        Returns:
            Normalized fused embedding vector.
        """
        tw = text_weight  if text_weight  is not None else self.text_weight
        iw = image_weight if image_weight is not None else self.image_weight

        t_vec = self.embed_text(text)
        i_vec = self.embed_image(image)

        # If CLIP unavailable image vector is zero — fall back to text-only
        if not self._clip_available or np.allclose(i_vec, 0):
            logger.info("ℹ️ Multimodal fusion using text-only (image vector is zero)")
            return t_vec

        # Normalize before weighting so scale is comparable
        t_norm = self._normalize(t_vec)
        i_norm = self._normalize(i_vec)

        fused = tw * t_norm + iw * i_norm
        return self._normalize(fused) if self.normalize else fused

    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts. Returns shape (N, dim).
        """
        vecs = [self.embed_text(t) for t in texts]
        return np.stack(vecs, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        a = self._normalize(vec_a)
        b = self._normalize(vec_b)
        return float(np.dot(a, b))

    def rank_texts_by_image(
        self,
        image: Union[Image.Image, bytes, str],
        texts: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Rank a list of text strings by visual similarity to an image.
        Useful for zero-shot image classification against medical labels.

        Returns:
            List of (text, score) tuples sorted descending.
        """
        img_vec   = self.embed_image(image)
        text_vecs = self.embed_texts_batch(texts)

        scores = [self.similarity(img_vec, tv) for tv in text_vecs]
        ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Internal CLIP helpers
    # ------------------------------------------------------------------

    def _clip_encode_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self._clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,        # CLIP max token length
            ).to(self.device)
            features = self._clip_model.get_text_features(**inputs)
            vec = features[0].cpu().float().numpy()
        return self._normalize(vec) if self.normalize else vec

    def _clip_encode_image(self, img: Image.Image) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self._clip_processor(
                images=img,
                return_tensors="pt",
            ).to(self.device)
            features = self._clip_model.get_image_features(**inputs)
            vec = features[0].cpu().float().numpy()
        return self._normalize(vec) if self.normalize else vec

    # ------------------------------------------------------------------
    # Internal BGE fallback
    # ------------------------------------------------------------------

    def _bge_encode_text(self, text: str) -> np.ndarray:
        vec = self._text_embedder.embed_query(text)
        vec = np.array(vec, dtype=np.float32)
        return self._normalize(vec) if self.normalize else vec

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    @staticmethod
    def _to_pil(image: Union[Image.Image, bytes, str]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Cannot convert {type(image)} to PIL Image")


# ---------------------------------------------------------------------------
# Convenience helper (mirrors embeddings.py pattern)
# ---------------------------------------------------------------------------

def initialize_visual_embedder(
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    device: Optional[str] = None,
) -> VisualEmbedder:
    """
    Factory function — mirrors initialize_embeddings() in src/models/embeddings.py.

    Args:
        clip_model_name: HuggingFace CLIP model ID.
        device:          'cpu', 'cuda', or None for auto-detect.

    Returns:
        Configured VisualEmbedder instance.
    """
    return VisualEmbedder(clip_model_name=clip_model_name, device=device)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embedder = VisualEmbedder()

    # Text embedding
    t_vec = embedder.embed_text("patient shows signs of hypertension")
    print(f"✅ Text embedding shape: {t_vec.shape}, norm={np.linalg.norm(t_vec):.4f}")

    # Synthetic image embedding
    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    i_vec    = embedder.embed_image(test_img)
    print(f"✅ Image embedding shape: {i_vec.shape}, norm={np.linalg.norm(i_vec):.4f}")

    # Fused embedding
    f_vec = embedder.embed_multimodal("rash on forearm", test_img)
    print(f"✅ Fused embedding shape: {f_vec.shape}, norm={np.linalg.norm(f_vec):.4f}")

    # Similarity
    sim = embedder.similarity(t_vec, i_vec)
    print(f"   text↔image cosine similarity: {sim:.4f}")
