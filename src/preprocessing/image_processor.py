"""
src/multimodal/image_processor.py

Medical Image Processor — handles loading, validation, preprocessing,
and metadata extraction for medical images fed into the RAG pipeline.

Supports: X-rays, skin photos, wound photos, lab reports (OCR).
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Standard sizes for different modalities
MODALITY_SIZES = {
    "xray":       (512, 512),
    "skin":       (384, 384),
    "wound":      (384, 384),
    "lab_report": (768, 1024),   # tall for text-heavy images
    "general":    (448, 448),
}

# Visual keywords used to auto-detect modality from filename / folder
MODALITY_HINTS = {
    "xray":       ["xray", "x-ray", "chest", "radiograph", "ct", "mri"],
    "skin":       ["skin", "derma", "rash", "lesion", "mole", "melanoma"],
    "wound":      ["wound", "burn", "laceration", "injury", "trauma", "cut"],
    "lab_report": ["lab", "report", "result", "test", "ecg", "ekg", "blood"],
}

# Risk keywords for visual context text (OCR results / captions)
VISUAL_RISK_KEYWORDS = {
    "emergency": [
        "cardiac arrest", "no pulse", "not breathing", "severe haemorrhage",
        "unconscious", "tension pneumothorax", "airway obstruction",
    ],
    "high": [
        "fracture", "deep laceration", "severe burn", "infected wound",
        "abscess", "chest pain", "shortness of breath", "melanoma",
        "stage 3", "stage 4", "metastatic",
    ],
    "medium": [
        "inflammation", "swelling", "redness", "mild infection",
        "superficial burn", "contusion", "bruise",
    ],
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ProcessedImage:
    """
    Container for a preprocessed medical image and its metadata.
    """
    # Core image data
    image: Image.Image                   # PIL image (preprocessed)
    image_array: np.ndarray              # HxWxC numpy array, uint8
    image_bytes: bytes                   # PNG bytes for API / embedding

    # Provenance
    source_path: Optional[str] = None
    modality: str = "general"

    # Enriched metadata (matches MedicalChunk metadata schema)
    metadata: Dict = field(default_factory=dict)

    # Optional OCR / caption text
    extracted_text: str = ""

    def __post_init__(self):
        # Attach basic size info to metadata
        h, w = self.image_array.shape[:2]
        self.metadata.setdefault("image_width",  w)
        self.metadata.setdefault("image_height", h)
        self.metadata.setdefault("modality",     self.modality)


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------

class MedicalImageProcessor:
    """
    Preprocesses medical images for multimodal RAG.

    Pipeline per image:
        1. Load & validate format / size
        2. Detect modality (xray / skin / wound / lab_report / general)
        3. Modality-specific preprocessing (contrast, denoise, resize)
        4. Extract text via OCR if pytesseract is available
        5. Return ProcessedImage with rich metadata
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        enhance_contrast: bool = True,
        run_ocr: bool = True,
        min_image_size: int = 32,           # pixels — reject smaller images
        max_image_bytes: int = 20 * 1024 * 1024,  # 20 MB hard limit
    ):
        self.target_size      = target_size         # overrides modality default if set
        self.enhance_contrast = enhance_contrast
        self.run_ocr          = run_ocr
        self.min_image_size   = min_image_size
        self.max_image_bytes  = max_image_bytes

        # Try to import pytesseract (optional dependency)
        self._ocr_available = False
        if run_ocr:
            try:
                import pytesseract
                self._pytesseract = pytesseract
                self._ocr_available = True
                logger.info("✅ pytesseract available — OCR enabled")
            except ImportError:
                logger.warning("⚠️ pytesseract not installed — OCR disabled. "
                               "Run: pip install pytesseract")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        source: Union[str, Path, bytes, Image.Image],
        modality: Optional[str] = None,
        extra_metadata: Optional[Dict] = None,
    ) -> ProcessedImage:
        """
        Process a single medical image from various input types.

        Args:
            source:         File path, raw bytes, or PIL Image.
            modality:       Force modality; auto-detected if None.
            extra_metadata: Additional metadata to merge (e.g. patient context).

        Returns:
            ProcessedImage with preprocessed image + metadata.
        """
        source_path = None

        # -- Load -------------------------------------------------------
        if isinstance(source, (str, Path)):
            source_path = str(source)
            img = self._load_from_path(Path(source))
            if modality is None:
                modality = self._detect_modality_from_path(Path(source))
        elif isinstance(source, bytes):
            img = self._load_from_bytes(source)
        elif isinstance(source, Image.Image):
            img = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # -- Validate ---------------------------------------------------
        self._validate(img, source_path)

        # -- Detect modality from image content if still unknown --------
        if modality is None:
            modality = self._detect_modality_from_image(img)

        # -- Preprocess -------------------------------------------------
        img = self._preprocess(img, modality)

        # -- Convert to array & bytes -----------------------------------
        arr   = np.array(img)
        b_buf = io.BytesIO()
        img.save(b_buf, format="PNG")
        img_bytes = b_buf.getvalue()

        # -- OCR --------------------------------------------------------
        extracted_text = ""
        if self._ocr_available and modality in ("lab_report", "general"):
            extracted_text = self._run_ocr(img)

        # -- Build metadata ---------------------------------------------
        metadata = self._build_metadata(
            img=img,
            modality=modality,
            source_path=source_path,
            extracted_text=extracted_text,
        )
        if extra_metadata:
            metadata.update(extra_metadata)

        logger.info(
            f"✅ Processed image | modality={modality} | "
            f"size={img.size} | ocr_chars={len(extracted_text)}"
        )

        return ProcessedImage(
            image=img,
            image_array=arr,
            image_bytes=img_bytes,
            source_path=source_path,
            modality=modality,
            metadata=metadata,
            extracted_text=extracted_text,
        )

    def process_batch(
        self,
        sources: List[Union[str, Path, bytes, Image.Image]],
        modality: Optional[str] = None,
    ) -> List[ProcessedImage]:
        """Process a batch of images, skipping failures."""
        results = []
        for i, src in enumerate(sources):
            try:
                results.append(self.process(src, modality=modality))
            except Exception as e:
                logger.error(f"❌ Failed to process image {i}: {e}")
        logger.info(f"📦 Batch processed {len(results)}/{len(sources)} images")
        return results

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_from_path(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )
        return Image.open(path).convert("RGB")

    def _load_from_bytes(self, data: bytes) -> Image.Image:
        if len(data) > self.max_image_bytes:
            raise ValueError(
                f"Image too large: {len(data)/1e6:.1f} MB > "
                f"{self.max_image_bytes/1e6:.0f} MB limit"
            )
        return Image.open(io.BytesIO(data)).convert("RGB")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, img: Image.Image, source_path: Optional[str]):
        w, h = img.size
        if w < self.min_image_size or h < self.min_image_size:
            raise ValueError(
                f"Image too small ({w}×{h}). "
                f"Minimum: {self.min_image_size}px each dimension."
            )

    # ------------------------------------------------------------------
    # Modality detection
    # ------------------------------------------------------------------

    def _detect_modality_from_path(self, path: Path) -> str:
        name = path.stem.lower()
        parent = path.parent.name.lower()
        combined = f"{name} {parent}"
        for modality, hints in MODALITY_HINTS.items():
            if any(h in combined for h in hints):
                return modality
        return "general"

    def _detect_modality_from_image(self, img: Image.Image) -> str:
        """
        Heuristic: X-rays tend to be near-greyscale with low saturation.
        Everything else defaults to 'general'.
        """
        arr = np.array(img.convert("HSV") if hasattr(img, "convert") else img)
        try:
            hsv = np.array(img.convert("HSV"))
            mean_sat = hsv[:, :, 1].mean()
            if mean_sat < 20:          # very low saturation → likely X-ray
                return "xray"
        except Exception:
            pass
        return "general"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: Image.Image, modality: str) -> Image.Image:
        """Apply modality-specific preprocessing pipeline."""
        # 1. Resize
        target = self.target_size or MODALITY_SIZES.get(modality, (448, 448))
        img = img.resize(target, Image.LANCZOS)

        # 2. Modality-specific transforms
        if modality == "xray":
            img = self._preprocess_xray(img)
        elif modality in ("skin", "wound"):
            img = self._preprocess_dermoscopy(img)
        elif modality == "lab_report":
            img = self._preprocess_document(img)
        else:
            if self.enhance_contrast:
                img = ImageEnhance.Contrast(img).enhance(1.2)

        return img

    def _preprocess_xray(self, img: Image.Image) -> Image.Image:
        """Greyscale + CLAHE-approximation via ImageOps equalize."""
        img = img.convert("L")                         # greyscale
        img = ImageOps.equalize(img)                   # histogram equalization
        img = ImageEnhance.Sharpness(img).enhance(1.5) # sharpen edges
        img = img.convert("RGB")                       # back to 3-channel
        return img

    def _preprocess_dermoscopy(self, img: Image.Image) -> Image.Image:
        """Contrast boost + slight denoise for skin/wound photos."""
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Color(img).enhance(1.1)
        img = img.filter(ImageFilter.MedianFilter(size=3))  # remove salt/pepper noise
        return img

    def _preprocess_document(self, img: Image.Image) -> Image.Image:
        """Greyscale + binarize for lab report OCR."""
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.point(lambda p: 255 if p > 140 else 0)  # threshold binarize
        img = img.convert("RGB")
        return img

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _run_ocr(self, img: Image.Image) -> str:
        """Extract text from image using pytesseract."""
        try:
            text = self._pytesseract.image_to_string(img)
            text = text.strip()
            logger.debug(f"   OCR extracted {len(text)} chars")
            return text
        except Exception as e:
            logger.warning(f"⚠️ OCR failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        img: Image.Image,
        modality: str,
        source_path: Optional[str],
        extracted_text: str,
    ) -> Dict:
        w, h = img.size
        risk_level = self._detect_visual_risk(extracted_text)

        return {
            "source":          source_path or "inline_image",
            "modality":        modality,
            "image_width":     w,
            "image_height":    h,
            "has_ocr_text":    len(extracted_text) > 0,
            "ocr_text_length": len(extracted_text),
            "risk_level":      risk_level,
            "domain":          self._modality_to_domain(modality),
            "input_type":      "image",
        }

    def _detect_visual_risk(self, text: str) -> str:
        """Scan OCR text for risk keywords (mirrors knowledge_base logic)."""
        text_lower = text.lower()
        for level in ("emergency", "high", "medium"):
            if any(kw in text_lower for kw in VISUAL_RISK_KEYWORDS[level]):
                return level
        return "low"

    def _modality_to_domain(self, modality: str) -> str:
        mapping = {
            "xray":       "respiratory",
            "skin":       "dermatology",
            "wound":      "general_medicine",
            "lab_report": "general_medicine",
            "general":    "general_medicine",
        }
        return mapping.get(modality, "general_medicine")


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def load_and_process_image(
    path: Union[str, Path],
    modality: Optional[str] = None,
    enhance_contrast: bool = True,
    run_ocr: bool = True,
) -> ProcessedImage:
    """One-liner convenience wrapper."""
    processor = MedicalImageProcessor(
        enhance_contrast=enhance_contrast,
        run_ocr=run_ocr,
    )
    return processor.process(path, modality=modality)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a synthetic test image (white square)
    test_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    processor = MedicalImageProcessor(run_ocr=False)
    result    = processor.process(test_img, modality="general")

    print(f"✅ ProcessedImage shape : {result.image_array.shape}")
    print(f"   modality             : {result.modality}")
    print(f"   metadata             : {result.metadata}")
    print(f"   image_bytes length   : {len(result.image_bytes)} bytes")
