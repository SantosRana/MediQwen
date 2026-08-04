# src/preprocessing/image_processor.py
import os
import io
import base64
import logging
from PIL import Image, ImageOps
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger("image_processor")

@dataclass
class ProcessingResult:
    """Carries processed multi-modal assets and metadata back to the graph state."""
    image_base64: str
    modality: str
    extracted_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalImageProcessor:
    """
    Applies EXIF rotation correction and clean RGB aspect-ratio scaling
    before sending visual frames directly to Ollama.
    """

    def process(self, image_path: str, modality: str = "general") -> ProcessingResult:
        """Loads, corrects orientation, downscales, and encodes image to clean Base64 PNG."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target visual asset could not be located: {image_path}")

        with Image.open(image_path) as pil_img:
            # Log original metadata
            logger.info(f"📸 Original format: {pil_img.format} | mode: {pil_img.mode} | size: {pil_img.size}")

            rgb_img = pil_img.convert("RGB")
            corrected_img = ImageOps.exif_transpose(rgb_img)

            # Preserve details with 1024 max dim across all modalities
            target_max_dim = 1024
            corrected_img.thumbnail((target_max_dim, target_max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            corrected_img.save(buffer, format="PNG")
            b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.info(f"📸 Processed PNG size sent to buffer: {corrected_img.size}")

            return ProcessingResult(
                image_base64=b64_string,
                modality=modality,
                metadata={
                    "input_type": "image",
                    "target_profile_dimensions": f"{corrected_img.size[0]}x{corrected_img.size[1]}",
                    "lossless_format": "png"
                }
            )