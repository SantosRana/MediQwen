# tests/unit/test_image_processor.py
import pytest
import numpy as np
from PIL import Image
from src.preprocessing.image_processor import MedicalImageProcessor

@pytest.fixture
def synthetic_highres_skin_photo(tmp_path):
    """Generates a high-res random pixel square layout to test resizing pipelines."""
    img_path = tmp_path / "raw_skin_lesion.png"
    # Emulate a large 1500x1500px phone camera frame matrix
    raw_array = np.uint8(np.random.randint(0, 255, (1500, 1500, 3)))
    Image.fromarray(raw_array).save(img_path)
    return str(img_path)

def test_image_processor_normalizes_and_encodes(synthetic_highres_skin_photo):
    """Validates that a raw image is downsized to modality specs and base64 encoded cleanly."""
    processor = MedicalImageProcessor(run_ocr=False)
    result = processor.process(synthetic_highres_skin_photo)
    
    # Verify name keyword pattern mapping ("skin" hint forces 384x384 target profile)
    assert result.modality == "skin"
    assert result.image.size == (384, 384)
    assert result.image_array.shape == (384, 384, 3)
    
    # Verify Ollama base64 image string is formatted successfully
    assert isinstance(result.image_base64, str)
    assert len(result.image_base64) > 0
    assert result.metadata["input_type"] == "image"