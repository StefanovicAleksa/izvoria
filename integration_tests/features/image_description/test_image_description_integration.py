# Path: integration_tests/features/image_description/test_image_description_integration.py

"""
Images through the real adapters and Postgres: RapidOCR reads a generated notice, the vision model's text is
checked against it, and the description round-trips through `image_descriptions`. The OCR check runs with a
stand-in vision answer (one real item, one invented), so it is exact and needs no GPU; a second class runs the
real Qwen2-VL on the same image and on the X-ray fixture, and skips without its weights.
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.document_conversion.data.rapidocr_engine import (
    CLASSIFIER_FILE, DETECTOR_FILE, RECOGNIZER_FILES, RapidOcrEngine,
)
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI
from app.features.image_description.data.repositories import ImageDescriptionRepository
from app.features.image_description.service.image_description_api import ImageDescriptionAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.domain.entities import FrameObservation
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI

pytestmark = pytest.mark.integration

OCR_DIR = settings.BASE_DIR / settings.DOC_OCR_MODEL_DIR
FONT = Path("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf")
XRAY = Path(__file__).resolve().parents[2] / "resources" / "test_medical_xray.jpg"
NOTICE_LINES = ["BUILDING MAINTENANCE NOTICE", "Water shut off on 14 March 2025", "Floors 2 to 5, from 09:00",
                "Contact: facilities desk, room 204"]


def _ocr_models_present() -> bool:
    return all((OCR_DIR / name).is_file() for name in (DETECTOR_FILE, CLASSIFIER_FILE, RECOGNIZER_FILES["latin"]))


def _notice(path: Path) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 1560, 860), outline="black", width=6)
    for i, line in enumerate(NOTICE_LINES):
        draw.text((120, 140 + i * 170), line, font=ImageFont.truetype(str(FONT), 64 if i == 0 else 52), fill="black")
    image.save(path)


@pytest.fixture
def storage_api(db_manager):
    return StorageAPI(StorageRepository(db_manager))


@pytest.fixture
def notice(tmp_path, storage_api):
    if not FONT.is_file():
        pytest.skip("Noto Serif is needed to render the synthetic notice")
    path = tmp_path / "notice.png"
    _notice(path)
    return storage_api.ingest_local_file(str(path), SourceType.IMAGE)


@pytest.fixture
def conversion_api(db_manager, storage_api):
    if not _ocr_models_present():
        pytest.skip(f"RapidOCR models not in {OCR_DIR}; run `python -m scripts.download_ocr_models`")
    api = DocumentConversionAPI(MagicMock(), storage_api, MagicMock(), MagicMock(), RapidOcrEngine(str(OCR_DIR), "latin"))
    yield api
    api.release()


class TestOcrCheckIntegration:
    def test_only_text_ocr_also_reads_is_kept_and_it_round_trips(self, db_manager, storage_api, conversion_api,
                                                                 notice):
        visual = MagicMock()
        visual.describe_image.return_value = FrameObservation(
            shows="A printed notice in a black frame, dated 21 April 2025",
            legible_text=["Water shut off on 14 March 2025", "Elevator closed until 30 June"],
        )
        repo = ImageDescriptionRepository(db_manager)
        api = ImageDescriptionAPI(repo, storage_api, visual, conversion_api)

        description = api.execute(notice.id, MagicMock())

        assert description.ocr_checked and description.ocr_lines_read == description.ocr_lines_detected
        assert "Water shut off on 14 March 2025" in description.ocr_text
        assert description.verified_items == ["Water shut off on 14 March 2025"]
        assert description.dropped_items == ["Elevator closed until 30 June"]
        # A date the notice doesn't show is marked in the description, not repeated
        assert "21 April" not in description.shows and "[number not confirmed]" in description.shows
        assert (description.width, description.height) == (1600, 900)

        assert repo.get_by_source_id(notice.id) == description
        assert api.execute(notice.id, MagicMock()).id == description.id
        excerpt = api.render(description)
        assert excerpt.startswith("Image (machine description): ")
        assert "Text on the image (read by OCR):\n" in excerpt and "Elevator" not in excerpt


@pytest.fixture(scope="module")
def vlm():
    if not Path(settings.VISION_MODEL_PATH).exists():
        pytest.skip(f"Qwen2-VL weights missing at {settings.VISION_MODEL_PATH}")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Qwen2-VL needs a CUDA GPU")
    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    adapter = QwenVLMAdapter()
    visual_api = VisualAnalysisAPI(repository=MagicMock(), vlm_adapter=adapter, storage_api=MagicMock())
    model = orchestrator.acquire_model(visual_api.manifest, adapter)
    try:
        yield visual_api, model
    finally:
        orchestrator.release_model(model, adapter, flush=True)


class TestRealVisionModelIntegration:
    def test_the_notice_is_described_and_its_kept_text_is_on_the_image(self, vlm, db_manager, storage_api,
                                                                       conversion_api, notice):
        visual_api, model = vlm
        api = ImageDescriptionAPI(ImageDescriptionRepository(db_manager), storage_api, visual_api, conversion_api)

        description = api.execute(notice.id, model)

        assert description.shows, "the vision model described nothing"
        assert "notice" in (description.shows + " ".join(description.verified_items)).lower()
        for item in description.verified_items:
            assert any(word.lower() in description.ocr_text.lower() for word in item.split())

    def test_the_xray_is_described_without_a_diagnosis(self, vlm):
        if not XRAY.is_file():
            pytest.skip(f"X-ray fixture missing at {XRAY}")
        visual_api, model = vlm

        observation = visual_api.describe_image(XRAY, model)

        assert observation is not None and observation.shows
        assert "x-ray" in observation.shows.lower() or "radiograph" in observation.shows.lower()
