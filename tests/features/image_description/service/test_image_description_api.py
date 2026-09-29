# Path: tests/features/image_description/service/test_image_description_api.py

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.exceptions.vram import OutOfVramError
from app.features.document_conversion.domain.entities import OcrPage
from app.features.image_description.domain.entities import ImageDescription
from app.features.image_description.exceptions import (
    EmptyImageError, ImageReadError, SourceImageMissingError,
)
from app.features.image_description.service.image_description_api import ImageDescriptionAPI
from app.features.image_description.utils.text_verification import UNCONFIRMED_NUMBER
from app.features.visual_analysis.domain.entities import FrameObservation

OBSERVATION = FrameObservation(shows="A whiteboard, room 204 and room 999", legible_text=["Q3 launch", "Go live 2026"],
                               marks="an arrow at Q3")
OCR = OcrPage(text="Q3 LAUNCH\nRoom 204", mean_confidence=0.93, lines_detected=3, lines_read=2)


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "board.png"
    Image.new("RGB", (64, 48), "white").save(path)
    return path


@pytest.fixture
def mocks(image_file):
    repo = MagicMock()
    repo.get_by_source_id.return_value = None
    repo.save.side_effect = lambda description: description
    storage = MagicMock()
    storage.get_source.return_value.file.file_path = str(image_file)
    visual = MagicMock()
    visual.describe_image.return_value = OBSERVATION
    conversion = MagicMock()
    conversion.read_image.return_value = OCR
    return {"repo": repo, "storage": storage, "visual": visual, "conversion": conversion}


@pytest.fixture
def api(mocks):
    return ImageDescriptionAPI(repository=mocks["repo"], storage_api=mocks["storage"], visual_api=mocks["visual"],
                               conversion_api=mocks["conversion"])


class TestExecute:
    def test_vision_text_is_checked_against_ocr(self, api, mocks, image_file):
        model = MagicMock(spec=LoadedModel)
        source_id = uuid.uuid4()

        description = api.execute(source_id, model)

        mocks["visual"].describe_image.assert_called_once_with(image_file, model)
        [(path, select_lines)] = [c.args for c in mocks["conversion"].read_image.call_args_list]
        assert path == image_file
        # A low-confidence OCR line stands only if the vision model read it too
        assert select_lines([("Q3 launch", 0.5), ("xqzt vrrm", 0.5), ("Room 204", 0.99)]) == [True, False, True]
        assert description.source_id == source_id and description.file_name == "board.png"
        assert (description.width, description.height) == (64, 48)
        assert description.ocr_checked and description.ocr_text == "Q3 LAUNCH\nRoom 204"
        assert description.verified_items == ["Q3 launch"]
        assert description.dropped_items == ["Go live 2026"]
        assert description.unverified_items == []
        assert description.shows == f"A whiteboard, room 204 and room {UNCONFIRMED_NUMBER}"
        assert description.ocr_incomplete
        mocks["repo"].save.assert_called_once_with(description)

    def test_without_ocr_the_vision_text_is_kept_unverified(self, api, mocks):
        mocks["conversion"].read_image.return_value = None

        description = api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))

        assert not description.ocr_checked
        assert description.unverified_items == ["Q3 launch", "Go live 2026"]
        assert description.dropped_items == [] and description.verified_items == []
        assert description.shows == OBSERVATION.shows

    def test_ocr_that_read_nothing_drops_every_vision_text_item(self, api, mocks):
        mocks["conversion"].read_image.return_value = OcrPage()

        description = api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))

        assert description.dropped_items == ["Q3 launch", "Go live 2026"]
        assert description.ocr_checked and description.ocr_text == ""

    def test_text_only_image_without_a_description_is_indexed(self, api, mocks):
        mocks["visual"].describe_image.return_value = None

        description = api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))

        assert description.shows == "" and description.ocr_text == OCR.text

    def test_nothing_recognised_raises(self, api, mocks):
        mocks["visual"].describe_image.return_value = None
        mocks["conversion"].read_image.return_value = OcrPage()

        with pytest.raises(EmptyImageError):
            api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))
        mocks["repo"].save.assert_not_called()

    def test_existing_description_is_returned_without_the_model(self, api, mocks):
        existing = ImageDescription(source_id=uuid.uuid4(), file_name="x.jpg", shows="old")
        mocks["repo"].get_by_source_id.return_value = existing

        assert api.execute(existing.source_id, MagicMock(spec=LoadedModel)) is existing
        mocks["visual"].describe_image.assert_not_called()
        mocks["conversion"].read_image.assert_not_called()

    def test_missing_vault_file_raises(self, api, mocks, tmp_path):
        mocks["storage"].get_source.return_value.file.file_path = str(tmp_path / "gone.jpg")

        with pytest.raises(SourceImageMissingError):
            api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))

    def test_file_that_is_not_an_image_raises_before_the_model_runs(self, api, mocks, tmp_path):
        bogus = tmp_path / "fake.jpg"
        bogus.write_bytes(b"not an image")
        mocks["storage"].get_source.return_value.file.file_path = str(bogus)

        with pytest.raises(ImageReadError):
            api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))
        mocks["visual"].describe_image.assert_not_called()

    def test_model_manager_errors_are_not_wrapped(self, api, mocks):
        mocks["visual"].describe_image.side_effect = OutOfVramError("no room")

        with pytest.raises(OutOfVramError):
            api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))


    def test_ocr_lines_withheld_by_the_selection_are_recorded_not_indexed(self, api, mocks):
        mocks["conversion"].read_image.return_value = OcrPage(
            text="Q3 LAUNCH", lines_detected=2, lines_read=2, withheld_lines=["Analbeig aftea long apok"])

        description = api.execute(uuid.uuid4(), MagicMock(spec=LoadedModel))

        assert description.ocr_text == "Q3 LAUNCH"
        assert description.ocr_withheld_lines == ["Analbeig aftea long apok"]


class TestRenderAndRelease:
    def test_render_uses_the_active_packs_labels(self, api, appliance_language):
        appliance_language("sr")
        description = ImageDescription(source_id=uuid.uuid4(), file_name="x.jpg", shows="a dog")

        assert api.render(description) == "Slika (mašinski opis): A dog."

    def test_release_drops_the_ocr_models(self, api, mocks):
        api.release()

        mocks["conversion"].release.assert_called_once_with()
