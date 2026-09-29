# Canvas: test_transcription_api_v6
# Path: tests/features/transcription/service/test_transcription_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.exceptions.model import CTranslateEngineError
from app.features.transcription.exceptions import LanguageMismatchError

API_MODULE = "app.features.transcription.service.transcription_api"


class TestTranscriptionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "vad_repo": MagicMock(),
            "orchestrator": MagicMock(),
            "adapter": MagicMock(**{"detect_language.return_value": ("en", 0.98)})
        }

    @pytest.fixture
    def api(self, mocks):
        return TranscriptionAPI(
            repository=mocks["repo"],
            vad_repository=mocks["vad_repo"],
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"]
        )

    def test_process_audio_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        existing_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].get_by_source_id.return_value = existing_transcript

        result = api.process_audio(source_id, "/dummy.wav")

        assert result == existing_transcript
        mocks["vad_repo"].get_map_by_source_id.assert_not_called()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_fails_without_vad_map(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = None

        empty_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].save.return_value = empty_transcript

        result = api.process_audio(source_id, "/dummy.wav")
        assert result == empty_transcript
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_success_flow_with_hints(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=5.0, confidence=0.99)])
        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        generated_segments = []
        mocks["adapter"].transcribe.return_value = generated_segments

        saved_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].save.return_value = saved_transcript

        result = api.process_audio(
            source_id,
            "/dummy.wav",
            language="fr",
            hotwords="bonjour:15.0",
            initial_prompt="French Context Primer",
            release_after=True
        )

        assert result == saved_transcript
        mocks["orchestrator"].acquire_model.assert_called_once()

        # Verifies domain adaptation payload is passed securely to the whisper engine
        mocks["adapter"].transcribe.assert_called_once_with(
            model=loaded_model_mock,
            audio_path="/dummy.wav",
            vad_map=fake_vad_map,
            language="fr",
            hotwords="bonjour:15.0",
            initial_prompt="French Context Primer"
        )
        mocks["repo"].save.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

    def test_process_audio_removes_text_whisper_copied_from_its_hotwords(self, api, mocks):
        """Whisper sometimes repeats its primed hotwords as if they were spoken; that text must never be saved."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(
            source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=9.0, confidence=0.99)]
        )
        mocks["repo"].save.side_effect = lambda transcript: transcript

        def word(text: str, start: float) -> TranscriptWord:
            return TranscriptWord(start_sec=start, end_sec=start + 0.4, word=text)

        spoken = TranscriptSegment(start_sec=0.0, end_sec=1.6, text="The ECB held rates.",
                                   words=[word(" The", 0.0), word(" ECB", 0.4), word(" held", 0.8), word(" rates.", 1.2)])
        echoed = TranscriptSegment(start_sec=5.0, end_sec=7.0, text="ECB, Lagarde, Frankfurt, Bunds, BOE.",
                                   words=[word(" ECB,", 5.0), word(" Lagarde,", 5.4), word(" Frankfurt,", 5.8),
                                          word(" Bunds,", 6.2), word(" BOE.", 6.6)])
        mocks["adapter"].transcribe.return_value = [spoken, echoed]

        result = api.process_audio(source_id, "/dummy.wav", hotwords="ECB, Lagarde, Frankfurt, Bunds, BOE, Bailey")

        assert [s.text for s in result.segments] == ["The ECB held rates."]
        mocks["repo"].save.assert_called_once_with(result)

    def test_process_audio_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=5.0, confidence=0.99)])
        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        mocks["adapter"].transcribe.side_effect = RuntimeError("cublas failed")

        with pytest.raises(CTranslateEngineError):
            api.process_audio(source_id, "/dummy.wav", release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )
        mocks["repo"].save.assert_not_called()

class TestTranscriptionAPILanguage:
    """The language guard and script normalisation, run as the appliance's configured language."""

    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(**{"get_by_source_id.return_value": None, "save.side_effect": lambda t: t}),
            "vad_repo": MagicMock(),
            "orchestrator": MagicMock(),
            "adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return TranscriptionAPI(mocks["repo"], mocks["vad_repo"], mocks["orchestrator"], mocks["adapter"])

    @pytest.fixture
    def as_language(self, language_pack):
        def activate(code):
            patcher = patch(f"{API_MODULE}.get_language_pack", return_value=language_pack(code))
            patcher.start()
            return patcher
        patchers = []
        yield lambda code: patchers.append(activate(code))
        for patcher in patchers:
            patcher.stop()

    @staticmethod
    def _vad(mocks, source_id):
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=5.0, confidence=0.99)])
        mocks["vad_repo"].get_map_by_source_id.return_value = vad_map
        return vad_map

    def test_rejects_file_confidently_in_another_language_and_releases_model(self, api, mocks, as_language):
        as_language("sr")
        source_id = uuid.uuid4()
        vad_map = self._vad(mocks, source_id)
        loaded = mocks["orchestrator"].acquire_model.return_value
        mocks["adapter"].detect_language.return_value = ("en", 0.97)

        with pytest.raises(LanguageMismatchError, match="Whisper heard 'en'.*Ingest it again with English selected"):
            api.process_audio(source_id, "/a.wav", release_after=True)

        mocks["adapter"].detect_language.assert_called_once_with(loaded, "/a.wav", vad_map)
        mocks["adapter"].transcribe.assert_not_called()
        mocks["repo"].save.assert_not_called()
        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["adapter"], flush=True)

    @pytest.mark.parametrize("detected, hint", [
        # Whisper routinely calls Serbian speech Croatian
        ("hr", "Ingest it again with Serbian selected."),
        # Slovenian's pack is legacy (not served)
        ("sl", "This appliance has no language pack for 'sl'."),
        ("sr", "Ingest it again with Serbian selected."),
        ("de", "This appliance has no language pack for 'de'."),
    ])
    def test_the_refusal_says_which_language_to_ingest_it_as(self, api, mocks, as_language, detected, hint):
        as_language("en")
        self._vad(mocks, uuid.uuid4())
        mocks["adapter"].detect_language.return_value = (detected, 0.95)

        with pytest.raises(LanguageMismatchError) as raised:
            api.process_audio(uuid.uuid4(), "/a.wav", release_after=True)

        assert str(raised.value).endswith(hint)

    @pytest.mark.parametrize("detection", [("hr", 0.9), ("bs", 0.9), ("sr", 0.99), ("en", 0.3)])
    def test_transcribes_accepted_or_unsure_detections(self, api, mocks, as_language, detection):
        """Whisper often calls Serbian Croatian or Bosnian; an unsure guess is never grounds to reject."""
        as_language("sr")
        self._vad(mocks, uuid.uuid4())
        mocks["adapter"].detect_language.return_value = detection
        mocks["adapter"].transcribe.return_value = []

        api.process_audio(uuid.uuid4(), "/a.wav")

        assert mocks["adapter"].transcribe.call_args.kwargs["language"] == "sr"

    def test_no_speech_to_judge_skips_the_guard(self, api, mocks, as_language):
        as_language("en")
        self._vad(mocks, uuid.uuid4())
        mocks["adapter"].detect_language.return_value = None
        mocks["adapter"].transcribe.return_value = []

        api.process_audio(uuid.uuid4(), "/a.wav")

        assert mocks["adapter"].transcribe.call_args.kwargs["language"] == "en"

    def test_serbian_cyrillic_is_saved_in_latin_word_by_word(self, api, mocks, as_language):
        as_language("sr")
        source_id = uuid.uuid4()
        self._vad(mocks, source_id)
        mocks["adapter"].detect_language.return_value = ("sr", 0.99)
        words = [TranscriptWord(start_sec=0.0, end_sec=0.5, word=" Лекар"),
                 TranscriptWord(start_sec=0.5, end_sec=1.0, word=" је"),
                 TranscriptWord(start_sec=1.0, end_sec=1.6, word=" Љубиша.")]
        mocks["adapter"].transcribe.return_value = [
            TranscriptSegment(start_sec=0.0, end_sec=1.6, text="Лекар је Љубиша.", words=words, avg_logprob=-0.2)
        ]

        result = api.process_audio(source_id, "/a.wav")

        segment = result.segments[0]
        assert segment.text == "Lekar je Ljubiša."
        assert [(w.word, w.start_sec, w.end_sec) for w in segment.words] == [
            (" Lekar", 0.0, 0.5), (" je", 0.5, 1.0), (" Ljubiša.", 1.0, 1.6)
        ]
        assert segment.avg_logprob == -0.2

    def test_english_text_is_never_rewritten(self, api, mocks, as_language):
        as_language("en")
        self._vad(mocks, uuid.uuid4())
        mocks["adapter"].detect_language.return_value = ("en", 0.99)
        segment = TranscriptSegment(start_sec=0.0, end_sec=1.0, text="Лекар", words=[])
        mocks["adapter"].transcribe.return_value = [segment]

        result = api.process_audio(uuid.uuid4(), "/a.wav")

        assert result.segments == [segment]
