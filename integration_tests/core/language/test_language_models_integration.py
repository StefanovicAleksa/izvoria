# Path: integration_tests/core/language/test_language_models_integration.py

import os
import uuid

import pytest

from app.core.config import Settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate
from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter

SERBIAN = Settings(APPLIANCE_LANGUAGE="sr")


@pytest.fixture(scope="module", autouse=True)
def gemma_weights():
    if not os.path.exists(SERBIAN.CONTEXTUAL_RETRIEVAL_MODEL_PATH):
        pytest.skip(f"Gemma 4 E4B weights not found at {SERBIAN.CONTEXTUAL_RETRIEVAL_MODEL_PATH}.")


@pytest.fixture
def serbian(monkeypatch):
    """This test process runs as a Serbian appliance (prompts, refusal and output language)."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "APPLIANCE_LANGUAGE", "sr")


def _run(model_type, path, adapter, work):
    """Loads `path` through the adapter the way the appliance does, runs `work`, and always releases it."""
    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    loaded = orchestrator.acquire_model(ModelManifest(model_type=model_type, model_path=path, required_vram_mb=5500),
                                        adapter)
    try:
        return loaded, work(loaded)
    finally:
        orchestrator.release_model(loaded, adapter)


@pytest.mark.integration
class TestSerbianSmallModels:
    """
    Gemma 4 E4B replaces Llama 3.2 3B and Qwen2.5-3B on Serbian appliances. Each adapter it replaces
    them in must load it (chat template, GBNF grammar, context size) and do its job on Serbian text. One model is
    loaded at a time, as on the appliance.
    """

    def test_contextual_retrieval_fits_its_full_context_and_writes_serbian(self, serbian):
        document = ("[Speaker_0]: Dobar dan, ja sam doktor Petrović. Pregledamo pacijenta sa akutnim infarktom. "
                    "[Speaker_1]: Primljen je juče sa jakim bolom u grudima. EKG pokazuje elevaciju ST segmenta. ") * 30

        _, context = _run(ModelType.CONTEXTUAL_RETRIEVAL, SERBIAN.CONTEXTUAL_RETRIEVAL_MODEL_PATH,
                               SLMContextAdapter(),
                               lambda m: SLMContextAdapter().generate_context(m, document, "[Speaker_1]: EKG pokazuje elevaciju."))

        assert context.strip()
        assert not any("Ѐ" <= ch <= "ӿ" for ch in context), "context must be Latin, like the transcript"

    def test_domain_classifier_scores_serbian_speech_under_its_grammar(self, serbian):
        scout = ("Pacijent je primljen sa bolom u grudima. EKG pokazuje elevaciju ST segmenta, troponin je povišen. "
                 "Ugradili smo stent u desnu koronarnu arteriju.")
        labels = {"CARDIOLOGY": "infarkt, stent, EKG, troponin", "ONCOLOGY": "tumor, hemoterapija, metastaze"}

        _, picked = _run(ModelType.DOMAIN_CLASSIFIER, SERBIAN.DOMAIN_CLASSIFIER_MODEL_PATH, LlamaAdapter(),
                         lambda m: LlamaAdapter().predict_subdomains(m, scout, labels))

        assert picked == ["CARDIOLOGY"]

    def test_phonetic_verifier_accepts_a_serbian_mishearing(self, serbian):
        sentence = "Ugradili smo stent zbog akutnog infarkd miokarda."
        start = sentence.index("infarkd miokarda")
        candidate = PhoneticCandidate(start=start, end=start + 16, heard="infarkd miokarda", term="infarkt miokarda",
                                      replacement="infarkt miokarda", score=0.9, min_probability=0.4)

        _, verdicts = _run(ModelType.DOMAIN_PHONETIC_CORRECTION, SERBIAN.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
                           QwenDomainAdapter(),
                           lambda m: QwenDomainAdapter().verify_candidates(m, sentence, "Pacijent je primljen juče.",
                                                                          [candidate], "kardiologija"))

        assert verdicts == [True]

    def test_visual_checker_and_synthesis_run_on_serbian_speech(self, serbian):
        _, audit = _run(ModelType.VISUAL_HALLUCINATION_CORRECTION, SERBIAN.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH,
                        QwenHallucinationAdapter(),
                        lambda m: QwenHallucinationAdapter().evaluate_visual_match(
                            m, uuid.uuid4(), "[Speaker_0]: Na snimku vidite EKG sa elevacijom ST segmenta.",
                            'An ECG strip; legible text: "ST elevation V2-V4".'))
        _, narrative = _run(ModelType.VISUAL_CONTEXT_SYNTHESIS, SERBIAN.VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH,
                            VisualSLMAdapter(),
                            lambda m: VisualSLMAdapter().synthesize_context(m, [{
                                "start_sec": 0.0, "end_sec": 5.0, "frame_ids": [],
                                "representative_description": 'An ECG strip with the text "ST elevation V2-V4".'}]))

        assert audit.is_visual_accepted is True
        assert narrative.strip()
