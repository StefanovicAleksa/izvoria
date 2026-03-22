# Canvas: it_test_adaptation_api_v2
# Path: integration_tests/features/domain_adaptation/test_adaptation_api.py

import pytest
import uuid
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.database.service.manager import db_manager as core_db_manager
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType

from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.data.scout_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.features.vad.domain.entities import VadMap, VadSegment


@pytest.fixture
def setup_finance_glossary():
    """
    Temporarily copies the isolated finance.json from integration test resources
    into the application's live glossary directory, ensuring the test is decoupled
    from future changes to the global application glossaries.
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    resource_path = project_root / "integration_tests" / "resources" / "finance.json"
    target_path = project_root / "data" / "glossaries" / "finance.json"

    assert resource_path.exists(), f"Missing test glossary at {resource_path}"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(resource_path, target_path)

    yield

    if target_path.exists():
        target_path.unlink()


@pytest.fixture
def real_infrastructure():
    source_id = uuid.uuid4()
    file_id = uuid.uuid4()

    with core_db_manager.session() as session:
        test_file = FileModel(
            id=file_id,
            file_path="/tmp/test_options_trading.wav",
            file_name="test_options_trading.wav",
            file_size_bytes=1024,
            mime_type="audio/wav"
        )
        session.add(test_file)

        test_source = SourceModel(
            id=source_id,
            name="test_options_trading.wav",
            source_type=SourceType.AUDIO.value,
            content_hash=f"domain-hash-{source_id}",
            file_id=file_id
        )
        session.add(test_source)

    vram_monitor = PyTorchVramMonitor()
    orchestrator = ModelManagerOrchestrator(vram_monitor)

    yield source_id, orchestrator

    # Teardown
    with core_db_manager.session() as session:
        source = session.query(SourceModel).filter_by(id=source_id).first()
        if source:
            session.delete(source)

        file_record = session.query(FileModel).filter_by(id=file_id).first()
        if file_record:
            session.delete(file_record)


@pytest.mark.integration
def test_full_domain_adaptation_pipeline(setup_finance_glossary, real_infrastructure, monkeypatch):
    source_id, orchestrator = real_infrastructure

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    audio_path = project_root / "integration_tests" / "resources" / "test_options_trading.wav"

    assert audio_path.exists(), f"Test audio missing at {audio_path}. Run generate_test_audio.py first!"

    monkeypatch.setattr(settings, "APPLIANCE_INDUSTRY", "FINANCE")

    repo = DomainAdaptationRepository(core_db_manager)
    scout_adapter = ScoutWhisperAdapter()
    llama_adapter = LlamaAdapter()

    # CRITICAL FIX: Updated to match new Trailer approach arguments
    sampler = SmartSampler(max_speech_per_zone_sec=120.0, min_duration_sec=2.0)

    api = DomainAdaptationAPI(
        repository=repo,
        orchestrator=orchestrator,
        scout_adapter=scout_adapter,
        llama_adapter=llama_adapter,
        smart_sampler=sampler
    )

    vad_map = VadMap(
        source_id=source_id,
        segments=[VadSegment(start_sec=0.0, end_sec=15.0, confidence=0.99)]
    )

    glossary = api.process_domain_adaptation(
        source_id=source_id, audio_path=str(audio_path), vad_map=vad_map
    )

    assert glossary is not None
    assert glossary.source_id == source_id
    assert glossary.industry == "FINANCE"

    assert "OPTIONS_TRADING" in glossary.sub_domains

    assert "strike" in glossary.hotword_string.lower() or "theta" in glossary.hotword_string.lower()
    assert ":15.0" in glossary.hotword_string

    # Assert GlossaryBuilder properly used the prompts.py template
    assert "Options Trading" in glossary.initial_prompt_string
    assert "strike" in glossary.initial_prompt_string.lower()