# Canvas: it_test_adaptation_real_audio_v5
# Path: integration_tests/features/domain_adaptation/test_adaptation_real_audio.py

import pytest
import uuid
from pathlib import Path

from app.core.database.service.manager import db_manager as core_db_manager
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType

from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.data.scout_whisper_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.data.deberta_adapter import DebertaClassifierAdapter
from app.features.domain_adaptation.utils.audio_sampler import AudioSampler
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.features.vad.domain.entities import VadMap, VadSegment


@pytest.fixture
def real_infrastructure():
    source_id = uuid.uuid4()
    file_id = uuid.uuid4()

    with core_db_manager.session() as session:
        # 1. Create the physical file record first
        test_file = FileModel(
            id=file_id,
            file_path="/tmp/test_options_trading.wav",
            file_name="test_options_trading.wav",
            file_size_bytes=1024,
            mime_type="audio/wav"
        )
        session.add(test_file)

        # 2. Create the Source record linking to the file
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
def test_full_domain_adaptation_pipeline(real_infrastructure, monkeypatch):
    source_id, orchestrator = real_infrastructure

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    audio_path = project_root / "integration_tests" / "resources" / "test_options_trading.wav"

    assert audio_path.exists(), f"Test audio missing at {audio_path}. Run generate_test_audio.py first!"

    monkeypatch.setenv("APPLIANCE_INDUSTRY", "FINANCE")

    repo = DomainAdaptationRepository(core_db_manager)
    scout_adapter = ScoutWhisperAdapter()
    classifier_adapter = DebertaClassifierAdapter()
    sampler = AudioSampler(slice_duration_sec=20.0)

    api = DomainAdaptationAPI(repo, orchestrator, scout_adapter, classifier_adapter, sampler)

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
    assert glossary.sub_domain == "OPTIONS_TRADING"
    assert "strike" in glossary.hotword_string.lower() or "theta" in glossary.hotword_string.lower()
    assert ":15.0" in glossary.hotword_string