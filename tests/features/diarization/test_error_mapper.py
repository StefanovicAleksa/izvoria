# Canvas: test_diarization_error_mapper_v1
# Path: tests/features/diarization/test_error_mapper.py

from sqlalchemy.exc import SQLAlchemyError

from app.core.database.exceptions import DatabaseError
from app.features.diarization.utils.error_mapper import map_diarization_error
from app.features.diarization import exceptions as exc


def test_maps_database_errors():
    db_err = DatabaseError("Connection lost")
    sa_err = SQLAlchemyError("Transaction failed")

    assert isinstance(map_diarization_error(db_err), exc.DiarizationDatabaseError)
    assert isinstance(map_diarization_error(sa_err), exc.DiarizationDatabaseError)

def test_maps_audio_segment_errors():
    err = ValueError("Audio is too short for TitaNet minimum duration.")
    mapped = map_diarization_error(err)
    assert isinstance(mapped, exc.DiarizationAudioSegmentError)

def test_maps_spectral_clustering_errors():
    err = Exception("scipy.linalg.LinAlgError: Singular matrix during clustering")
    mapped = map_diarization_error(err)
    assert isinstance(mapped, exc.DiarizationClusteringError)

def test_maps_zero_speakers_error():
    err = Exception("Found 0 speakers in the audio context.")
    mapped = map_diarization_error(err)
    assert isinstance(mapped, exc.DiarizationZeroSpeakersError)

def test_maps_nemo_manifest_errors():
    err = Exception("OmegaConf failed to parse input_manifest.json")
    mapped = map_diarization_error(err)
    assert isinstance(mapped, exc.DiarizationManifestError)

def test_maps_general_tensor_inference_errors():
    err = RuntimeError("CUDA Out of memory allocating tensor")
    mapped = map_diarization_error(err)
    assert isinstance(mapped, exc.DiarizationInferenceError)