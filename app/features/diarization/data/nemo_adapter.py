# Canvas: nemo_diarization_adapter_v1
# Path: app/features/diarization/data/nemo_adapter.py

import os
import json
import tempfile
import uuid
from pathlib import Path

from omegaconf import OmegaConf
# noinspection PyPackageRequirements
from nemo.collections.asr.models import ClusteringDiarizer

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.vad.domain.entities import VadMap
from app.features.diarization.domain.interfaces import IDiarizationModelAdapter
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.utils.error_mapper import map_diarization_error


class NeMoDiarizationAdapter(IDiarizationModelAdapter):
    """
    Adapter for NVIDIA NeMo's ClusteringDiarizer (TitaNet + Spectral Clustering).
    Uses 'Oracle VAD' to inject our pre-computed VAD Map, saving significant compute time.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        """
        Validates the embedding model exists.
        Note: ClusteringDiarizer instantiates during the inference call,
        so we just wrap the validated path here.
        """
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")

            return LoadedModel(manifest=manifest, model_instance=manifest.model_path)
        except Exception as e:
            raise map_diarization_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        """Clears the reference to allow Python GC to clean up."""
        try:
            del model.model_instance
            return True
        except Exception as e:
            raise map_diarization_error(e) from e

    def generate_diarization_map(
            self,
            model: LoadedModel,
            audio_path: str,
            vad_map: VadMap,
            source_id: uuid.UUID
    ) -> DiarizationMap:
        try:
            audio_path_obj = Path(audio_path).resolve()
            audio_id = audio_path_obj.stem

            # 1. Create a secure temporary directory for NeMo's mandatory I/O operations
            with tempfile.TemporaryDirectory(prefix="onyx_diarization_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 2. Convert our VadMap into an RTTM file (The "Oracle VAD" input)
                vad_rttm_path = tmp_path / f"{audio_id}.rttm"
                with open(vad_rttm_path, "w", encoding="utf-8") as f:
                    for seg in vad_map.segments:
                        duration = seg.end_sec - seg.start_sec
                        # Standard RTTM format: SPEAKER {file_id} 1 {start} {duration} <NA> <NA> {speaker} <NA> <NA>
                        f.write(f"SPEAKER {audio_id} 1 {seg.start_sec:.3f} {duration:.3f} <NA> <NA> speech <NA> <NA>\n")

                # 3. Create the input manifest JSON expected by NeMo
                manifest_path = tmp_path / "input_manifest.json"
                manifest_data = {
                    "audio_filepath": str(audio_path_obj),
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": str(vad_rttm_path)
                }
                with open(manifest_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(manifest_data) + "\n")

                # 4. Construct the OmegaConf Graph for the ClusteringDiarizer
                cfg = OmegaConf.create({
                    "diarizer": {
                        "manifest_filepath": str(manifest_path),  #
                        "out_dir": str(tmp_path),  #
                        "oracle_vad": True,
                        # If True, uses RTTM files provided in manifest file to get speech activity timestamps
                        "collar": 0.25,  #
                        "ignore_overlap": True,  #
                        "speaker_embeddings": {
                            "model_path": model.model_instance,  # .nemo local model path
                            "parameters": {
                                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],  #
                                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],  #
                                "multiscale_weights": [1, 1, 1, 1, 1],  #
                                "save_embeddings": False  #
                            }
                        },
                        "clustering": {
                            "parameters": {
                                "oracle_num_speakers": False,  #
                                "max_num_speakers": 20,  #
                                "enhanced_count_thres": 80,  #
                                "max_rp_threshold": 0.25,  #
                                "sparse_search_volume": 30  #
                            }
                        }
                    }
                })

                # 5. Initialize and execute the Diarizer
                diarizer = ClusteringDiarizer(cfg=cfg)
                diarizer.diarize()

                # 6. Parse the generated output RTTM file
                pred_rttm_path = tmp_path / "pred_rttms" / f"{audio_id}.rttm"
                if not pred_rttm_path.exists():
                    raise FileNotFoundError(f"NeMo failed to produce an output RTTM file at {pred_rttm_path}")

                diarization_segments = []
                with open(pred_rttm_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == "SPEAKER":
                            start_sec = float(parts[3])
                            duration = float(parts[4])
                            speaker_id = parts[7]

                            diarization_segments.append(DiarizationSegment(
                                start_sec=start_sec,
                                end_sec=start_sec + duration,
                                speaker_id=speaker_id
                            ))

                return DiarizationMap(source_id=source_id, segments=diarization_segments)

        except Exception as e:
            raise map_diarization_error(e) from e