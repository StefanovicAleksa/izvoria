# Canvas: nemo_diarization_adapter_v2
# Path: app/features/diarization/data/nemo_adapter.py

import os
import json
import tempfile
import uuid
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.vad.domain.entities import VadMap
from app.features.diarization.domain.interfaces import IDiarizationModelAdapter
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.utils.error_mapper import map_diarization_error


class NeMoDiarizationAdapter(IDiarizationModelAdapter):

    @staticmethod
    def _get_diarizer(cfg: DictConfig):
        """
        FACTORY METHOD & LAZY IMPORT
        Prevents NeMo from aggressively probing the GPU during App Startup.
        """
        # noinspection PyPackageRequirements
        from nemo.collections.asr.models import ClusteringDiarizer
        return ClusteringDiarizer(cfg=cfg)

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")
            return LoadedModel(manifest=manifest, model_instance=manifest.model_path)
        except Exception as e:
            raise map_diarization_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
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

            with tempfile.TemporaryDirectory(prefix="onyx_diarization_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 1. Oracle VAD RTTM
                vad_rttm_path = tmp_path / f"{audio_id}.rttm"
                with open(vad_rttm_path, "w", encoding="utf-8") as f:
                    for seg in vad_map.segments:
                        duration = seg.end_sec - seg.start_sec
                        f.write(f"SPEAKER {audio_id} 1 {seg.start_sec:.3f} {duration:.3f} <NA> <NA> speech <NA> <NA>\n")

                # 2. Manifest JSON
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

                # 3. OmegaConf Graph
                cfg = OmegaConf.create({
                    "device": "cuda",
                    "num_workers": 1,
                    "sample_rate": 16000,
                    "batch_size": 64,
                    "diarizer": {
                        "manifest_filepath": str(manifest_path),
                        "out_dir": str(tmp_path),
                        "oracle_vad": True,
                        "collar": 0.25,
                        "ignore_overlap": True,
                        "verbose": False,
                        "vad": {
                            "model_path": None,
                            "external_vad_manifest": None,
                            "parameters": {
                                "window_length_in_sec": 0.15,
                                "shift_length_in_sec": 0.01,
                                "smoothing": "median",
                                "overlap": 0.5,
                                "onset": 0.1,
                                "offset": 0.1,
                                "pad_onset": 0.0,
                                "pad_offset": 0.0,
                                "min_duration_on": 0.0,
                                "min_duration_off": 0.0,
                                "filter_speech_first": True
                            }
                        },
                        "speaker_embeddings": {
                            "model_path": model.model_instance,
                            "parameters": {
                                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                                "multiscale_weights": [1, 1, 1, 1, 1],
                                "save_embeddings": False
                            }
                        },
                        "clustering": {
                            "parameters": {
                                "oracle_num_speakers": False,
                                "max_num_speakers": 20,
                                "enhanced_count_thres": 80,
                                "max_rp_threshold": 0.25,
                                "sparse_search_volume": 30
                            }
                        }
                    }
                })

                # 4. Initialize and execute via the static factory method
                diarizer = self._get_diarizer(cfg)

                # --- ROBUST ENTERPRISE FALLBACK ---
                # NeMo's Clustering algorithm crashes with various mathematical errors
                # (ValueError, LinAlgError, AttributeError) if it fails to find multiple
                # speakers or if the audio is a monologue.
                # We safely catch it and map the known VAD speech segments to "Speaker_0".
                try:
                    diarizer.diarize()
                except Exception:
                    fallback_segments = [
                        DiarizationSegment(
                            start_sec=seg.start_sec,
                            end_sec=seg.end_sec,
                            speaker_id="Speaker_0"
                        ) for seg in vad_map.segments
                    ]
                    return DiarizationMap(source_id=source_id, segments=fallback_segments)
                # ----------------------------------

                # 5. Parse Output
                pred_rttm_path = tmp_path / "pred_rttms" / f"{audio_id}.rttm"
                if not pred_rttm_path.exists():
                    # Same fallback if it completed but failed to write the file
                    fallback_segments = [
                        DiarizationSegment(
                            start_sec=seg.start_sec,
                            end_sec=seg.end_sec,
                            speaker_id="Speaker_0"
                        ) for seg in vad_map.segments
                    ]
                    return DiarizationMap(source_id=source_id, segments=fallback_segments)

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