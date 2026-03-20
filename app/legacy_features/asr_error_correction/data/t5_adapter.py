import os
import difflib
import torch
from transformers import pipeline
from typing import List

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.legacy_features.asr_error_correction.domain.interfaces import IT5AnomalyAdapter
from app.legacy_features.asr_error_correction.domain.entities import AnomalyTag
from app.legacy_features.asr_error_correction.utils.error_mapper import map_asr_correction_error


class T5AnomalyAdapter(IT5AnomalyAdapter):

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model directory missing: {manifest.model_path}")

            hf_pipeline = pipeline(
                "text2text-generation",
                model=manifest.model_path,
                device=0
            )
            return LoadedModel(manifest=manifest, model_instance=hf_pipeline)

        except Exception as e:
            raise map_asr_correction_error(e, context="Load T5-ASR-EC") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            raise map_asr_correction_error(e, context="Unload T5-ASR-EC") from e

    def detect_anomalies(self, model: LoadedModel, transcript: ReconstructedTranscript) -> List[AnomalyTag]:
        try:
            hf_pipeline = model.model_instance
            anomalies = []

            for sentence in transcript.sentences:
                result = hf_pipeline(sentence.text, max_length=256, truncation=True)
                t5_correction = result[0]['generated_text']

                similarity = difflib.SequenceMatcher(None, sentence.text, t5_correction).ratio()

                if similarity < 0.95:
                    tag = AnomalyTag(
                        sentence_id=sentence.sentence_id,  # Fixed attribute
                        original_start_sec=sentence.start_sec,  # Fixed attribute
                        original_end_sec=sentence.end_sec,  # Fixed attribute
                        original_text=sentence.text,
                        speaker_id=sentence.speaker_id,
                        confidence_of_error=1.0 - similarity,
                        anomaly_reason=f"T5 Phonetic Dissonance Detected. Suggested: {t5_correction}"
                    )
                    anomalies.append(tag)

            return anomalies

        except Exception as e:
            raise map_asr_correction_error(e, context="T5 Anomaly Detection") from e