import os
import torch
from typing import List
from llama_cpp import Llama

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.features.asr_error_correction.domain.interfaces import IQwenCorrectionAdapter
from app.features.asr_error_correction.domain.entities import (
    DocumentGlossary,
    AnomalyTag,
    DomainCorrectedTranscript,
    CorrectionAuditRecord
)
from app.features.asr_error_correction.data.prompts import PHASE_3_SYSTEM_PROMPT
from app.features.asr_error_correction.utils.error_mapper import map_asr_correction_error


class QwenCorrectionAdapter(IQwenCorrectionAdapter):

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")

            llm = Llama(
                model_path=manifest.model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )
            return LoadedModel(manifest=manifest, model_instance=llm)

        except Exception as e:
            raise map_asr_correction_error(e, context="Load Qwen 2.5 GGUF") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            raise map_asr_correction_error(e, context="Unload Qwen 2.5 GGUF") from e

    def apply_corrections(
            self,
            model: LoadedModel,
            transcript: ReconstructedTranscript,
            anomalies: List[AnomalyTag],
            glossary: DocumentGlossary
    ) -> DomainCorrectedTranscript:
        try:
            llm = model.model_instance

            # Map using the exact UUID
            anomaly_map = {a.sentence_id: a for a in anomalies}
            corrected_sentences = []
            audit_trail = []

            glossary_text = "\n".join([f"- {item.term} ({item.context})" for item in glossary.items])

            for sentence in transcript.sentences:
                # Target exact UUID match
                if sentence.sentence_id not in anomaly_map:
                    corrected_sentences.append(sentence)
                    continue

                anomaly = anomaly_map[sentence.sentence_id]
                prompt = f"{PHASE_3_SYSTEM_PROMPT}\n\nDomain Glossary:\n{glossary_text}\n\nOriginal Text: {sentence.text}\nT5 Flag: {anomaly.anomaly_reason}\n\nCorrected Text:"

                response = llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.2,
                    echo=False
                )

                corrected_text = response["choices"][0]["text"].strip()
                applied_terms = [item.term for item in glossary.items if item.term.lower() in corrected_text.lower()]

                audit_record = CorrectionAuditRecord(
                    anomaly_tag=anomaly,
                    corrected_text=corrected_text,
                    applied_glossary_terms=applied_terms
                )
                audit_trail.append(audit_record)

                new_sentence = sentence.model_copy(update={"text": corrected_text})
                corrected_sentences.append(new_sentence)

            return DomainCorrectedTranscript(
                source_id=transcript.source_id,
                sentences=corrected_sentences,
                audit_trail=audit_trail
            )

        except Exception as e:
            raise map_asr_correction_error(e, context="Qwen 2.5 Grounded Correction") from e