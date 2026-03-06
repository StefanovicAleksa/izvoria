import os
import json
import torch
from llama_cpp import Llama

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.features.asr_error_correction.domain.interfaces import ILlamaGlossaryAdapter
from app.features.asr_error_correction.domain.entities import DocumentGlossary, GlossaryItem
from app.features.asr_error_correction.data.prompts import PHASE_1_SYSTEM_PROMPT
from app.features.asr_error_correction.utils.error_mapper import map_asr_correction_error


class LlamaGlossaryAdapter(ILlamaGlossaryAdapter):

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")

            # Note: chat_format="chatml" or standard Llama 3 format ensures chat_completion works
            llm = Llama(
                model_path=manifest.model_path,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False
            )
            return LoadedModel(manifest=manifest, model_instance=llm)

        except Exception as e:
            raise map_asr_correction_error(e, context="Load Llama 3.2 GGUF") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            raise map_asr_correction_error(e, context="Unload Llama 3.2 GGUF") from e

    def extract_glossary(self, model: LoadedModel, transcript: ReconstructedTranscript) -> DocumentGlossary:
        try:
            llm = model.model_instance
            full_text = " ".join([s.text for s in transcript.sentences])

            # Using Chat Completion to enforce JSON output natively at the C++ level
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": PHASE_1_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract glossary from this transcript:\n\n{full_text}"}
                ],
                response_format={
                    "type": "json_object",
                    # Optionally, you can pass a full JSON schema here for strict key enforcement
                },
                max_tokens=1024,
                temperature=0.1,
            )

            raw_output = response["choices"][0]["message"]["content"].strip()

            # No regex needed; llama.cpp guarantees this parses.
            parsed_json = json.loads(raw_output)

            items = [
                GlossaryItem(term=item["term"], context=item["context"])
                for item in parsed_json.get("items", [])
            ]

            return DocumentGlossary(
                source_id=transcript.source_id,
                domain_category=parsed_json.get("domain_category", "Unknown"),
                items=items
            )

        except Exception as e:
            raise map_asr_correction_error(e, context="Llama 3.2 Glossary Extraction") from e