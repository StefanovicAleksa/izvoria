# Canvas: domain_qwen_adapter_v21
# Path: app/features/domain_phonetic_correction/data/qwen_adapter.py

import os
import gc
from typing import List, Any, Dict
import logging

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticPatchAdapter
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticPatch
from app.features.domain_phonetic_correction.data.prompts import QWEN_PHONETIC_SYSTEM_PROMPT
from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error
from app.features.domain_phonetic_correction.utils.xml_parser import parse_xml_patches

logger = logging.getLogger(__name__)


class QwenDomainAdapter(IDomainPhoneticPatchAdapter):
    @staticmethod
    def _get_qwen_model(model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")
            llm = self._get_qwen_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_phonetic_error(e, context=f"Load Qwen 2.5 Model: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_phonetic_error(e, context="Unload Qwen 2.5 Model") from e

    def generate_patches(
            self,
            model: LoadedModel,
            text_chunk: str,
            domain_dictionary: str
    ) -> List[DomainPhoneticPatch]:
        try:
            llm: Any = model.model_instance
            system_content = QWEN_PHONETIC_SYSTEM_PROMPT.format(domain_dictionary=domain_dictionary)

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"<INPUT_TEXT>\n{text_chunk}\n</INPUT_TEXT>"}
            ]

            # FIX: Removed the buggy continuation loop.
            # Gave the LLM a healthy max_tokens of 800.
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )

            choice = response["choices"][0]
            final_raw_output = choice["message"].get("content", "")
            finish_reason = choice.get("finish_reason", "stop")

            if finish_reason == "length":
                logger.warning(
                    "LLM hit max_tokens limit. Regex Parser will salvage completed patches and ignore the truncated tail.")

            # Strip C++ null terminators safely
            final_raw_output = final_raw_output.replace('\x00', '')

            return parse_xml_patches(final_raw_output)

        except Exception as e:
            raise map_phonetic_error(e, context="Generate Qwen XML Patches") from e