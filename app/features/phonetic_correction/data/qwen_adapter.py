# Canvas: qwen_adapter_v5
# Path: app/features/phonetic_correction/data/qwen_adapter.py

import os
import json
import gc
from typing import List, Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.phonetic_correction.domain.interfaces import IPhoneticPatchAdapter
from app.features.phonetic_correction.domain.entities import PhoneticPatch
from app.features.phonetic_correction.data.prompts import QWEN_PHONETIC_SYSTEM_PROMPT
from app.features.phonetic_correction.utils.error_mapper import map_phonetic_error


class QwenAdapter(IPhoneticPatchAdapter):
    """
    Infrastructure adapter for Qwen 2.5 (GGUF).
    Executes the Read-Only JSON Patch strategy with strict DLL isolation.
    """

    @staticmethod
    def _get_qwen_model(model_path: str) -> Any:
        """FACTORY METHOD: Prevents llama-cpp-python CUDA collisions with PyTorch."""
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
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
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            return True
        except Exception as e:
            raise map_phonetic_error(e, context="Unload Qwen 2.5 Model") from e

    def generate_patches(
            self,
            model: LoadedModel,
            text_chunk: str,
            domain_dictionary: str
    ) -> List[PhoneticPatch]:
        try:
            # Lazy load the TypedDicts to keep the global namespace clean
            from llama_cpp import (
                ChatCompletionRequestSystemMessage,
                ChatCompletionRequestUserMessage,
                ChatCompletionRequestResponseFormat
            )

            llm: Any = model.model_instance
            system_content = QWEN_PHONETIC_SYSTEM_PROMPT.format(domain_dictionary=domain_dictionary)

            system_msg = ChatCompletionRequestSystemMessage(role="system", content=system_content)
            user_msg = ChatCompletionRequestUserMessage(role="user",
                                                        content=f"<INPUT_TEXT>\n{text_chunk}\n</INPUT_TEXT>")
            response_format = ChatCompletionRequestResponseFormat(type="json_object")

            # Force Qwen to output strict JSON using llama.cpp's response_format
            response = llm.create_chat_completion(
                messages=[system_msg, user_msg],
                response_format=response_format,
                temperature=0.0,
                max_tokens=2048  # CRITICAL FIX: Increased to 2048 to prevent mid-JSON cutoff
            )

            raw_output = response["choices"][0]["message"]["content"].strip()

            # Clean markdown code blocks just in case
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]

            parsed_json = json.loads(raw_output)
            raw_corrections = parsed_json.get("corrections", [])

            # Safety check if the LLM hallucinated a dict instead of a list
            if isinstance(raw_corrections, dict):
                raw_corrections = [raw_corrections]
            elif not isinstance(raw_corrections, list):
                raw_corrections = []

            patches: List[PhoneticPatch] = []
            for item in raw_corrections:
                # Type guard to ensure we are looking at a valid object
                if isinstance(item, dict) and "original_text" in item and "corrected_text" in item:
                    orig = item["original_text"]
                    corr = item["corrected_text"]

                    # ==========================================================
                    # CRITICAL FIX: The "Random 0" Sanitizer
                    # ==========================================================
                    # LLMs frequently hallucinate integers or nulls instead of ""
                    if corr in [0, "0", None, False, "null", "None"]:
                        corr = ""

                    # Ignore the patch entirely if the target text isn't a string
                    if not orig or not isinstance(orig, str):
                        continue

                    patches.append(
                        PhoneticPatch(
                            original=str(orig),
                            correction=str(corr)
                        )
                    )

            return patches

        except json.JSONDecodeError as e:
            # CRITICAL FIX: Graceful Degradation if Qwen hallucinates bad JSON
            print(f"\n[WARNING] Qwen produced invalid JSON for chunk. Skipping patches. Reason: {e}")
            return []
        except Exception as e:
            raise map_phonetic_error(e, context="Generate Qwen JSON Patches") from e