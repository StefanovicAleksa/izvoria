# Canvas: qwen_adapter_v2
# Path: app/features/phonetic_correction/data/qwen_adapter.py

import os
import json
import torch
from typing import List

# Import the strict TypedDicts directly from llama_cpp
from llama_cpp import (
    Llama,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestResponseFormat
)

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.phonetic_correction.domain.interfaces import IPhoneticPatchAdapter
from app.features.phonetic_correction.domain.entities import PhoneticPatch
from app.features.phonetic_correction.data.prompts import QWEN_PHONETIC_SYSTEM_PROMPT
from app.features.phonetic_correction.utils.error_mapper import map_phonetic_error


class QwenAdapter(IPhoneticPatchAdapter):
    """
    Infrastructure adapter for Qwen 2.5 (GGUF).
    Fulfills the IPhoneticPatchAdapter contract by executing the Read-Only JSON strategy.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")

            # Qwen 2.5 requires a healthy context window for the dictionary + chunk
            llm = Llama(
                model_path=manifest.model_path,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False
            )
            return LoadedModel(manifest=manifest, model_instance=llm)

        except Exception as e:
            raise map_phonetic_error(e, context=f"Load Qwen 2.5 Model: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            llm: Llama = model.model_instance

            system_content = QWEN_PHONETIC_SYSTEM_PROMPT.format(domain_dictionary=domain_dictionary)

            # Strictly type-hinted variables to satisfy PyCharm and Clean Architecture requirements
            system_msg = ChatCompletionRequestSystemMessage(role="system", content=system_content)
            user_msg = ChatCompletionRequestUserMessage(role="user",
                                                        content=f"<INPUT_TEXT>\n{text_chunk}\n</INPUT_TEXT>")
            response_format = ChatCompletionRequestResponseFormat(type="json_object")

            # Force Qwen to output strict JSON using llama.cpp's response_format
            response = llm.create_chat_completion(
                messages=[system_msg, user_msg],
                response_format=response_format,
                temperature=0.0,  # Absolute zero temperature for maximum determinism
                max_tokens=1024
            )

            raw_output = response["choices"][0]["message"]["content"].strip()

            # Parse the JSON and map to Domain Entities
            parsed_json = json.loads(raw_output)
            raw_corrections = parsed_json.get("corrections", [])

            patches: List[PhoneticPatch] = []
            for item in raw_corrections:
                if "original_text" in item and "corrected_text" in item:
                    patches.append(
                        PhoneticPatch(
                            original=str(item["original_text"]),
                            correction=str(item["corrected_text"])
                        )
                    )

            return patches

        except Exception as e:
            raise map_phonetic_error(e, context="Generate Qwen JSON Patches") from e