# Canvas: visual_analysis_vlm_adapter_v1
# Path: app/features/visual_analysis/data/vlm_adapter.py

import os
import gc
import base64
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_analysis.domain.interfaces import IVisualLanguageModelAdapter
from app.features.visual_analysis.data.prompts import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT_TEMPLATE
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error


class LlamaVLMAdapter(IVisualLanguageModelAdapter):
    """
    Adapter for executing Vision-Language Models (specifically LLaVA 1.5 7B) via llama.cpp.
    Manages the multimodal projection and base64 image encoding.
    """

    @staticmethod
    def _get_llava_model(base_model_path: str, mmproj_path: str) -> Any:
        """
        FACTORY METHOD & LAZY IMPORT
        Prevents C-extension collisions with PyTorch during application startup.
        Loads the Multimodal Projector to translate pixels into LLM tokens.
        """
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        # By typing this as Any, we satisfy PyCharm's strict inspector
        # while passing the exact handler llama.cpp expects under the hood.
        chat_handler: Any = Llava15ChatHandler(clip_model_path=mmproj_path)

        return Llama(
            model_path=base_model_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1,  # Offload all layers to the GPU
            n_ctx=4096,       # Vision models require larger context windows for image embeddings
            verbose=False     # Suppress spammy C++ token generation logs
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"VLM base model file missing: {manifest.model_path}")
            if not os.path.exists(settings.VISION_MMPROJ_PATH):
                raise FileNotFoundError(f"VLM MMProj file missing: {settings.VISION_MMPROJ_PATH}")

            llm = self._get_llava_model(manifest.model_path, settings.VISION_MMPROJ_PATH)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_visual_analysis_error(e, context=f"Load VLM Adapter: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            return True
        except Exception as e:
            raise map_visual_analysis_error(e, context="Unload VLM Adapter") from e

    def analyze_frame(
        self,
        model: LoadedModel,
        image_path: Path,
        original_text: str,
        context_string: str
    ) -> str:
        try:
            llm: Any = model.model_instance

            # 1. Ensure image exists and encode to Base64
            if not image_path.exists():
                raise FileNotFoundError(f"Extracted frame missing from disk: {image_path}")

            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # 2. Format the user text prompt
            user_text = VLM_USER_PROMPT_TEMPLATE.format(
                context_string=context_string,
                original_text=original_text
            )

            # 3. Construct the Multimodal Chat Payload
            messages = [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ]

            # 4. Execute Inference
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=150,   # Cap the output to ensure a concise description
                temperature=0.1   # Near-zero creativity for factual, grounded analysis
            )

            output_text = response['choices'][0]['message']['content'].strip()
            return output_text

        except Exception as e:
            raise map_visual_analysis_error(e, context=f"VLM Frame Analysis on {image_path.name}") from e