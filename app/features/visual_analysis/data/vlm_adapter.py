# Canvas: visual_analysis_vlm_adapter_v8
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


class QwenVLMAdapter(IVisualLanguageModelAdapter):
    """
    Dedicated Adapter for Qwen 2.5 VL via llama.cpp.
    FIXED: Removed the incompatible LLaVA 1.5 Chat Handler to prevent C++ Segfaults.
    """

    @staticmethod
    def _get_qwen_model(model_path: str, mmproj_path: str) -> Any:
        from llama_cpp import Llama

        # We pass the clip_model_path directly to the constructor.
        # Modern llama.cpp detects the architecture and uses the correct internal multimodal logic.
        return Llama(
            model_path=model_path,
            # chat_format="chatml", # Uncomment if your specific GGUF requires explicit ChatML format
            clip_model_path=mmproj_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"VLM base model file missing: {manifest.model_path}")

            if not os.path.exists(settings.VISION_MMPROJ_PATH):
                raise FileNotFoundError(f"VLM MMProj file missing: {settings.VISION_MMPROJ_PATH}")

            llm = self._get_qwen_model(manifest.model_path, settings.VISION_MMPROJ_PATH)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_visual_analysis_error(e, context=f"Load Qwen VLM Adapter: {manifest.model_path}") from e

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
            raise map_visual_analysis_error(e, context="Unload Qwen VLM Adapter") from e

    def analyze_frame(
            self,
            model: LoadedModel,
            image_path: Path,
            original_text: str,
            context_string: str
    ) -> str:
        try:
            llm: Any = model.model_instance

            if not image_path.exists():
                raise FileNotFoundError(f"Extracted frame missing from disk: {image_path}")

            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")

            user_text = VLM_USER_PROMPT_TEMPLATE.format(
                context_string=context_string,
                original_text=original_text
            )

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

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1
            )
            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            raise map_visual_analysis_error(e, context=f"Qwen-VL Frame Analysis on {image_path.name}") from e