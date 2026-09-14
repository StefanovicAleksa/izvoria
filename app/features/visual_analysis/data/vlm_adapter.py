# Canvas: visual_analysis_vlm_adapter_v9
# Path: app/features/visual_analysis/data/vlm_adapter.py

import gc
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_analysis.domain.interfaces import IVisualLanguageModelAdapter
from app.features.visual_analysis.data.prompts import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT_TEMPLATE
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error

# Bounds Qwen2-VL's dynamic-resolution vision tower so a single large frame can't
# blow past the VRAM budget the Model Orchestrator reserved for this phase.
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 1280 * 28 * 28


class QwenVLMAdapter(IVisualLanguageModelAdapter):
    """
    Adapter for Qwen2-VL via native Hugging Face Transformers (4-bit quantized).

    Runs on transformers instead of llama.cpp/GGUF: llama.cpp's mtmd vision encoder
    has a confirmed, unresolved upstream CUDA bug specific to the Qwen-VL model
    family's vision graph construction (SIGSEGV in ggml_gallocr_alloc_graph, inside
    clip_image_batch_encode) - resolution-independent and reproduced across
    NVIDIA/AMD/Intel backends by multiple people, so no build flag or model
    requantization routes around it. See
    https://github.com/ggml-org/llama.cpp/issues/23407 and
    https://github.com/ggml-org/llama.cpp/issues/17012 for the upstream tracking.
    integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py
    documents the original crash against the GGUF build for reference.
    """

    @staticmethod
    def _get_qwen_model(model_path: str) -> Any:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # model_path points at a checkpoint already quantized to 4-bit and saved back
        # to disk (see integration_tests/features/visual_analysis or the model prep
        # step) - the quantization config is embedded in its config.json and picked up
        # automatically, so it doesn't need to be passed here.
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
            local_files_only=True,
        )

        return {"model": hf_model, "processor": processor}

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"VLM directory missing: {manifest.model_path}")

            payload = self._get_qwen_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=payload)
        except Exception as e:
            raise map_visual_analysis_error(e, context=f"Load Qwen2-VL Transformers Adapter: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance") and model.model_instance is not None:
                payload = model.model_instance
                if isinstance(payload, dict):
                    payload.pop("model", None)
                    payload.pop("processor", None)
                model.model_instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_visual_analysis_error(e, context="Unload Qwen2-VL Transformers Adapter") from e

    def analyze_frame(
            self,
            model: LoadedModel,
            image_path: Path,
            original_text: str,
            context_string: str
    ) -> str:
        try:
            payload = model.model_instance
            hf_model = payload["model"]
            processor = payload["processor"]

            if not image_path.exists():
                raise FileNotFoundError(f"Extracted frame missing from disk: {image_path}")

            image = Image.open(image_path).convert("RGB")

            user_text = VLM_USER_PROMPT_TEMPLATE.format(
                context_string=context_string,
                original_text=original_text
            )

            messages = [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text}
                    ]
                }
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(hf_model.device)

            with torch.inference_mode():
                generated_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.1,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()

        except Exception as e:
            raise map_visual_analysis_error(e, context=f"Qwen2-VL Frame Analysis on {image_path.name}") from e
