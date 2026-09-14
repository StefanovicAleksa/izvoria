# Canvas: visual_analysis_vlm_adapter_v10
# Path: app/features/visual_analysis/data/vlm_adapter.py

import gc
import os
from pathlib import Path
from typing import Any, List

import torch
from PIL import Image

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_analysis.domain.interfaces import IVisualLanguageModelAdapter
from app.features.visual_analysis.data.prompts import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT_TEMPLATE
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error
from app.features.visual_analysis.utils.frame_description_parser import parse_frame_descriptions

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

    def analyze_frames(
            self,
            model: LoadedModel,
            image_paths: List[Path],
            original_text: str,
            context_string: str
    ) -> List[str]:
        if not image_paths:
            return []

        try:
            payload = model.model_instance
            hf_model = payload["model"]
            processor = payload["processor"]

            images = [self._open_image(image_path) for image_path in image_paths]
            if len(images) > 1:
                # All frames share one context window, so each gets a smaller pixel budget than a lone frame
                images = [self._fit_pixel_budget(image, settings.VISION_MULTI_IMAGE_MAX_PIXELS) for image in images]

            frame_count = len(images)
            user_text = VLM_USER_PROMPT_TEMPLATE.format(
                context_string=context_string,
                original_text=original_text,
                frame_count=frame_count,
                frame_lines="\n".join(f"Frame {index}: <description>" for index in range(1, frame_count + 1))
            )

            messages = [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user_text}]
                }
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images, return_tensors="pt").to(hf_model.device)

            with torch.inference_mode():
                # Greedy decoding: evidence descriptions must be reproducible, and sampling invents details
                generated_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=settings.VISION_MAX_NEW_TOKENS_PER_FRAME * frame_count,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return parse_frame_descriptions(output_text, frame_count)

        except Exception as e:
            raise map_visual_analysis_error(
                e, context=f"Qwen2-VL analysis of {len(image_paths)} frame(s) starting at {image_paths[0].name}"
            ) from e

    @staticmethod
    def _open_image(image_path: Path) -> Any:
        if not image_path.exists():
            raise FileNotFoundError(f"Extracted frame missing from disk: {image_path}")
        with Image.open(image_path) as image:
            return image.convert("RGB")

    @staticmethod
    def _fit_pixel_budget(image: Any, max_pixels: int) -> Any:
        """Downscales (never upscales) so width * height <= max_pixels, keeping the aspect ratio."""
        width, height = image.size
        if width * height <= max_pixels:
            return image
        scale = (max_pixels / float(width * height)) ** 0.5
        return image.resize((max(28, int(width * scale)), max(28, int(height * scale))), Image.Resampling.BICUBIC)
