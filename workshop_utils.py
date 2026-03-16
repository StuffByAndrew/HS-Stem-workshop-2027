import gc
import mediapy as mp

import torch
from diffusers import DiffusionPipeline

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata

device = 'cuda'

def im_to_np(im):
  im = (im / 2 + 0.5).clamp(0, 1)
  im = im.detach().cpu().permute(1, 2, 0).numpy()
  im = (im * 255).round().astype("uint8")
  return im


# Garbage collection function to free memory
def flush():
    gc.collect()
    torch.cuda.empty_cache()

import torch
import sys
import os
import contextlib
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

class DeepFloydTextEmbedder:
    _first_load_done = False  # class-level flag

    def __init__(
        self,
        model_id="DeepFloyd/IF-I-L-v1.0",
        device="cuda",
        dtype=torch.float16,
        variant="fp16",
        flush_fn=None,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.variant = variant
        self.flush_fn = flush_fn

        self._text_encoder = None
        self._pipe = None

    def _flush(self):
        if self.flush_fn:
            try:
                self.flush_fn()
            except:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self):

        if self._pipe is not None:
            return

        self._flush()

        suppress = DeepFloydTextEmbedder._first_load_done

        if suppress:
            devnull = open(os.devnull, "w")
            ctx = contextlib.ExitStack()
            ctx.enter_context(contextlib.redirect_stdout(devnull))
            ctx.enter_context(contextlib.redirect_stderr(devnull))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            self._text_encoder = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                variant=self.variant,
                torch_dtype=self.dtype,
            ).to(self.device)

            self._pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                text_encoder=self._text_encoder,
                unet=None,
                safety_checker=None,
                torch_dtype=self.dtype,
                variant=self.variant,
            )

        DeepFloydTextEmbedder._first_load_done = True

        self._flush()

    @torch.inference_mode()
    def __call__(self, prompts):

        if isinstance(prompts, str):
            prompts = [prompts]

        self._load()

        pairs = [self._pipe.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*pairs)

        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)

        return prompt_embeds, negative_prompt_embeds

    def close(self):
        if self._pipe is not None:
            del self._pipe
        if self._text_encoder is not None:
            del self._text_encoder
        self._pipe = None
        self._text_encoder = None
        self._flush()


# --- Example usage (simple API) ---
# embedder = DeepFloydTextEmbedder(device=device, flush_fn=flush)
# prompt_embeds, neg_embeds = embedder([
#     "painting of a snowy mountain village",
#     "painting of a horse",
# ])
# embedder.close()

import torch
from diffusers import DiffusionPipeline


class DeepFloydIF:
    def __init__(self, device="cuda"):
        self.device = device
        self.dtype = torch.float16

        self.stage_1 = self._load_stage1()
        self.stage_2 = self._load_stage2()
        self.stage_3 = self._load_stage3()

    def _load_stage1(self):
        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe.enable_model_cpu_offload()
        pipe.to(self.device)
        return pipe

    def _load_stage2(self):
        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe.enable_model_cpu_offload()
        pipe.to(self.device)
        return pipe

    def _load_stage3(self):
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe.enable_model_cpu_offload()
        pipe.to(self.device)
        return pipe

    # ---------------------------
    # User-facing API
    # ---------------------------

    def sample_stage1(
        self,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction="mean",
        generator=None,
    ):
        return sample_stage_1(
            self.stage_1,
            prompt_embeds,
            negative_prompt_embeds,
            views,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            reduction=reduction,
            generator=generator,
        )

    def sample_stage2(
        self,
        image_64,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction="mean",
        noise_level=50,
        generator=None,
    ):
        return sample_stage_2(
            self.stage_2,
            image_64,
            prompt_embeds,
            negative_prompt_embeds,
            views,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            reduction=reduction,
            noise_level=noise_level,
            generator=generator,
        )

    def sample_stage3(
        self,
        image_256,
        prompt,
        noise_level=0,
        generator=None,
    ):
        image = self.stage_3(
            prompt=prompt,
            image=image_256,
            noise_level=noise_level,
            output_type="pt",
            generator=generator,
        ).images

        return image * 2 - 1


# model = DeepFloydIF()
# image_64 = model.sample_stage1(prompt_embeds, negative_prompt_embeds, views)
