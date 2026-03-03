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