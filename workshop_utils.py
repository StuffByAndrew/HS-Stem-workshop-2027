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
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

class DeepFloydTextEmbedder:
    """
    Minimal wrapper to compute text + negative (null) embeddings using
    DeepFloyd/IF-I-L-v1.0's T5 text encoder via Diffusers' encode_prompt.

    Typical use:
        embedder = DeepFloydTextEmbedder(device="cuda")
        prompt_embeds, neg_embeds = embedder(["a cat", "a dog"])
    """

    def __init__(
        self,
        model_id: str = "DeepFloyd/IF-I-L-v1.0",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
        flush_fn=None,
    ):
        self.model_id = model_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.variant = variant
        self.flush_fn = flush_fn  # optional: pass your flush() function

        self._text_encoder = None
        self._pipe = None

    def _flush(self):
        if self.flush_fn is not None:
            try:
                self.flush_fn()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load(self):
        if self._pipe is not None and self._text_encoder is not None:
            return

        self._flush()

        # Load text encoder
        self._text_encoder = T5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            variant=self.variant,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Load pipeline (no UNet to save memory)
        self._pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            text_encoder=self._text_encoder,
            unet=None,
            safety_checker=None,
            torch_dtype=self.dtype,
            variant=self.variant,
        )
        self._flush()

    @torch.inference_mode()
    def __call__(self, prompts: list[str]):
        """
        Returns:
            prompt_embeds: Tensor [N, ...]
            negative_prompt_embeds: Tensor [N, ...] (null embeds)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        self._load()

        # encode_prompt returns (prompt_embeds, negative_prompt_embeds)
        pairs = [self._pipe.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*pairs)

        prompt_embeds = torch.cat(prompt_embeds, dim=0).to(self.device)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0).to(self.device)

        return prompt_embeds, negative_prompt_embeds

    def close(self):
        """Free memory."""
        try:
            if self._text_encoder is not None:
                del self._text_encoder
            if self._pipe is not None:
                del self._pipe
        finally:
            self._text_encoder = None
            self._pipe = None
            self._flush()

    # Optional: allow `with DeepFloydTextEmbedder(...) as e:`
    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# --- Example usage (simple API) ---
# embedder = DeepFloydTextEmbedder(device=device, flush_fn=flush)
# prompt_embeds, neg_embeds = embedder([
#     "painting of a snowy mountain village",
#     "painting of a horse",
# ])
# embedder.close()