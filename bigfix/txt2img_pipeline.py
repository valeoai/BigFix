# Lightweight, diffusers-style wrapper around the txt-to-img MaskGIT trainer.
#
#   from bigfix import MaskGITPipeline
#
#   # from local checkpoints
#   pipe = MaskGITPipeline.from_pretrained(
#       vit_folder="./saved_networks/ImageNet_384_large.pth",
#       vqgan_folder="./saved_networks/vq_ds16_c2i.pt",
#   ).to("cuda")
#
#   # or straight from a Hugging Face Hub repo, e.g. https://huggingface.co/llvictorll/BigFIX
#   pipe = MaskGITPipeline.from_pretrained(
#       repo_id="llvictorll/BigFIX",
#       vit_filename="BigFix_XLarge_aplha02_res512_MiroFT.pth",
#       vqgan_filename="vq_ds16_t2i.pt",
#       config="xlarge_txt2img_gpic.yaml",
#       img_size=512,
#   ).to("cuda")
#
#   image = pipe(
#       prompt="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
#       num_inference_steps=32,
#       guidance_scale=3.0,
#       seed=42,
#   ).images[0]
#
#   image.save("t2i_example.png")
#
# This reuses bigfix.trainer.txt_trainer.MaskGIT directly (the same construction recipe as app.py and
# zero_shot_imagenet_t2i_score.py: load a YAML config into an args Namespace, point it at the
# checkpoints, build the trainer), skipping the distributed/dataloader machinery bigfix/main.py sets up
# for actual training -- so it works standalone, without torchrun.
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from bigfix.sampler.halton_sampler import TxtHaltonSampler
from bigfix.trainer.txt_trainer import MaskGIT
from bigfix.utils.utils import load_args_from_file

# Fields the current trainer reads unconditionally (bigfix/trainer/txt_trainer.py) that some shipped
# configs (e.g. bigfix/config/base_txt2img.yaml) predate, plus a fallback for fields we may override
# below. Defaulted instead of crashing on a stale yaml or clobbering a config's own value.
_ARG_DEFAULTS = {
    "extra_cond": False,
    "top_p": 1.0,
    "skip_t5_init": False,
    "compute_reward_on_the_fly": False,
    "drop_reward": 0.5,
    "dtype": "bfloat16",
    "use_ema": False,
    "compile": False,
}


@dataclass
class MaskGITOutput:
    """ Mirrors diffusers' pipeline output so `pipe(...).images[0]` works the same way. """
    images: List[Image.Image]


class MaskGITPipeline:
    """ Standalone (single-process) inference wrapper around the txt-to-img MaskGIT trainer. """

    def __init__(self, model: MaskGIT):
        self.model = model

    @classmethod
    def from_pretrained(cls, vit_folder=None, vqgan_folder=None, repo_id=None,
                         vit_filename=None, vqgan_filename=None, local_dir="./saved_networks/",
                         config="base_txt2img.yaml", device=None, dtype=None,
                         vit_size=None, img_size=None, use_ema=None, compile=None):
        """ Load the transformer + VQGAN checkpoints and build the pipeline.

        Either pass local checkpoint paths (`vit_folder`, `vqgan_folder`), or a Hugging Face
        Hub repo (`repo_id` + `vit_filename`/`vqgan_filename`, e.g. the two files at
        https://huggingface.co/llvictorll/BigFIX), which are downloaded into `local_dir` the
        same way app.py does with `huggingface_hub.hf_hub_download` (cached there afterwards).

           :param
            vit_folder    -> str: path to the transformer checkpoint, or a folder containing
                             "current.pth" (same convention as --vit-folder in bigfix/main.py)
            vqgan_folder  -> str: path to the pretrained VQGAN checkpoint
            repo_id       -> str: Hugging Face Hub repo to download the checkpoints from instead
                             of using local paths, e.g. "llvictorll/BigFIX"
            vit_filename  -> str: transformer checkpoint filename inside `repo_id`
            vqgan_filename -> str: VQGAN checkpoint filename inside `repo_id`
            local_dir     -> str: where `repo_id` downloads land (default "./saved_networks/")
            config        -> str: base YAML config to load defaults from (architecture, sampler
                             settings, ...): either a file path, or the name of a config shipped
                             with the package, see bigfix/config/*.yaml -- pick one matching the
                             checkpoint's training run when possible (e.g.
                             "xlarge_txt2img_gpic.yaml" for a "xlarge" checkpoint)
            device        -> str | torch.device: defaults to "cuda" if available, else "cpu"
            dtype         -> str: override the config's dtype ("bfloat16" or "float32"; controls
                             the autocast precision used during the forward passes, not the
                             stored weight dtype). None (default) keeps the config's own value,
                             falling back to "bfloat16" if the config doesn't set one
            vit_size      -> str: override the config's vit_size (tiny|small|base|large|xlarge|
                             xxlarge) -- must match what the checkpoint was trained with
            img_size      -> str: override the config's img_size -- must match what the
                             checkpoint was trained with (it sizes the transformer's positional
                             embedding table, so a mismatch fails to load rather than silently
                             degrading)
            use_ema       -> bool: override the config's use_ema (load/use the EMA weights if
                             the checkpoint has them). None (default) keeps the config's value
            compile       -> bool: override the config's compile (torch.compile the transformer).
                             None (default) keeps the config's value
        """
        if repo_id is not None:
            if vit_filename is None or vqgan_filename is None:
                raise ValueError("repo_id was given: also pass vit_filename and vqgan_filename.")
            vit_folder = hf_hub_download(repo_id=repo_id, filename=vit_filename, local_dir=local_dir)
            vqgan_folder = hf_hub_download(repo_id=repo_id, filename=vqgan_filename, local_dir=local_dir)
        elif vit_folder is None or vqgan_folder is None:
            raise ValueError(
                "Pass either (vit_folder and vqgan_folder), or (repo_id, vit_filename, vqgan_filename)."
            )

        args = load_args_from_file(config)

        for name, default in _ARG_DEFAULTS.items():
            if not hasattr(args, name):
                setattr(args, name, default)

        args.vit_folder = vit_folder
        args.vqgan_folder = vqgan_folder
        args.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if dtype is not None:
            args.dtype = dtype
        if vit_size is not None:
            args.vit_size = vit_size
        if img_size is not None:
            args.img_size = img_size
        if use_ema is not None:
            args.use_ema = use_ema
        if compile is not None:
            args.compile = compile
        args.resume = True      # actually load the checkpoints above
        args.test_only = True   # we're doing inference, not training
        args.is_master = True
        args.is_multi_gpus = False
        args.global_rank = 0

        model = MaskGIT(args)
        model.vit.eval()
        model.ae.eval()
        return cls(model)

    def to(self, device):
        """ Move every submodule to `device`, mirroring diffusers' `pipe.to("cuda")`. """
        device = torch.device(device)
        self.model.args.device = device
        self.model.vit.to(device)
        self.model.ae.to(device)
        if not getattr(self.model.args, "skip_t5_init", False):
            self.model.t5_model.to(device)
        return self

    def _encode_prompt(self, prompts):
        t5_input = self.model.t5_tok(
            prompts, truncation=True, padding="max_length", max_length=120, return_tensors="pt"
        ).to(self.model.args.device)
        return self.model.t5_model(**t5_input).last_hidden_state

    @staticmethod
    def _to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = ((images.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(img) for img in images]

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]], negative_prompt: Optional[str] = None,
                 num_images_per_prompt: int = 1, num_inference_steps: Optional[int] = None,
                 guidance_scale: Optional[float] = None, seed: Optional[int] = None,
                 verbose: bool = False) -> MaskGITOutput:
        """ Generate image(s) from a text prompt.
           :param
            prompt                -> str | list[str]: the text prompt(s)
            negative_prompt       -> str: optional negative prompt for classifier-free guidance
            num_images_per_prompt -> int: how many samples to draw per prompt
            num_inference_steps   -> int: overrides the config's sampler `step` for this call
            guidance_scale        -> float: overrides the config's sampler `w` (CFG weight) for
                                     this call
            seed                  -> int: sets the global torch seed before sampling for this
                                     call (the underlying sampler draws from the global RNG, so
                                     unlike diffusers this isn't a scoped `torch.Generator`)
            verbose               -> bool: show a progress bar over sampling steps
           :return
            MaskGITOutput(images=[PIL.Image, ...])
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.model.args.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [p for p in prompts for _ in range(num_images_per_prompt)]

        txt_emb = self._encode_prompt(prompts)
        neg_txt_emb = self._encode_prompt([negative_prompt] * len(prompts)) if negative_prompt else None

        sampler = TxtHaltonSampler(
            sm_temp_min=self.model.args.sm_temp_min, sm_temp_max=self.model.args.sm_temp,
            temp_pow=1, temp_warmup=self.model.args.temp_warmup,
            w=self.model.args.cfg_w if guidance_scale is None else guidance_scale,
            step=self.model.args.step if num_inference_steps is None else num_inference_steps,
            randomize=self.model.args.randomize, top_k=self.model.args.top_k,
            top_p=self.model.args.top_p,
        )

        try:
            images, _, _ = sampler(self.model, txt_emb=txt_emb, neg_txt_emb=neg_txt_emb, verbose=verbose)
        finally:
            # TxtHaltonSampler.__call__ unconditionally leaves trainer.vit in train() mode when
            # it returns (bigfix/sampler/halton_sampler.py) -- correct for its original mid-training
            # visualization use, but this pipeline is inference-only and reuses the same model
            # across calls, so without this the *second* call onward would run with dropout
            # active. Restore eval() every time, including if sampling raised.
            self.model.vit.eval()
        return MaskGITOutput(images=self._to_pil(images))
