"""Reward models used to score generated images against their prompt.

They feed the reward-conditioned text-to-image training (`--extra-cond`, `--compute-reward-on-the-fly`; see
bigfix/trainer/txt_trainer.py and bigfix/utils/reward_utils.py) and the RL fine-tuning.

Every backend is optional: `pip install -e ".[reward]"` installs them, and a missing one only raises an
ImportError when its reward class is instantiated. All rewards take images in [-1, 1] with shape (B, 3, H, W)
plus the list of prompts, and return one score per image.
"""
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModel

try:
    import ImageReward as RM
except Exception:
    RM = None

try:
    from aesthetic_predictor import predict_aesthetic
except Exception:
    predict_aesthetic = None

try:
    import hpsv2
except Exception:
    hpsv2 = None


def _from_pretrained_with_fallback(loader_cls, candidates, cache_dir=None, token=None, **kwargs):
    """
    Try multiple HF model IDs and return the first successfully loaded object.
    This is useful when some repos are gated/renamed/unavailable.
    """
    last_exc = None
    for repo_id in candidates:
        try:
            return loader_cls.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                token=token,
                **kwargs,
            ), repo_id
        except TypeError:
            # Backward compatibility with older transformers versions lacking `token=`.
            try:
                return loader_cls.from_pretrained(
                    repo_id,
                    cache_dir=cache_dir,
                    **kwargs,
                ), repo_id
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Could not load any repo from candidates={candidates}. Last error: {last_exc}"
    )


class HPSReward(nn.Module):

    def __init__(self):
        super().__init__()
        if hpsv2 is None:
            raise ImportError("hpsv2 is not installed. Install hpsv2 to enable HPSReward.")

    @torch.no_grad()
    def forward(self, images, prompt):

        imgs = ((images + 1) / 2).clamp(0, 1)

        repeat = images.size(0) // len(prompt)
        prompt = [p for p in prompt for _ in range(repeat)]

        rewards = []

        for img, txt in zip(imgs, prompt):

            with tempfile.NamedTemporaryFile(suffix=".png") as f:

                to_pil_image(img.cpu()).save(f.name)

                rewards.append(
                    hpsv2.score(
                        f.name,
                        txt,
                    )
                )

        return torch.tensor(
            rewards,
            device=images.device,
        )


class CLIPRealismReward(nn.Module):
    def __init__(self, device="cuda", cache_dir=None, token=None):
        super().__init__()
        # Load CLIP model (ViT-L/14 for better quality)
        from transformers import CLIPProcessor, CLIPModel

        model_candidates = [
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-base-patch32",
        ]
        self.clip_model, used_repo = _from_pretrained_with_fallback(
            CLIPModel,
            model_candidates,
            cache_dir=cache_dir,
            token=token,
        )
        self.clip_model = self.clip_model.eval().to(device)
        self.device = device
        self.processor, _ = _from_pretrained_with_fallback(
            CLIPProcessor,
            [used_repo] + [r for r in model_candidates if r != used_repo],
            cache_dir=cache_dir,
            token=token,
        )
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, prompt, **kwargs):
        """
        images: Tensor (B, 3, H, W) in [-1, 1]
        Returns: realism score in [0, 1]
        """
        # Convert [-1,1] → PIL images
        imgs_01 = (images + 1) / 2  # [0,1]
        imgs_01 = imgs_01.clamp(0, 1)

        repeat_factor = images.size(0) // len(prompt)
        prompt = [c for c in prompt for _ in range(repeat_factor)]

        # Preprocess for CLIP
        inputs = self.processor(
            text=prompt,
            images=[to_pil_image(img.cpu()) for img in imgs_01],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(images.device)

        # Get embeddings
        outputs = self.clip_model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds

        # Cosine similarity → reward
        similarity = F.cosine_similarity(img_emb, txt_emb)
        return similarity

class AestheticReward(nn.Module):

    def __init__(self):

        super().__init__()
        if predict_aesthetic is None:
            raise ImportError("aesthetic-predictor is not installed. Install aesthetic-predictor to enable AestheticReward.")

    @torch.no_grad()
    def forward(self, images, prompt=None):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")

        imgs = ((images.clamp(-1, 1) + 1) / 2).clamp(0, 1)

        scores = []
        for img in imgs:
            pil_img = to_pil_image(img.cpu())
            score = predict_aesthetic(pil_img)
            scores.append(float(score))

        return torch.tensor(scores, device=images.device, dtype=torch.float32)

class PickaPickReward(nn.Module):

    def __init__(self, device="cuda", cache_dir=None, token=None):
        super().__init__()
        self.device = device
        model_candidates = [
            "yuvalkirstain/PickScore_v1",
        ]
        self.processor, used_repo = _from_pretrained_with_fallback(
            AutoProcessor,
            model_candidates,
            cache_dir=cache_dir,
            token=token,
        )
        self.model, _ = _from_pretrained_with_fallback(
            AutoModel,
            [used_repo] + [r for r in model_candidates if r != used_repo],
            cache_dir=cache_dir,
            token=token,
        )
        self.model = self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, images, prompt, **kwargs):
        """
        img: PIL image
        txt: string prompt
        returns: float
        """

        image_inputs = self.processor(
            images=(images+1)/2,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        def _as_embedding(x):
            if torch.is_tensor(x):
                return x
            if hasattr(x, "image_embeds") and x.image_embeds is not None:
                return x.image_embeds
            if hasattr(x, "text_embeds") and x.text_embeds is not None:
                return x.text_embeds
            if hasattr(x, "pooler_output") and x.pooler_output is not None:
                return x.pooler_output
            if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
                return x.last_hidden_state.mean(dim=1)
            raise TypeError("Unsupported model output type for embedding extraction")

        with torch.no_grad():
            # embed
            image_embs = _as_embedding(self.model.get_image_features(**image_inputs))
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = _as_embedding(self.model.get_text_features(**text_inputs))
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = (image_embs * text_embs).sum(dim=-1).squeeze()

        return scores


class ImageRewardModel(nn.Module):

    def __init__(self):
        super().__init__()

        if RM is None:
            raise ImportError(
                "ImageReward is not available or has missing deps (e.g. clip). "
                "Install image-reward and clip to enable ImageRewardModel."
            )

        self.reward = RM.load("ImageReward-v1.0")

    @torch.no_grad()
    def forward(self, images, prompt):

        imgs = ((images + 1) / 2).clamp(0, 1)

        repeat = images.size(0) // len(prompt)
        prompt = [p for p in prompt for _ in range(repeat)]

        rewards = []

        for img, txt in zip(imgs, prompt):

            with tempfile.NamedTemporaryFile(suffix=".png") as f:

                to_pil_image(img.cpu()).save(f.name)

                score = self.reward.score(
                    txt,
                    f.name,
                )

            rewards.append(score)

        return torch.tensor(
            rewards,
            device=images.device,
            dtype=torch.float32,
        )

class MultiReward(nn.Module):
    def __init__(self, reward_fns, weights=None, normalize=True):
        """
        reward_fns: list of reward functions (nn.Module), each returns (B,) scores
        weights: list of floats, same length as reward_fns
        normalize: whether to z-norm each reward before combining
        """
        super().__init__()
        self.reward_fns = nn.ModuleList(reward_fns)
        self.weights = weights if weights is not None else [1.0] * len(reward_fns)
        self.normalize = normalize

    @torch.no_grad()
    def forward(self, images, **kwargs):
        rewards = []
        batch_size = images.size(0)
        for fn in self.reward_fns:
            r = fn(images=images, **kwargs)

            # Reward backends can emit shapes like (B,), (B,1), or lists/arrays.
            # Convert everything to a flat (B,) tensor for safe stacking.
            if not torch.is_tensor(r):
                r = torch.as_tensor(r, device=images.device, dtype=torch.float32)
            else:
                r = r.to(device=images.device, dtype=torch.float32)

            if r.ndim == 0:
                r = r.unsqueeze(0)

            if r.size(0) != batch_size:
                raise ValueError(
                    f"Reward output batch mismatch from {fn.__class__.__name__}: "
                    f"got first dim {r.size(0)}, expected {batch_size}."
                )

            if r.ndim > 1:
                r = r.reshape(batch_size, -1)
                if r.size(1) != 1:
                    # If a backend emits extra dims, reduce to one score per sample.
                    r = r.mean(dim=1, keepdim=True)
                r = r.squeeze(1)

            if self.normalize:
                r = (r - r.mean()) / (r.std() + 1e-6)
            rewards.append(r)

        rewards = torch.stack(rewards, dim=0)  # (num_rewards, B)
        weights = torch.tensor(self.weights, device=images.device).view(-1, 1)
        final_reward = (rewards * weights).sum(dim=0) / weights.sum()
        return final_reward
