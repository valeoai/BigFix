import torch


def init_reward_fn(args, is_master=False):
    """Build and return the configured reward function.

    Returns:
        tuple: (reward_fn, reward_enabled, reward_component)
    """
    reward_component = getattr(args, "reward_component", "multi").strip().lower()
    cache_dir = getattr(args, "hf_cache_dir", None)
    hf_token = getattr(args, "hf_token", None)

    try:
        from bigfix.utils.rewards import (
            HPSReward,
            CLIPRealismReward,
            AestheticReward,
            PickaPickReward,
            ImageRewardModel,
            MultiReward,
        )

        reward_catalog = {
            "hps": (lambda: HPSReward(), 0.30),
            "clip": (
                lambda: CLIPRealismReward(device=args.device, cache_dir=cache_dir, token=hf_token),
                0.30,
            ),
            # "aesthetic": (lambda: AestheticReward(), 0.10),
            #"pick": (lambda: PickaPickReward(device=args.device, cache_dir=cache_dir, token=hf_token), 0.15),
            "imagereward": (lambda: ImageRewardModel(), 0.15),
        }

        if reward_component == "multi":
            components_raw = getattr(args, "reward_components", "hps,clip,imagereward")
            selected = [c.strip().lower() for c in components_raw.split(",") if c.strip()]
            if len(selected) == 0:
                selected = ["hps", "clip", "imagereward"]

            reward_fns = []
            reward_weights = []
            for name in selected:
                if name not in reward_catalog:
                    continue
                ctor, weight = reward_catalog[name]
                reward_fns.append(ctor().to(args.device))
                reward_weights.append(weight)

            if len(reward_fns) == 0:
                raise RuntimeError("No valid reward component could be initialized.")

            reward_fn = MultiReward(
                reward_fns=reward_fns,
                weights=reward_weights,
                normalize=True,
            ).to(args.device)
            reward_fn.component_names = selected
        else:
            if reward_component not in reward_catalog:
                reward_component = "hps"
            reward_fn = reward_catalog[reward_component][0]().to(args.device)
            reward_fn.component_names = [reward_component]

        if is_master:
            print(f"[Reward] Enabled reward extractor: {reward_component}")

        return reward_fn, True, reward_component

    except Exception as exc:
        if is_master:
            print(f"[Reward] Failed to initialize reward extractor, fallback to zeros: {exc}")
        return None, False, reward_component


@torch.no_grad()
def extract_reward_from_img_txt(images, prompts, args, reward_fn=None, reward_enabled=False, return_components=False):
    if not reward_enabled:
        reward_fn, reward_enabled, _ = init_reward_fn(args=args, is_master=getattr(args, "is_master", False))

    bsz = images.size(0)
    if (reward_fn is None) or (not reward_enabled):
        rewards = torch.zeros(bsz, device=args.device, dtype=torch.float32)
        if return_components:
            details = {
                "scores": {},
                "normalized_scores": {},
                "normalized_total": rewards,
            }
            return rewards, details, reward_fn, reward_enabled
        return rewards, reward_fn, reward_enabled

    rewards = reward_fn(images=images, prompt=prompts)
    if not torch.is_tensor(rewards):
        rewards = torch.as_tensor(rewards, device=args.device, dtype=torch.float32)
    else:
        rewards = rewards.to(device=args.device, dtype=torch.float32)

    if rewards.ndim == 0:
        rewards = rewards.unsqueeze(0)
    if rewards.ndim > 1:
        rewards = rewards.reshape(bsz, -1).mean(dim=1)

    if rewards.size(0) != bsz:
        raise ValueError(f"Reward batch mismatch: got {rewards.size(0)} values for batch size {bsz}.")

    if return_components:
        if not hasattr(reward_fn, "reward_fns"):
            single_name = getattr(reward_fn, "component_names", ["score_1"])[0]
            norm_single = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            details = {
                "scores": {single_name: rewards},
                "normalized_scores": {single_name: norm_single},
                "normalized_total": norm_single,
            }
            return rewards, details, reward_fn, reward_enabled

        component_names = list(getattr(reward_fn, "component_names", []))
        if len(component_names) == 0:
            component_names = [f"score_{i+1}" for i in range(len(reward_fn.reward_fns))]

        raw_components = []
        for fn in reward_fn.reward_fns:
            r = fn(images=images, prompt=prompts)
            if not torch.is_tensor(r):
                r = torch.as_tensor(r, device=args.device, dtype=torch.float32)
            else:
                r = r.to(device=args.device, dtype=torch.float32)
            if r.ndim == 0:
                r = r.unsqueeze(0)
            if r.ndim > 1:
                r = r.reshape(bsz, -1).mean(dim=1)
            raw_components.append(r)

        limit = min(3, len(raw_components))
        weights = getattr(reward_fn, "weights", [1.0] * len(raw_components))
        weights_t = torch.tensor(weights[:limit], device=args.device, dtype=torch.float32)
        norm_components = []
        score_map = {}
        norm_map = {}

        for i in range(limit):
            name = component_names[i] if i < len(component_names) else f"score_{i+1}"
            raw_i = raw_components[i]
            norm_i = (raw_i - raw_i.mean()) / (raw_i.std() + 1e-6)
            score_map[name] = raw_i
            norm_map[name] = norm_i
            norm_components.append(norm_i)

        if limit == 0:
            normalized_total = torch.zeros(bsz, device=args.device, dtype=torch.float32)
        else:
            stack = torch.stack(norm_components, dim=0)
            normalized_total = (stack * weights_t.view(-1, 1)).sum(dim=0) / (weights_t.sum() + 1e-6)

        details = {
            "scores": score_map,
            "normalized_scores": norm_map,
            "normalized_total": normalized_total,
        }
        return rewards, details, reward_fn, reward_enabled

    return rewards, reward_fn, reward_enabled
