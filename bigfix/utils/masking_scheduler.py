import math
import contextlib
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _eval_mode(model):
    """ Temporarily switch every module in `model` to eval(), restoring each submodule's own
    previous `.training` flag afterwards -- unlike `model.train(bool)`, which recursively forces
    the *same* mode onto every submodule and would clobber wrappers that intentionally keep an
    inner module in a different mode than themselves (e.g. bigfix/network/ema.py's `EMA`, whose
    `self.module` is permanently eval() while the wrapper's own flag stays True). """
    states = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, was_training in states:
            m.training = was_training


def get_mask_code(code, r=None, mode="arccos", value=None, p_resample=0,
                   resample_method="shuffle", resample_p_schedule=False,
                   resample_kwargs=None, **kargs):
    """ Replace the code token by *value* according the the *mode* scheduler
       :param
        code             -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
        mode             -> str:                the rate of value to mask
        value            -> int:                mask the code by the value
        p_resample       -> float:              probability of corrupting an already-"decoded"
                                                 (unmasked) token, to mimic a sampler committing
                                                 a wrong token during iterative decoding
        resample_method  -> str:                "shuffle"|"local_swap"|"freq"|"codebook_nn"|
                                                 "self_sample", see `inject_random_tok`
        resample_p_schedule -> bool:            scale p_resample by how early in the (simulated)
                                                 decoding trajectory this sample sits, see below
        resample_kwargs  -> dict:               extra, method-specific kwargs forwarded to
                                                 `inject_random_tok` (e.g. {"model": ...,
                                                 "model_kwargs": {...}} for "self_sample")
       :return
        masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
        mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
    """
    b, h, w = code.size()
    device = code.device
    if r is None: # Shift masking rate to the correct resolution
        r = torch.rand(b)
        m = 16 * 16 # default value
        n = h * w
        alpha = math.sqrt(m / n)
        r = (alpha * r) / (1 + (alpha - 1) * r)

    if mode == "root":      # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    mask_code = code.detach().clone()
    # Sample the number of tokens + localization to mask
    mask = torch.rand(size=code.size()).to(device) < val_to_mask.view(b, 1, 1).to(device)

    if value > 0:  # Mask the selected token by the value
        mask_code[mask] = torch.full_like(mask_code[mask], value)

    # inject random tokens
    if p_resample > 0:
        p = p_resample
        if resample_p_schedule:
            # A real iterative sampler (e.g. ConfidenceSampler) injects the most randomness on
            # its earliest steps and tapers it off as decoding progresses -- see the
            # `r_temp * gumbel * (1 - ratio)` term in bigfix/sampler/confidence_sampler.py. `r` here
            # plays the same role as that step ratio (r ~ 0 is an early step with little
            # context committed yet, r ~ 1 is a late step), so scale the corruption
            # probability by (1 - r) to mirror that same decay instead of corrupting context
            # tokens at a constant rate regardless of where in the trajectory they sit.
            p = p_resample * (1 - r).view(b, 1, 1).to(device)
        mask_code = inject_random_tok(code, mask_code, ~mask, p, method=resample_method,
                                       mask_value=value, **(resample_kwargs or {}))
    # reconstruct every token, nothing is masked
    loss_mask = torch.ones_like(code).bool()

    return mask_code, mask, loss_mask


def inject_random_tok(code, mask_code, context, p, method="shuffle", mask_value=None, **kwargs):
    """ Corrupt a random `p`-fraction of the *context* (already "decoded"/unmasked) tokens,
    replacing them by a wrong-but-plausible token. This trains the model to not blindly trust
    its visible context, the same way a real iterative sampler can commit a wrong token early
    on and has to live with (or fix) it on later steps.
       :param
        code        -> torch.LongTensor [B, H, W]: the ground-truth, unmasked code
        mask_code   -> torch.LongTensor [B, H, W]: `code` with a subset already replaced by the
                       mask token
        context     -> torch.BoolTensor [B, H, W]: positions eligible for corruption (typically
                       `~mask`, i.e. the tokens that were *not* replaced by the mask token)
        p           -> float | torch.Tensor:       corruption probability, either a scalar or a
                       per-sample tensor broadcastable to `code`'s shape (e.g. [B, 1, 1])
        method      -> str: which corruption strategy to use:
            - "shuffle"     : swap with a token sampled uniformly from the same image (a single
                              shared permutation across the batch). Cheap but can pull in a
                              token from a totally unrelated region.
            - "local_swap"  : swap with one of the token's own spatial neighbours (kwarg
                              `swap_radius`, default 1). Real sampler mistakes tend to be
                              spatially coherent, so this is a closer, still-cheap proxy.
            - "freq"        : swap with a token drawn from the codebook's empirical usage
                              frequency (kwarg `codebook_freq`, a [codebook_size] tensor of
                              relative frequencies, e.g. `torch.bincount(code.flatten(),
                              minlength=codebook_size)`). Mimics a model's bias towards common
                              tokens when it errs.
            - "codebook_nn" : swap with one of the token's nearest neighbours in the VQ
                              embedding space (kwarg `codebook_knn`, precomputed once with
                              `build_codebook_knn`). Mimics confusing visually/semantically
                              similar codes, which is closer to how a trained model actually
                              errs than an arbitrary swap.
            - "self_sample" : swap with a token the model itself samples at that position from
                              `mask_code` (kwargs `model`, `model_kwargs`), exactly like
                              bigfix/sampler/confidence_sampler.py does at inference. The most
                              faithful option: it reproduces the actual object of exposure
                              bias -- a token the model found plausible enough to commit that
                              happens to be wrong -- instead of a heuristic swap.
       :return
        out_code -> torch.LongTensor [B, H, W]
    """
    if method == "shuffle":
        replacement = _shuffle_replacement(code)
    elif method == "local_swap":
        replacement = _local_swap_replacement(code, radius=kwargs.get("swap_radius", 1))
    elif method == "freq":
        replacement = _freq_replacement(code, kwargs["codebook_freq"])
    elif method == "codebook_nn":
        replacement = _codebook_nn_replacement(code, kwargs["codebook_knn"])
    elif method == "self_sample":
        replacement = _self_sample_replacement(mask_code, kwargs["model"], kwargs.get("model_kwargs", {}),
                                                mask_value=mask_value, sm_temp=kwargs.get("sm_temp", 1.0))
    else:
        raise ValueError(f"Unknown resample_method '{method}', expected one of "
                          f"'shuffle', 'local_swap', 'freq', 'codebook_nn', 'self_sample'")

    return _splice(mask_code, context, p, replacement)


def _splice(mask_code, context, p, replacement):
    """ Overwrite `mask_code` at a random `p`-fraction of `context` positions with the matching
    entries of `replacement`. `p` may be a scalar or a tensor broadcastable to `mask_code`. """
    random_prob = torch.rand_like(mask_code, dtype=torch.float)
    p = p.to(random_prob.device) if torch.is_tensor(p) else p
    random_mask = (random_prob < p) & context.bool()

    out_code = mask_code.clone()
    out_code[random_mask] = replacement[random_mask]
    return out_code


def _shuffle_replacement(code):
    """ [shuffle] Replace every token by another token sampled uniformly from the same image,
    using one permutation shared across the whole batch. """
    b, h, w = code.shape
    total = h * w

    # Generate one permutation index for all batches (shared shuffle)
    idx = torch.randperm(total, device=code.device)

    # Apply shuffle: [B, total] -> [B, H, W]
    return code.view(b, total)[:, idx].view(b, h, w)


def _local_swap_replacement(code, radius=1):
    """ [local_swap] Replace every token by one of its spatial neighbours within `radius`,
    a different random neighbour drawn independently for every position. Border positions
    replicate-pad so every position has a full neighbourhood to draw from. """
    b, h, w = code.shape
    device = code.device
    pad = radius

    code_pad = F.pad(code.unsqueeze(1).float(), (pad, pad, pad, pad), mode="replicate").squeeze(1).long()

    dy = torch.randint(-radius, radius + 1, (b, h, w), device=device)
    dx = torch.randint(-radius, radius + 1, (b, h, w), device=device)
    ys = torch.arange(h, device=device).view(1, h, 1) + pad + dy
    xs = torch.arange(w, device=device).view(1, 1, w) + pad + dx
    bs = torch.arange(b, device=device).view(b, 1, 1).expand(b, h, w)

    return code_pad[bs, ys, xs]


def _freq_replacement(code, codebook_freq):
    """ [freq] Replace every token by one drawn i.i.d. from the codebook's empirical usage
    frequency (unigram) distribution, ignoring spatial content entirely.
       :param codebook_freq -> torch.FloatTensor [codebook_size]: relative usage frequency
        (does not need to be normalized) of every codebook entry, e.g. a running or per-batch
        `torch.bincount(code.flatten(), minlength=codebook_size)`
    """
    b, h, w = code.shape
    sampled = torch.multinomial(codebook_freq.float(), num_samples=b * h * w, replacement=True)
    return sampled.view(b, h, w).to(code.device)


def _codebook_nn_replacement(code, codebook_knn):
    """ [codebook_nn] Replace every token by one of its nearest neighbours in VQ embedding
    space, picked uniformly at random among the precomputed neighbours.
       :param codebook_knn -> torch.LongTensor [codebook_size, k]: index of the k nearest
        neighbours of every codebook entry (excluding itself), see `build_codebook_knn`
    """
    codebook_knn = codebook_knn.to(code.device)
    k = codebook_knn.shape[1]

    neighbours = codebook_knn[code]                                   # [B, H, W, k]
    choice = torch.randint(0, k, code.shape, device=code.device)      # [B, H, W]
    return torch.gather(neighbours, -1, choice.unsqueeze(-1)).squeeze(-1)


def _self_sample_replacement(mask_code, model, model_kwargs, mask_value=None, sm_temp=1.0):
    """ [self_sample] Replace every token by one the model itself samples at that position from
    the *current* (partially masked) input, exactly like bigfix/sampler/confidence_sampler.py does at
    inference. Runs a no-grad forward pass and temporarily switches the model to eval() (its
    training/eval mode is restored afterwards).
       :param mask_code    -> torch.LongTensor [B, H, W]: the current (partially masked) input
        model          -> nn.Module: the transformer to sample from (commonly the EMA copy, to
                          match what bigfix/sampler/confidence_sampler.py uses at inference and to
                          avoid perturbing the model that is about to be trained on this batch)
        model_kwargs   -> dict: forwarded as `model(x=mask_code, **model_kwargs)`, typically
                          {"y": labels, "drop_label": drop_label}
        mask_value     -> int: index of the mask token; its logit is zeroed out so a corrupted
                          context token can't itself become the mask token
        sm_temp        -> float: softmax temperature before sampling
    """
    with _eval_mode(model), torch.no_grad():
        logit = model(x=mask_code, **model_kwargs)
        if mask_value is not None:
            logit = logit.clone()
            logit[..., mask_value] = -float("inf")
        prob = torch.softmax(logit * sm_temp, dim=-1)
        sampled = torch.distributions.Categorical(probs=prob).sample()
    return sampled.view(mask_code.shape)


def build_codebook_knn(codebook_embedding, k=8, chunk_size=2048):
    """ Precompute the k nearest neighbours (Euclidean distance) of every entry of a codebook,
    for use with `method="codebook_nn"`. This is O(codebook_size^2); call it once (e.g. right
    after loading the autoencoder) and cache/reuse the result rather than recomputing it every
    training step.
       :param codebook_embedding -> torch.FloatTensor [codebook_size, dim]: e.g.
        `ae.quantize.embedding.weight`, or `get_codebook_embedding(ae)` for autoencoders that
        have no learned embedding table (see below)
        k          -> int: number of neighbours to keep per entry
        chunk_size -> int: rows processed at a time, to bound peak memory
       :return
        knn -> torch.LongTensor [codebook_size, k]
    """
    n = codebook_embedding.shape[0]
    knn = torch.empty(n, k, dtype=torch.long, device=codebook_embedding.device)
    sq_norms = (codebook_embedding ** 2).sum(-1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = codebook_embedding[start:end]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        dist = sq_norms[start:end, None] + sq_norms[None, :] - 2 * chunk @ codebook_embedding.t()
        dist[torch.arange(end - start), torch.arange(start, end)] = float("inf")  # exclude itself
        knn[start:end] = torch.topk(dist, k, dim=-1, largest=False).indices

    return knn


def get_codebook_embedding(ae, device=None):
    """ Best-effort extraction of a `[codebook_size, dim]` embedding matrix from an
    autoencoder, for use with `build_codebook_knn`. Supports:
        - a learned VQ codebook exposed as `ae.quantize.embedding.weight`
          (see bigfix/network/vq_model.py)
        - the fixed color-quantization codec in bigfix/network/autoencoder.py (`AutoEncoder`), whose
          codes are a deterministic (R, G, B) triple: neighbours in that 3D color space are
          reconstructed directly from `num_levels`/`codebook_size`, without needing a forward
          pass through the model
       :param ae -> nn.Module: the autoencoder
       :return
        embedding -> torch.FloatTensor [codebook_size, dim]
    """
    if hasattr(ae, "quantize") and hasattr(ae.quantize, "embedding"):
        return ae.quantize.embedding.weight.detach()

    if hasattr(ae, "num_levels") and hasattr(ae, "codebook_size"):
        device = device or next(ae.parameters(), torch.zeros(1)).device
        idx = torch.arange(ae.codebook_size, device=device)
        blue = torch.fmod(idx, ae.num_levels)
        green = torch.fmod(idx // ae.num_levels, ae.num_levels)
        red = idx // (ae.num_levels ** 2)
        return torch.stack([red, green, blue], dim=-1).float()

    raise AttributeError(
        "get_codebook_embedding: don't know how to extract a codebook embedding from "
        f"{type(ae).__name__}; pass the embedding matrix to build_codebook_knn directly instead."
    )
