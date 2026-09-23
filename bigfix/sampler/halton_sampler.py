import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


class HaltonSampler(object):
    """
    Halton Sampler is a sampling strategy for iterative masked token prediction in image generation models.

    It follows a Halton-based scheduling approach to determine which tokens to predict at each step.
    """

    def __init__(self, sm_temp_min=1, sm_temp_max=1, temp_pow=1, w=4, step=64, randomize=False, top_k=-1, top_p=1.0, temp_warmup=0, lin_cfg_warmup=False, scheduler="arccos", **kwargs):
        """
        Initializes the HaltonSampler with configurable parameters.

        params:
            sm_temp_min  -> float: Minimum softmax temperature.
            sm_temp_max  -> float: Maximum softmax temperature.
            w            -> float: Weight parameter for the CFG.
            step         -> int: Number of steps in the sampling process.
            randomize    -> bool: Whether to randomize the Halton sequence for the generation.
            top_k        -> int: If > 0, applies top-k sampling for token selection.
            top_p        -> int: If < 1.0, applies top-p sampling for token selection.
            temp_warmup  -> int: Number of initial steps where temperature is reduced.
        """
        super().__init__()
        self.sm_temp_min = sm_temp_min
        self.sm_temp_max = sm_temp_max
        self.temp_pow = temp_pow
        self.w = w
        self.step = step
        self.randomize = randomize
        self.top_k = top_k
        self.top_p = top_p
        self.basic_halton_mask = None  # Placeholder for the Halton-based mask
        self.temp_warmup = temp_warmup
        self.lin_cfg_warmup = lin_cfg_warmup
        self.scheduler = scheduler
        # Linearly interpolate the temperature over the sampling steps
        self.temperature = torch.linspace(self.sm_temp_min, self.sm_temp_max, self.step)

    def __str__(self):
        """Returns a string representation of the sampler configuration."""
        return f"Scheduler: Halton, Steps: {self.step}, " \
               f"sm_temp_min: {self.sm_temp_min}, sm_temp_max: {self.sm_temp_max}, w: {self.w}, " \
               f"Top_k: {self.top_k}, temp_warmup: {self.temp_warmup}"

    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None, verbose=True):
        """
        Runs the Halton-based sampling process.

        Args:
            trainer    -> MaskGIT: The model trainer.
            init_code  -> torch.Tensor: Pre-initialized latent code.
            nb_sample  -> int: Number of images to generate.
            labels     -> torch.Tensor: Class labels for conditional generation.
            verbose    -> bool: Whether to display progress.

        Returns:
            Tuple: Generated images, list of intermediate codes, list of masks used during generation.
        """

        # Build the Halton mask
        trainer.vit.eval()
        h = trainer.h
        w = trainer.w
        self.basic_halton_mask = self.build_halton_mask(h)
        l_codes = []  # List to store intermediate latent codes
        l_mask = []  # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)]*2 + [random.randint(0, 999) for _ in range(nb_sample - 20)]
                labels = torch.LongTensor(labels[:nb_sample]).to(trainer.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(trainer.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
            else:  # Initialize a code
                code = torch.full((nb_sample, h, w), trainer.args.mask_value).to(trainer.args.device)

            # Randomizing the mask sequence if enabled
            if self.randomize:
                randomize_mask = torch.randint(0,  h*w, (nb_sample,))
                halton_mask = torch.zeros(nb_sample, h*w, 2, dtype=torch.long)
                for i_h in range(nb_sample):
                    rand_halton = torch.roll(self.basic_halton_mask.clone(), randomize_mask[i_h].item(), 0)
                    halton_mask[i_h] = rand_halton
            else:
                halton_mask = self.basic_halton_mask.clone().unsqueeze(0).expand(nb_sample, h*w, 2)

            # Softmax temperature
            bar = tqdm(range(self.step), leave=False) if verbose else range(self.step)
            prev_r = 0
            for index in bar:
                # Compute the number of tokens to predict following an arccos scheduler
                r = self.compute_r(
                    index=index, total_steps=self.step, num_tokens=h * w, schedule=self.scheduler
                )

                # Construct the mask for the current step
                _mask = halton_mask.clone()[:, prev_r:r]
                mask = torch.zeros(nb_sample, h, w, dtype=torch.long)
                for i_mask in range(nb_sample):
                    mask[i_mask, _mask[i_mask, :, 0], _mask[i_mask, :, 1]] = 1
                mask = mask.bool()

                # Choose softmax temperature
                _temp = self.temperature[index] ** self.temp_pow
                if index < self.temp_warmup:
                    _temp *= 0.5  # Reduce temperature during warmup

                with trainer.autocast:
                    if self.w != 0: # Model Prediction with cfg
                        logit = trainer.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                            torch.cat([labels, labels], dim=0),
                                            torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = self.w
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = trainer.vit(code.clone(), labels, ~drop)

                pred_code, prob = self._sample(logit.view(nb_sample*h*w, -1),
                                               temperature=self.temperature[index].item(),
                                               top_k=self.top_k, top_p=self.top_p,
                                               sample_logits=True)

                # Update code with new predictions
                code[mask] = pred_code.view(nb_sample, h, w)[mask]

                l_codes.append(pred_code.view(nb_sample, h, w).clone())
                l_mask.append(mask.view(nb_sample, h, w).clone().float())

                prev_r = 0 # r

            # Decode the final prediction
            code = torch.clamp(code, 0, trainer.args.codebook_size - 1)
            x = trainer.ae.decode_code(code)
            x = torch.clamp(x, -1, 1)

        trainer.vit.train()  # Restore training mode
        return x, l_codes, l_mask

    def _sample(self, logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
        logits = logits * temperature
        if top_k > 0 or top_p < 1.0:
            logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            idx = torch.topk(probs, k=1, dim=-1)[1]
        return idx, probs

    @staticmethod
    def build_halton_mask(input_size):
        """ Generate a halton 'quasi-random' sequence in 2D.
          :param
            input_size -> int: size of the mask, (input_size x input_size).
            nb_point   -> int: number of points to be sample, it should be high to cover the full space.
            h_base     -> torch.LongTensor: seed for the sampling.
          :return:
            mask -> Torch.LongTensor: (input_size x input_size) the mask where each value corresponds to the order of sampling.
        """

        def halton(b, n_sample):
            """Naive Generator function for Halton sequence."""
            n, d = 0, 1
            res = []
            for index in range(n_sample):
                x = d - n
                if x == 1:
                    n = 1
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    n = (b + 1) * y - x
                res.append(n / d)
            return res

        nb_point = input_size*1_000
        # Sample 2D mask
        data_x = torch.asarray(halton(2, nb_point)).view(-1, 1)
        data_y = torch.asarray(halton(3, nb_point)).view(-1, 1)
        mask = torch.cat([data_x, data_y], dim=1) * input_size
        mask = torch.floor(mask)

        # remove duplicate
        indexes = np.unique(mask.numpy(), return_index=True, axis=0)[1]
        mask = [mask[index].numpy().tolist() for index in sorted(indexes)]
        return torch.LongTensor(np.array(mask))

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=-1, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
              https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/generate.py
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    @staticmethod
    def compute_r(index, total_steps, num_tokens, schedule="arccos"):
        """
        Compute number of tokens to predict at a given step.

        Args:
            index (int): current step index (0-based)
            total_steps (int): total number of steps
            num_tokens (int): total tokens (h * w)
            schedule (str): one of ["linear", "sqrt", "square", "cos", "arccos"]

        Returns:
            int: number of tokens to predict
        """
        # normalized progress in [0, 1]
        ratio = (index + 1) / total_steps

        if schedule == "linear":
            p = ratio

        elif schedule == "sqrt":
            p = math.sqrt(ratio)

        elif schedule == "square":
            p = ratio ** 2

        elif schedule == "cos":
            # smooth start, faster end
            p = 1 - math.cos(ratio * math.pi / 2)

        elif schedule == "arccos":
            # your original schedule
            p = 1 - (math.acos(ratio) / (math.pi * 0.5))

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        r = int(p * num_tokens)

        # ensure monotonic growth and at least index+1 tokens
        r = max(index + 1, r)

        # never exceed total tokens
        r = min(r, num_tokens)

        return r


class TxtHaltonSampler(HaltonSampler):
    def __init__(self, pos_cond=None, neg_cond=None, scheduler="arccos", start_correction=32, **kwargs):
        super().__init__(**kwargs)
        if neg_cond is None:
            neg_cond = [0.2, 0.2, 0.2]
        if pos_cond is None:
            pos_cond = [0.8, 0.8, 0.8]

        self.pos_cond = pos_cond
        self.neg_cond = neg_cond
        self.start_correction = start_correction
        self.scheduler = scheduler

    def __call__(self, trainer, txt_emb, init_code=None, neg_txt_emb=None, verbose=True):
        """
        Runs the Halton-based sampling process.

        Args:
            trainer    -> MaskGIT: The model trainer.
            txt_emb    -> torch.Tensor: Text condition
            verbose    -> bool: Whether to display progress.

        Returns:
            Tuple: Generated images, list of intermediate codes, list of masks used during generation.
        """

        # Build the Halton mask
        trainer.vit.eval()
        h = trainer.h
        w = trainer.w
        self.basic_halton_mask = self.build_halton_mask(h)
        l_codes = []  # List to store intermediate latent codes
        l_mask = []  # Save the intermediate masks
        nb_sample = txt_emb.size(0)
        if trainer.args.nb_class > 0:
            pos_cond = torch.FloatTensor([self.pos_cond]*nb_sample).to(trainer.args.device)
            neg_cond = torch.FloatTensor([self.neg_cond]*nb_sample).to(trainer.args.device)
        else:
            pos_cond = neg_cond = None

        if neg_txt_emb is None:
            neg_txt_emb = torch.zeros_like(txt_emb)

        with torch.no_grad():
            if init_code is not None:  # Start with a pre-define code
                code = init_code
            else:  # Initialize a code
                code = torch.full((nb_sample, h, w), trainer.args.mask_value).to(trainer.args.device)

            # Randomizing the mask sequence if enabled
            if self.randomize:
                randomize_mask = torch.randint(0,  h*w, (nb_sample,))
                halton_mask = torch.zeros(nb_sample, h*w, 2, dtype=torch.long)
                for i_h in range(nb_sample):
                    rand_halton = torch.roll(self.basic_halton_mask.clone(), randomize_mask[i_h].item(), 0)
                    halton_mask[i_h] = rand_halton
            else:
                halton_mask = self.basic_halton_mask.clone().unsqueeze(0).expand(nb_sample, h*w, 2)

            # Softmax temperature
            bar = tqdm(range(self.step), leave=False) if verbose else range(self.step)
            prev_r = 0
            for index in bar:
                r = self.compute_r(
                    index=index,
                    total_steps=self.step,
                    num_tokens=h * w,
                    schedule=self.scheduler,  # or "linear", "sqrt", "square", "cos"
                )

                # Construct the mask for the current step
                _mask = halton_mask.clone()[:, prev_r:r]
                mask = torch.zeros(nb_sample, h, w, dtype=torch.long)
                for i_mask in range(nb_sample):
                    mask[i_mask, _mask[i_mask, :, 0], _mask[i_mask, :, 1]] = 1
                mask = mask.bool()

                # Choose softmax temperature
                _temp = self.temperature[index] ** self.temp_pow
                if index < self.temp_warmup:
                    _temp *= 0.5  # Reduce temperature during warmup

                with trainer.autocast:
                    if self.w != 0: # Model Prediction with cfg
                        logit = trainer.vit(x=torch.cat([code.clone(), code.clone()], dim=0),
                                            text_embs=torch.cat([txt_emb, neg_txt_emb], dim=0),
                                            cond=torch.cat([pos_cond, neg_cond], dim=0) if trainer.args.extra_cond else None
                                            )
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        # diff = (logit_u - logit_c).abs().mean().item()
                        # print("Mean absolute logit diff when changing cond:", diff)
                        _w = ((index+1) / self.step) * self.w if self.lin_cfg_warmup else self.w
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = trainer.vit(x=code.clone(), text_embs=txt_emb, cond=pos_cond)

                pred_code, prob = self._sample(logit.view(nb_sample*h*w, -1),
                                               temperature=self.temperature[index].item(),
                                               top_k=self.top_k, top_p=self.top_p,
                                               sample_logits=True)

                # Update code with new predictions
                code[mask] = pred_code.view(nb_sample, h, w)[mask]

                l_codes.append(pred_code.view(nb_sample, h, w).clone())
                l_mask.append(mask.view(nb_sample, h, w).clone().float())

                if index > self.step - self.start_correction:
                    prev_r = 0
                else:
                    prev_r = r

            # Decode the final prediction
            code = torch.clamp(code, 0, trainer.args.codebook_size - 1)
            l_codes.append(code)
            x = trainer.ae.decode_code(code)
            x = torch.clamp(x, -1, 1)

        trainer.vit.train()  # Restore training mode
        return x, l_codes, l_mask
