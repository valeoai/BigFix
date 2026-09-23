# Trainer for Txt-to-Img MaskGIT
import os
import random
import time
from datetime import datetime

from tqdm import tqdm
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, T5EncoderModel

from bigfix.trainer.abstract_trainer import Trainer
from bigfix.network.txt_transformer import Transformer
from bigfix.network.vq_model import VQ_models
from bigfix.dataset.dataloader import get_data
from bigfix.sampler.halton_sampler import TxtHaltonSampler as HaltonSampler
from bigfix.utils.viz import reconstruction
from bigfix.utils.masking_scheduler import get_mask_code
from bigfix.utils.reward_utils import extract_reward_from_img_txt


def capitalize_first(s: str):
    return s[:1].upper() + s[1:] if s else s

def ensure_period(s: str) -> str:
    s = s.rstrip()  # remove trailing spaces
    if not s.endswith("."):
        s += "."
    return s

def expand_if_single_word(text: str) -> str:
    # Remove extra whitespace
    cleaned = text.strip()
    # Count words
    words = cleaned.split()

    if len(words) < 3:
        return f"An image of {cleaned}, realistic, high-quality."
    else:
        return cleaned


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        print(f"Init Txt-to-Image Maskgit on [GPU{args.global_rank}]")

        self.args = args                                                        # Main argument see bigfix/main.py
        self.input_size = self.args.img_size // self.args.f_factor              # Define input size for transformer
        self.h = self.w = self.input_size

        # Load transformer (Masked Bidirectional Transformer) and VQGAN models
        self.vit = self.get_network("vit")  # Load Masked Bidirectional Transformer
        self.ae = self.get_network("vqgan-llama")  # Load VQGAN
        skip_t5_init = getattr(self.args, "skip_t5_init", False)
        if (not skip_t5_init): #  and self.args.data in ["laion_raw", "finetune" "cc12m_raw", "imagenet_txt", "synthetics", "gpic+fine_t2i", "gcip"]:
            self.t5_model, self.t5_tok = self.get_network("t5")

        # Define loss function and optimizer
        self.criterion = self.get_loss("cross_entropy", ignore_index=-100)  # Get cross-entropy loss
        self.optim = None

        # Set up automatic mixed precision for training efficiency
        if self.args.device != 'cpu' and self.args.dtype == "bfloat16":
            self.autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        self.train_data, self.test_data = None, None

        # Optional on-the-fly reward extraction (enabled by flag or when nb_class > 3).
        self.compute_reward_on_the_fly = bool(
            getattr(self.args, "compute_reward_on_the_fly", False)
            or getattr(self.args, "nb_class", 0) > 3
        )
        self.reward_drop_prob = float(getattr(self.args, "drop_reward", 0.5))
        self.reward_models_ready = False
        self.reward_init_attempted = False
        self.reward_models = {}
        self._reward_quantiles = {
            "clip": [0.12874773, 0.26218823, 0.29246536, 0.31649169, 0.42867613],
            "pick": [0.15627505, 0.18532692, 0.19062732, 0.19668772, 0.22934763],
            "imagereward": [-2.0559988, 0.47728617, 1.14411628, 1.62788853, 2.0071547],
        }
        if self.args.is_master and self.compute_reward_on_the_fly:
            print("[Reward] On-the-fly reward computation enabled.")

        # Sampler selection
        if args.sampler == "halton":
            self.sampler = HaltonSampler(sm_temp_min=self.args.sm_temp_min, sm_temp_max=self.args.sm_temp,
                                         temp_pow=1, temp_warmup=self.args.temp_warmup, w=self.args.cfg_w,
                                         step=self.args.step, randomize=self.args.randomize,
                                         top_k=self.args.top_k, top_p=self.args.top_p)
        else:
            self.sampler = None

        if self.args.is_master:
            print("Parameters: ", self.args)
            args_str = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
            self.log_add_txt("Parameters", args_str, self.args.iter)

    def _init_on_the_fly_reward_models(self):
        if self.reward_models_ready or self.reward_init_attempted:
            return

        self.reward_init_attempted = True

        try:
            from bigfix.utils.rewards import (
                CLIPRealismReward,
                PickaPickReward,
                ImageRewardModel,
            )

            self.reward_models = {
                "clip": CLIPRealismReward(device=self.args.device).to(self.args.device),
                "pick": PickaPickReward(device=self.args.device).to(self.args.device),
                "imagereward": ImageRewardModel().to(self.args.device),
            }
            self.reward_models_ready = True
            if self.args.is_master:
                print("[Reward] Loaded CLIP/PickaPick/ImageReward backends.")
        except Exception as exc:
            self.reward_models_ready = False
            self.reward_models = {}
            if self.args.is_master:
                print(f"[Reward] Falling back to dataset rewards (backend init failed): {exc}")

    def _normalize_reward_shape(self, reward, bsz):
        if reward is None:
            return torch.ones((bsz, 3), device=self.args.device, dtype=torch.float32)

        if not torch.is_tensor(reward):
            reward = torch.as_tensor(reward, device=self.args.device, dtype=torch.float32)
        else:
            reward = reward.to(self.args.device, dtype=torch.float32)

        if reward.ndim == 0:
            reward = reward.view(1, 1).repeat(bsz, 3)
        elif reward.ndim == 1:
            if reward.size(0) == bsz:
                reward = reward.view(bsz, 1).repeat(1, 3)
            elif reward.size(0) == 3:
                reward = reward.view(1, 3).repeat(bsz, 1)
            else:
                reward = reward.view(1, -1).repeat(bsz, 1)
        else:
            reward = reward.view(bsz, -1)

        if reward.size(1) < 3:
            reward = torch.cat([reward, reward[:, -1:].repeat(1, 3 - reward.size(1))], dim=1)
        elif reward.size(1) > 3:
            reward = reward[:, :3]

        return reward.contiguous()

    @torch.no_grad()
    def _compute_on_the_fly_reward(self, images, prompts, fallback_reward):
        if (not self.compute_reward_on_the_fly) or (images is None):
            bsz = images.size(0) if images is not None else len(prompts)
            return self._normalize_reward_shape(fallback_reward, bsz)

        self._init_on_the_fly_reward_models()
        if not self.reward_models_ready:
            return self._normalize_reward_shape(fallback_reward, images.size(0))

        try:
            q_clip = torch.tensor(self._reward_quantiles["clip"], device=self.args.device, dtype=torch.float32)
            q_pick = torch.tensor(self._reward_quantiles["pick"], device=self.args.device, dtype=torch.float32)
            q_ir = torch.tensor(self._reward_quantiles["imagereward"], device=self.args.device, dtype=torch.float32)

            s_clip = self.reward_models["clip"](images, prompts).flatten().to(self.args.device, dtype=torch.float32)
            s_pick = self.reward_models["pick"](images, prompts).flatten().to(self.args.device, dtype=torch.float32)
            s_ir = self.reward_models["imagereward"](images, prompts).flatten().to(self.args.device, dtype=torch.float32)

            r_clip = torch.bucketize(s_clip, q_clip)
            r_pick = torch.bucketize(s_pick, q_pick)
            r_ir = torch.bucketize(s_ir, q_ir)

            # bucketize with N thresholds yields classes in [0, N], so normalize by (quantize_size - 1) == N
            reward_divisor = float(max(q_clip.numel()-1, q_pick.numel()-1, q_ir.numel()-1))
            reward = torch.stack([r_clip, r_pick, r_ir], dim=1).float() / reward_divisor
            return reward.contiguous()
        except Exception as exc:
            if self.args.is_master:
                print(f"[Reward] Falling back to dataset rewards (runtime failure): {exc}")
            return self._normalize_reward_shape(fallback_reward, images.size(0))


    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder|ema, the architecture to load
            :return
                model -> nn.Module: the network
        """
        preprocess = None
        if archi == "vit":
            if self.args.is_master:
                if not os.path.exists(self.args.vit_folder):
                    os.makedirs(self.args.vit_folder)
                    print(f"Folder created: {self.args.vit_folder}")

            # Define transformer architecture parameters
            hidden_dim, depth, heads = self.transformer_size(self.args.vit_size)
            model = Transformer(
                input_size=self.input_size,
                hidden_dim=hidden_dim, codebook_size=self.args.codebook_size,
                depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=self.args.dropout,
                register=self.args.register, proj=self.args.proj, extra_cond=self.args.extra_cond,
            )

            # Load model checkpoint if resuming training
            if self.args.resume:
                if os.path.isfile(self.args.vit_folder + "current.pth"):
                    ckpt = self.args.vit_folder + "current.pth"
                else:
                    ckpt = self.args.vit_folder

                if os.path.isfile(ckpt):
                    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                    state_dict = checkpoint['model_state_dict']
                    # state_dict.pop("tok_emb.weight", None)
                    # state_dict.pop("pos_emb.weight", None)
                    new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict, strict=False)

                    # Update the current epoch and iteration
                    self.args.iter = checkpoint['iter']
                    self.args.global_epoch = checkpoint['global_epoch']

                    if self.args.is_master:
                        print("Load ckpt from:", ckpt)
                        print("Number of iteration(s):", self.args.iter)
                else:
                    if self.args.is_master:
                        print("No checkpoint found")

        elif archi == "vqgan-llama":
            # Initialize and load VQGAN model
            model = VQ_models[f"VQ-{self.args.f_factor}"](codebook_size=16384, codebook_embed_dim=8)
            checkpoint = torch.load(self.args.vqgan_folder, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model = model.eval()

        elif archi == "t5":
            preprocess = AutoTokenizer.from_pretrained("google/flan-t5-xl")
            model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
            model = model.eval()
        else:
            model = None

        model = model.to(self.args.device)

        if self.args.compile: # Enable model compilation if using PyTorch 2.0+
            model = torch.compile(model)

        if self.args.is_multi_gpus and archi == "vit":  # Enable multi-GPU training if available
            model = DDP(model, device_ids=[self.args.device])

        if archi != "vit": # disable the training for the rest
            for p in model.parameters():
                p.requires_grad = False

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        if preprocess is not None:
            return model, preprocess
        else:
            return model

    def train(self, log_iter=5_000):
        """ Train the model for one epoch """
        self.vit.train()
        cum_loss, cum_acc, i_bar = 0., 0., 0.
        # Deque to store loss and accuracy over a moving window
        window_loss = deque(maxlen=self.args.grad_cum)
        window_acc = deque(maxlen=self.args.grad_cum)

        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        for i_bar, data in enumerate(bar):
            # Determine whether to update gradients based on gradient accumulation steps
            update_grad = (i_bar % self.args.grad_cum) == self.args.grad_cum - 1
            # Adjust the learning rate with warmup and cosine decay for the 10% last iter
            self.adapt_learning_rate()

            if self.args.data in ["cc12m", "cc12m_raw"]:
                data = {
                    "txt": data[0],
                    "code": data[1],
                    "txt_emb": data[2],
                    "reward": data[3]
                }

            elif self.args.data in ["laion_raw", "finetune", "cc12m_raw"]:
                data["txt"] = [
                    ensure_period(capitalize_first(expand_if_single_word(t.decode("utf-8", errors="replace"))))
                    for t in data["txt"]
                ]
                t5_input = self.t5_tok(data["txt"], truncation=True, padding="max_length", max_length=120, return_tensors="pt").to(self.args.device)
                with self.autocast:  # Perform forward pass using mixed precision (if available)
                    data["txt_emb"] = self.t5_model(**t5_input).last_hidden_state

            elif self.args.data in ["imagenet_txt", "synthetics", "gpic+fine_t2i", "gcip", "gpic+fine_t2i", "gpic+fine_t2i+rendered_text"]:

                # VQGAN encoding img to tokens
                _, _, [_, _, code] = self.ae.encode(data["img"].to(self.args.device))
                data["code"] = code.reshape(-1, self.h, self.w)

                data["txt"] = [
                    ensure_period(capitalize_first(expand_if_single_word(t)))
                    for t in data["txt"]
                ]
                t5_input = self.t5_tok(data["txt"], truncation=True, padding="max_length", max_length=120, return_tensors="pt").to(self.args.device)
                with self.autocast:  # Perform forward pass using mixed precision (if available)
                    data["txt_emb"] = self.t5_model(**t5_input).last_hidden_state


            code = data["code"].long().to(self.args.device)
            txt_emb = data["txt_emb"].to(self.args.device)

            img_for_reward = data.get("img", None)
            if img_for_reward is not None:
                img_for_reward = img_for_reward.to(self.args.device)
            reward = self._compute_on_the_fly_reward(
                images=img_for_reward,
                prompts=data.get("txt", []),
                fallback_reward=data.get("reward", None),
            )

            b = code.size(0)
            reward = self._normalize_reward_shape(reward, b)

            b, h, w = code.size()

            # Randomly drop conditions for conditional generation (CFG)
            drop_text = torch.rand(b, device=self.args.device) < self.args.drop_label
            txt_emb_d = torch.where(drop_text.view(b, 1, 1), torch.zeros_like(txt_emb), txt_emb)
            # Randomly drop reward for conditional generation (CFG)
            drop_reward = torch.rand_like(reward.float()) < self.reward_drop_prob
            reward_d = reward.masked_fill(drop_reward, -1)

            # Apply masking to encoded codes
            masked_code, mask, loss_ign = get_mask_code(
                code, value=self.args.mask_value, codebook_size=self.args.codebook_size,
                mode=self.args.sched_mode, p_resample=0.2)

            # Create target code, replacing ignored tokens with -100
            target_code = torch.where(loss_ign, code.detach().clone(), -100)

            with self.autocast:  # Perform forward pass using mixed precision (if available)
                # Model prediction
                pred = self.vit(x=masked_code, text_embs=txt_emb_d, cond=reward_d)
                # Compute loss using cross-entropy over masked tokens
                loss = self.criterion(pred.reshape(b * h * w, self.args.codebook_size + 1),
                                      target_code.reshape(b * h * w)).contiguous()

                # Normalize loss for gradient accumulation
                loss = loss / self.args.grad_cum

            loss.backward()

            # Perform gradient update if gradient accumulation step is reached
            if update_grad:
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.args.grad_clip)  # Clip gradient
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            # Compute running loss and accuracy
            cum_loss += loss.cpu().item() * self.args.grad_cum
            acc = torch.max(pred.reshape(b * h * w, self.args.codebook_size + 1).data, 1)[1]
            acc = (acc.view(b * h * w) == code.view(b * h * w)).float()[mask.view(b * h * w)]
            cum_acc += acc.mean().cpu().item()
            window_loss.append(loss.cpu().item() * self.args.grad_cum)
            window_acc.append(acc.mean().item())

            # Logging and visualization
            if update_grad:
                if self.args.is_multi_gpus:  # Synchronize logs across multiple GPUs if applicable
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                    mini_batch_acc = self.all_gather(torch.tensor(window_acc).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()
                    mini_batch_acc = torch.tensor(window_acc).mean()

                if self.args.is_master:  # Master process logs metrics
                    self.log_add_scalar('Train/MiniBatchLoss', mini_batch_loss, self.args.iter)
                    self.log_add_scalar('Train/MiniBatchAcc', mini_batch_acc, self.args.iter)
                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)

                # Save model and visualize samples periodically
                if self.args.iter % log_iter == 0 and self.args.is_master:
                    if (not self.args.skip_t5_init):
                        default_prompt = [
                            "An elephant riding a vintage bicycle through a bustling city street, hyper-realistic detail, 4k UHD.",
                            "A joyful man walking his loyal dog through a picturesque park at sunset, cinematic lighting.",
                            "A mystical fox in an enchanted forest, glowing flora, and soft mist, rendered in Unreal Engine.",
                            "A sleek airplane soaring above the clouds during a vibrant sunset, with a stunning view of the horizon.",
                            "An underwater paradise teeming with colorful, exotic fish swimming through coral reefs in crystal-clear ocean water.",
                            "A Pikachu enjoying an elegant five-star meal with a breathtaking view of the Eiffel Tower, during a golden sunset.",
                            "A towering mecha robot overlooking a vibrant favela, painted in bold, abstract expressionist style.",
                            "An insect-robot chef expertly crafting a gourmet meal in a high-tech futuristic kitchen, intricate details.",
                            "A cozy wooden cabin perched on a snowy mountain peak, glowing warmly in the night, styled like a classic Disney movie, featured on ArtStation.",
                            "A cute little matte low poly isometric cherry blossom forest island, waterfalls, lighting, soft shadows, trending on Artstation, 3D render, Monument Valley, Fez video game.",
                            "A shanty version of Tokyo, new rustic style, bold colors with all colors palette, video game, Genshin, tribe, fantasy, Overwatch.",
                            "A cozy gingerbread house nestled in a dusting of powdered sugar snow adorned with vibrant candy canes and shimmering gumdrops.",
                            "A teddy bear wearing a blue ribbon taking a selfie in a small boat in the center of a lake.",
                            "A pirate ship trapped in a cosmic maelstrom nebula rendered in cosmic beach whirlpool engine.",
                            "Volumetric lighting, spectacular ambient lights, light pollution, cinematic atmosphere, Art Nouveau style illustration art, artwork by SenseiJaye, intricate detail.",
                            "A blue jay stops on the top of a helmet of a Japanese samurai, background with sakura tree.",
                            "A shark flying in the desert.",
                            "A tiny astronaut hatching from an transparent egg on the moon.",
                            "An old-world galleon navigating through turbulent ocean waves under a stormy sky lit by flashes of lightning.",
                            "An oil painting of rain at a traditional Chinese town.",
                            "Portrait photo of an Asian old warrior chief, tribal panther makeup, blue on red, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting.",
                            "A female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She wears a dark-colored outfit with subtle patterns. The overall style of the artwork is reminiscent of fantasy or supernatural genres",
                            "Digital art, portrait of an anthropomorphic roaring Tiger warrior with full armor, close up in the middle of a battle, behind him there is a banner with the text 'Open Source'.",
                            "Photo of a dog and a cat both standing on a red box, with a blue ball in the middle with a parrot standing on top of the ball. The box has the text 'Victory' written on it",
                            "Selfie photo of a wizard with long beard and purple robes, he is apparently in the middle of Tokyo. Probably taken from a phone.",
                            "A vibrant street wall covered in colorful graffiti, the centerpiece spells 'HALTON', in a storm of colors.",
                            "Photo of a young woman with long, wavy brown hair tied in a bun and glasses. She has a fair complexion and is wearing subtle makeup, emphasizing her eyes and lips. She is dressed in a black top. The background appears to be an urban setting with a building facade, and the sunlight casts a warm glow on her face.",
                            "Anime art of a steampunk inventor in their workshop, surrounded by gears, gadgets, and steam. He is holding a blue potion and a red potion, one in each hand",
                            "Photo of picturesque scene of a road surrounded by lush green trees and shrubs. The road is wide and smooth, leading into the distance. On the right side of the road, there's a blue sports car parked with the license plate spelling 'SD32B'. The sky above is partly cloudy, suggesting a pleasant day. The trees have a mix of green and brown foliage. There are no people visible in the image. The overall composition is balanced, with the car serving as a focal point.",
                            "Photo of a young man in a black suit, white shirt, and black tie. He has a neatly styled haircut and is looking directly at the camera with a neutral expression. The background consists of a textured wall with horizontal lines. The photograph is in black and white, emphasizing contrasts and shadows. The man appears to be in his late twenties or early thirties, with fair skin and short, dark hair.",
                            "Photo of a woman on the beach, shot from above. She is facing the sea, while wearing a white dress. She has long blonde hair",
                        ]
                        t5_input = self.t5_tok(random.sample(default_prompt, 8), truncation=True, padding="max_length", max_length=120,
                                               return_tensors="pt").to(self.args.device)
                        with self.autocast:  # Perform forward pass using mixed precision (if available)
                            txt_emb_sampling = self.t5_model(**t5_input).last_hidden_state
                    else:
                        txt_emb_path = f"./saved_networks/txt_emb_template_{random.randint(0, 3)}.pt"
                        txt_emb_sampling = torch.load(txt_emb_path)  # shape: (N, L, D)
                        txt_emb_sampling = txt_emb_sampling.to(self.args.device)  # move to GPU if needed

                    gen_sample = self.sampler(self, txt_emb=txt_emb_sampling)[0][:10]
                    reco_sample = reconstruction(self, code=code[:10], unmasked_code=pred.argmax(dim=-1)[:10], mask=mask[:10])
                    self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                    self.log_add_txt("Images/Caption", "\n".join(f"- {c}" for c in data["txt"][:10]), self.args.iter)

                    # Save the current model state
                    self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, "current.pth"),
                                      optimizer=self.optim, iter=self.args.iter, global_epoch=self.args.global_epoch)

                # Increment global iteration counter
                self.args.iter += 1
            # Stop training if max iterations are reached
            if self.args.iter > self.args.max_iter:
                if self.args.is_master:
                    print("End of training: reached max iterations")
                avg_loss = cum_loss / (i_bar + 1e-4)
                avg_acc = cum_acc / (i_bar + 1e-4)
                return {"done": True, "loss": avg_loss, "accuracy": avg_acc}

        avg_loss = cum_loss / (i_bar + 1e-4)
        avg_acc = cum_acc / (i_bar + 1e-4)
        return {"done": False, "loss": avg_loss, "accuracy": avg_acc}

    def fit(self):
        """ Train the model """

        # Load training and testing data if specified
        self.train_data, self.test_data = get_data(
            self.args.data, self.args.img_size, self.args.data_folder, self.args.bsize,
            self.args.num_workers, self.args.is_multi_gpus, self.args.seed
        )
        # Load optim
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.999), weight_decay=0.03)  # Get AdamW Optimizer
        if self.args.is_master:
            print("Start training:")

        finished = False
        while not finished:
            start = time.time()
            train_stats = self.train(log_iter=self.args.log_iter)
            clock_time = (time.time() - start)
            finished = train_stats["done"]
            train_loss = train_stats["loss"]
            train_accuracy = train_stats["accuracy"]

            if self.args.is_master:
                now = datetime.now()
                print(f"\rIter {self.args.iter},"
                      f" Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f},"
                      f" Time: {int(clock_time // 3600):.0f}:{int((clock_time % 3600) // 60):02d}:{int(clock_time % 60):02d},"
                      f" Date: {now.date()} {now.hour:02}:{now.minute:02}")

        if self.args.is_master:
            print(f"End of training!")

    def sample_test(self, folder, save_emb=False):
        default_promt = [
           "an elephant riding a vintage bicycle through a bustling city street, hyper-realistic detail, 4k UHD.",
           "a joyful man walking his loyal dog through a picturesque park at sunset, cinematic lighting.",
           "a mystical fox in an enchanted forest, glowing flora, and soft mist, rendered in Unreal Engine.",
           "a sleek airplane soaring above the clouds during a vibrant sunset, with a stunning view of the horizon.",
           "an underwater paradise teeming with colorful, exotic fish swimming through coral reefs in crystal-clear ocean water.",
           "pikachu enjoying an elegant five-star meal with a breathtaking view of the Eiffel Tower, during a golden sunset.",
           "a towering mecha robot overlooking a vibrant favela, painted in bold, abstract expressionist style.",
           "an insect-robot chef expertly crafting a gourmet meal in a high-tech futuristic kitchen, intricate details.",
           "a cozy wooden cabin perched on a snowy mountain peak, glowing warmly in the night, styled like a classic Disney movie, featured on ArtStation.",
           "a cute little matte low poly isometric cherry blossom forest island, waterfalls, lighting, soft shadows, trending on Artstation, 3D render, Monument Valley, Fez video game.",
           "a shanty version of Tokyo, new rustic style, bold colors with all colors palette, video game, Genshin, tribe, fantasy, Overwatch.",
           "a cozy gingerbread house nestled in a dusting of powdered sugar snow adorned with vibrant candy canes and shimmering gumdrops.",
           "a teddy bear wearing a blue ribbon taking a selfie in a small boat in the center of a lake.",
           "a pirate ship trapped in a cosmic maelstrom nebula rendered in cosmic beach whirlpool engine.",
           "volumetric lighting, spectacular ambient lights, light pollution, cinematic atmosphere, Art Nouveau style illustration art, artwork by SenseiJaye, intricate detail.",
           "a blue jay stops on the top of a helmet of a Japanese samurai, background with sakura tree.",
           "a shark flying in the desert.",
           "an old-world galleon navigating through turbulent ocean waves under a stormy sky lit by flashes of lightning.",
           "an oil painting of rain at a traditional Chinese town.",
           "portrait photo of an Asian old warrior chief, tribal panther makeup, blue on red, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting.",
           "a female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She wears a dark-colored outfit with subtle patterns. The overall style of the artwork is reminiscent of fantasy or supernatural genres",
           "digital art, portrait of an anthropomorphic roaring Tiger warrior with full armor, close up in the middle of a battle, behind him there is a banner with the text 'Open Source'.",
           "photo of a dog and a cat both standing on a red box, with a blue ball in the middle with a parrot standing on top of the ball. The box has the text 'Victory' written on it",
           "selfie photo of a wizard with long beard and purple robes, he is apparently in the middle of Tokyo. Probably taken from a phone.",
           "a vibrant street wall covered in colorful graffiti, the centerpiece spells 'HALTON', in a storm of colors",
           "photo of a young woman with long, wavy brown hair tied in a bun and glasses. She has a fair complexion and is wearing subtle makeup, emphasizing her eyes and lips. She is dressed in a black top. The background appears to be an urban setting with a building facade, and the sunlight casts a warm glow on her face.",
           "anime art of a steampunk inventor in their workshop, surrounded by gears, gadgets, and steam. He is holding a blue potion and a red potion, one in each hand",
           "photo of picturesque scene of a road surrounded by lush green trees and shrubs. The road is wide and smooth, leading into the distance. On the right side of the road, there's a blue sports car parked with the license plate spelling 'SD32B'. The sky above is partly cloudy, suggesting a pleasant day. The trees have a mix of green and brown foliage. There are no people visible in the image. The overall composition is balanced, with the car serving as a focal point.",
           "photo of a young man in a black suit, white shirt, and black tie. He has a neatly styled haircut and is looking directly at the camera with a neutral expression. The background consists of a textured wall with horizontal lines. The photograph is in black and white, emphasizing contrasts and shadows. The man appears to be in his late twenties or early thirties, with fair skin and short, dark hair.",
           "photo of a woman on the beach, shot from above. She is facing the sea, while wearing a white dress. She has long blonde hair",
        ]
        aug_prompt = ". The image is ultra-detailed, with cinematic lighting, high clarity, intricate textures, and beautifully composed. Emphasize depth and dynamic shadows for a lifelike, visually captivating result."
        default_promt = [_t + aug_prompt for _t in default_promt]
        self.vit.eval()
        bsize = 10
        if not os.path.exists(folder):
            os.makedirs(folder)

        t5_model, t5_tok = self.get_network("t5")
        with torch.no_grad():
            images = torch.zeros(len(default_promt), 3, self.args.img_size, self.args.img_size)
            cpt = 0
            _tmp = 0
            for i in range(0, len(default_promt), bsize):
                txt = default_promt[cpt:cpt + bsize]
                # Text extraction using T5-large
                t5_input = t5_tok(txt, truncation=True, padding="max_length", max_length=120, return_tensors="pt").to(self.args.device)
                txt_emb = t5_model(**t5_input).last_hidden_state
                if save_emb:
                    torch.save(txt_emb.cpu(), os.path.join(folder, f"../saved_networks/txt_emb_template_{_tmp}.pt"))
                gen_sample = self.sampler(self, txt_emb=txt_emb)[0]
                images[cpt:cpt + bsize] = gen_sample
                cpt += bsize
                _tmp += 1

            x = vutils.make_grid(images, nrow=10, padding=0, normalize=True)
            save_image(x, os.path.join(folder, f"test_{datetime.now()}.jpg"))

        del t5_model
        del t5_tok