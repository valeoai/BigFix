import os
import json
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


# -------------------------------------------------
# Utils
# -------------------------------------------------
def clean_answer(raw_answer: str) -> str:
    raw_answer = re.sub(r'\[INST\].*?\[\/INST\]', '', raw_answer, flags=re.DOTALL)
    raw_answer = re.sub(r'\s+', ' ', raw_answer).strip()
    raw_answer = re.sub(
        r'^(In (the|this) (image|photo|picture|scene)[,:]?\s*|'
        r'The (image|photo|picture|scene) (shows|depicts|features)[,:]?\s*)',
        '',
        raw_answer,
        flags=re.IGNORECASE
    )
    return raw_answer[:1].upper() + raw_answer[1:]


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class ImageNetDataset(Dataset):
    def __init__(self, root_dir: str):
        self.image_paths = sorted(Path(root_dir).rglob("*.JPEG"))
        print(f"[Dataset] Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return {"image": image, "path": str(path)}


# -------------------------------------------------
# Caption generator
# -------------------------------------------------
@torch.no_grad()
def generate_caption(model, processor, images, device):
    prompt = (
        "[INST] <image>\nAnalyze the following image in detail. Identify all prominent objects, " 
             "their attributes (color, material, shape, size, texture), their " 
             "spatial relationships, the overall scene and setting, the lighting " 
             "conditions, and any relevant style or composition details. Based on your analysis, " 
             "generate a caption of the image. It should be descriptive enough to allow a diffusion " 
             "model to accurately reconstruct the image. Include specific details rather than general " 
             "descriptions. For example, instead of ’a blue car,’ describe it as ’a shiny, dark blue "
             "vintage sedan with chrome bumpers parked on a cobblestone street. [/INST]"
    )

    inputs = processor(
        images=images,
        text=[prompt] * len(images),
        return_tensors="pt",
        padding=True
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    captions = [
        clean_answer(processor.decode(out, skip_special_tokens=True))
        for out in output_ids
    ]

    return captions


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Paths
    imagenet_root = "path/to/ImageNet"
    output_jsonl = f"imagenet_captions_rank{rank}.jsonl"

    # Load model
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16
    ).to(device)

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()

    # Dataset + Sampler
    dataset = ImageNetDataset(imagenet_root)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    def collate_fn(batch):
        return {
            "image": [b["image"] for b in batch],
            "path": [b["path"] for b in batch]
        }

    loader = DataLoader(
        dataset,
        batch_size=12,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Inference
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"Rank {rank}", disable=rank != 0):
            images = batch["image"]
            paths = batch["path"]

            captions = generate_caption(model, processor, images, device)

            for p, c in zip(paths, captions):
                f.write(json.dumps(
                    {"image_path": p, "caption": c},
                    ensure_ascii=False
                ) + "\n")

    dist.barrier()
    if rank == 0:
        print("✅ Caption generation finished")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
