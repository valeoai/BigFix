import os
import glob
import webdataset as wds
from PIL import Image
import json

# Increase maximum allowed text chunk
# PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10 MB
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


from torchvision.datasets import ImageFolder

from bigfix.dataset.dataset import ImageNetKaggle
from bigfix.dataset.dataset import CodeDataset


def get_data(data, img_size, data_folder, bsize, num_workers, is_multi_gpus, seed):
    """ Class to load data """

    def _text_from_value(value):
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                txt = _text_from_value(item)
                if txt:
                    parts.append(txt)
            if not parts:
                return None
            return " ".join(parts)
        return None

    def _extract_caption_from_meta(meta):
        preferred_keys = [
            "caption", "text", "prompt", "original_prompt", "txt",
            "description", "sentence", "title",
        ]

        if isinstance(meta, dict):
            for key in preferred_keys:
                if key in meta:
                    txt = _text_from_value(meta.get(key))
                    if txt:
                        return txt

            # RenderedText shards often store OCR text as list in ocr_annotation.text
            if "ocr_annotation" in meta and isinstance(meta["ocr_annotation"], dict):
                txt = _text_from_value(meta["ocr_annotation"].get("text"))
                if txt:
                    return txt

            for value in meta.values():
                txt = _extract_caption_from_meta(value)
                if txt:
                    return txt

        elif isinstance(meta, (list, tuple)):
            return _text_from_value(meta)

        return _text_from_value(meta)

    def _decode_txt(value):
        if isinstance(value, dict):
            txt = _extract_caption_from_meta(value)
            if txt:
                return txt
        txt = _text_from_value(value)
        if txt:
            return txt
        return str(value)

    def build_t2i_webdataset(urls, tag):
        if len(urls) == 0:
            raise FileNotFoundError(
                f"No WebDataset shards found for {tag} in: {data_folder}"
            )

        t_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        def safe_handler(exn):
            print(f"[{tag}/WDS] Skipped sample: {exn}")
            return True

        def preprocess(sample):
            img = None
            for key in ["jpg", "jpeg", "png", "webp"]:
                if key in sample:
                    img = sample[key]
                    break

            if img is None:
                raise ValueError(f"No supported image key found in {tag} sample")

            txt = None
            for key in ["txt", "text", "caption", "prompt"]:
                if key in sample:
                    txt = _decode_txt(sample[key])
                    break

            if txt is None and "json" in sample:
                meta = sample["json"]
                if isinstance(meta, dict):
                    pass
                elif isinstance(meta, bytes):
                    meta = meta.decode("utf-8", errors="replace")
                    meta = json.loads(meta)
                elif isinstance(meta, str):
                    meta = json.loads(meta)
                else:
                    raise ValueError(f"Unsupported json metadata type: {type(meta)}")
                txt = _extract_caption_from_meta(meta)

            if txt is None:
                raise ValueError(f"No supported caption key found in {tag} sample")

            return {
                "img": t_train(img),
                "txt": txt,
                "reward": torch.ones(3)
            }

        return (
            wds.WebDataset(urls, resampled=True, nodesplitter=wds.split_by_node, handler=safe_handler)
            .shuffle(2_000)
            .decode("pilrgb", handler=safe_handler)
            .map(preprocess, handler=safe_handler)
            .batched(bsize)
        )

    if data == "imagenet":
        t_train = transforms.Compose([transforms.Resize(img_size),
                                      transforms.CenterCrop((img_size, img_size)),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                      ])

        t_test = transforms.Compose([transforms.Resize(img_size),
                                     transforms.CenterCrop((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                     ])

        try:
            data_train = ImageFolder(data_folder + "/train", transform=t_train)
            data_test = ImageFolder(data_folder + "val", transform=t_test)

        except:
            data_train = ImageNetKaggle(data_folder, split="train", img_size=img_size, transform=t_train)
            data_test = ImageNetKaggle(data_folder, split="val", img_size=img_size, transform=t_test)

    elif data == "imagenet_feat":
        data_train = CodeDataset(data_folder + "Train")
        data_test = CodeDataset(data_folder + "Eval")

    elif data == "gpic":
        shard_patterns = [
            os.path.join(data_folder, "train", "*.tar"),
        ]
        urls = []
        for pattern in shard_patterns:
            urls.extend(glob.glob(pattern, recursive=True))

        print(f"Found {len(urls)} shards for GPIC training corresponding to {len(urls)*12500} images.")
        urls = sorted(list(set(urls)))
        dataset = build_t2i_webdataset(urls, tag="GPIC")

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None

    elif data == "fine_t2i":
        shard_patterns = [
            os.path.join(data_folder, "Fine-T2I", "**", "*.tar"),
        ]
        urls = []
        for pattern in shard_patterns:
            urls.extend(glob.glob(pattern, recursive=True))

        urls = sorted(list(set(urls)))
        print(f"Found {len(urls)} shards for Fine-T2I training.")
        dataset = build_t2i_webdataset(urls, tag="Fine-T2I")

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None

    elif data == "rendered_text":
        shard_patterns = [
            os.path.join(data_folder, "RenderedText", "*.tar"),
        ]
        urls = []
        for pattern in shard_patterns:
            urls.extend(glob.glob(pattern, recursive=True))

        urls = sorted(list(set(urls)))
        print(f"Found {len(urls)} shards for RenderedText training.")
        dataset = build_t2i_webdataset(urls, tag="RenderedText")

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None

    elif data in ["gpic+fine_t2i"]:
        gpic_patterns = [
            os.path.join(data_folder, "GPIC", "train", "*.tar"),
        ]
        fine_patterns = [
            os.path.join(data_folder, "Fine-T2I", "**", "*.tar"),
        ]

        gpic_urls = []
        for pattern in gpic_patterns:
            gpic_urls.extend(glob.glob(pattern, recursive=True))

        fine_urls = []
        for pattern in fine_patterns:
            fine_urls.extend(glob.glob(pattern, recursive=True))

        gpic_urls = sorted(list(set(gpic_urls)))
        fine_urls = sorted(list(set(fine_urls)))

        print(f"Found {len(gpic_urls)} shards for GPIC training.")
        print(f"Found {len(fine_urls)} shards for Fine-T2I training.")

        gpic_ds = build_t2i_webdataset(gpic_urls, tag="GPIC")
        fine_ds = build_t2i_webdataset(fine_urls, tag="Fine-T2I")

        dataset = wds.RandomMix(
            [gpic_ds, fine_ds],
            probs=[0.2, 0.8],
            longest=True,
        )

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.unbatched().shuffle(1000).batched(bsize)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None
    
    elif data in ["gpic+fine_t2i+rendered_text"]:
        gpic_patterns = [
            os.path.join(data_folder, "GPIC", "train", "*.tar"),
        ]
        fine_patterns = [
            os.path.join(data_folder, "Fine-T2I", "**", "*.tar"),
        ]
        rendered_text_patterns = [
            os.path.join(data_folder, "RenderedText", "**", "*.tar"),
        ]
        rendered_text_urls = []
        for pattern in rendered_text_patterns:
            rendered_text_urls.extend(glob.glob(pattern, recursive=True))
        rendered_text_urls = sorted(list(set(rendered_text_urls)))


        gpic_urls = []
        for pattern in gpic_patterns:
            gpic_urls.extend(glob.glob(pattern, recursive=True))

        fine_urls = []
        for pattern in fine_patterns:
            fine_urls.extend(glob.glob(pattern, recursive=True))

        gpic_urls = sorted(list(set(gpic_urls)))
        fine_urls = sorted(list(set(fine_urls)))

        print(f"Found {len(gpic_urls)} shards for GPIC training.")
        print(f"Found {len(fine_urls)} shards for Fine-T2I training.")
        print(f"Found {len(rendered_text_urls)} shards for RenderedText training.")
        gpic_ds = build_t2i_webdataset(gpic_urls, tag="GPIC")
        fine_ds = build_t2i_webdataset(fine_urls, tag="Fine-T2I")
        rendered_text_ds = build_t2i_webdataset(rendered_text_urls, tag="RenderedText")

        dataset = wds.RandomMix(
            [gpic_ds, fine_ds, rendered_text_ds],
            probs=[0.2, 0.7, 0.1],
            longest=True,
        )

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.unbatched().shuffle(1000).batched(bsize)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None

    elif data == "fine_t2i":

        fine_patterns = [
            os.path.join(data_folder, "Fine-T2I", "synthetic_enhanced_prompt_square_resolution", "*.tar"),
            os.path.join(data_folder, "Fine-T2I", "curated", "*.tar"),
        ]

        fine_urls = []
        for pattern in fine_patterns:
            fine_urls.extend(glob.glob(pattern, recursive=True))

        fine_urls = sorted(list(set(fine_urls)))

        print(f"Found {len(fine_urls)} shards for Fine-T2I training.")

        dataset = build_t2i_webdataset(fine_urls, tag="Fine-T2I")

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.unbatched().shuffle(1000).batched(bsize)
        train_loader = train_loader.with_epoch(10_000)

        return train_loader, None


    else:
        data_train = None
        data_test = None

    train_sampler = DistributedSampler(data_train, shuffle=True, seed=seed) if is_multi_gpus else None
    train_loader = DataLoader(data_train, batch_size=bsize,
                              shuffle=False if is_multi_gpus else True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, sampler=train_sampler)

    if data_test is not None:
        test_sampler = DistributedSampler(data_test, shuffle=True, seed=seed) if is_multi_gpus else None
        test_loader = DataLoader(data_test, batch_size=bsize,
                                 shuffle=False if is_multi_gpus else True,
                                 num_workers=num_workers, pin_memory=True,
                                 drop_last=True, sampler=test_sampler)
    else:
        test_loader = None
    return train_loader, test_loader