# Extract feature
import os
import io
from pathlib import Path
import re
import time
import glob
import json
import random
import shutil
import zipfile
import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import webdataset as wds

import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, T5EncoderModel, AutoModel, CLIPProcessor, CLIPModel

import bigfix
from bigfix.network.vq_model import VQ_models

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


def tensor2pil(image):
    """ Transform a tensor Image into
    """
    image = ((image + 1) / 2) * 255
    image = image.permute(1, 2, 0).clip(0, 255).cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def clean_answer(raw_answer):
    pattern = r'\[INST\].*?\[\/INST\]'
    cleaned_answer = re.sub(pattern, '', raw_answer, flags=re.DOTALL).strip()
    cleaned_answer = re.sub(r'\n\n', ' ', cleaned_answer)  # delete line spaces in the text
    cleaned_answer = re.sub(r'\n', ' ', cleaned_answer)  # to have everything on one line
    return cleaned_answer


class WebDataset(object):
    def __init__(self, data_folder, bsize, img_size, data="cc12m", shard_id="{00000..01242}.tar"):
        self.shard_id = shard_id
        folder = os.path.join(data_folder, self.shard_id)
        self.img_size = img_size
        self.bsize = bsize
        if data == "cc12m":
            self.t_train = transforms.Compose([transforms.Resize((self.img_size+72, self.img_size+72), antialias=True),
                                               transforms.CenterCrop((self.img_size, self.img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                                              ])
        else:
            self.t_train = transforms.Compose([transforms.Resize(self.img_size+196, antialias=True),
                                               transforms.CenterCrop((self.img_size, self.img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                                               ])

        self.dataset = (
            wds.WebDataset(folder)
            .decode("pilrgb")
            .map(self.preprocess(data))
            .batched(bsize)
        )
        if data == "cc12m":
            # Mandatory to replace the <PERSON> tag in cc12m
            self.first_name = []
            with open(Path(bigfix.__file__).resolve().parent / "utils" / "first_name.txt", 'r') as file:
                # Iterate through each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace (like newline characters) and add the name to the list
                    self.first_name.append(line.strip())

    def preprocess(self, data):
        if data == "cc12m":
            return self.preprocess_cc12m
        elif data == "seg_any":
            return self.preprocess_seg_any
        if data == "laion":
            return self.preprocess_laion
        else:
            return

    def preprocess_cc12m(self, sample):
        keys = sample['__key__']
        images = sample['jpg']
        text = sample['txt']
        json = sample['json']

        img = self.t_train(images)

        # This will replace each <PERSON> tag with a name from the list
        caption = json['caption']
        person_tags = re.findall(r"<PERSON>", text)
        for i in range(len(person_tags)):
            name = random.randint(0, len(self.first_name)-1)
            caption = caption.replace("<PERSON>", self.first_name[name], 1)

        return keys, img, caption

    def preprocess_seg_any(self, sample):
        if "jpg" not in sample.keys():
            # set fake label
            return "", torch.zeros(3, self.img_size, self.img_size), [None]
        else:
            keys = sample['__key__']
            images = sample['jpg']
            # transform the image
            img = self.t_train(images)
            # set fake caption
            caption = [None]
            return keys, img, caption

    def preprocess_laion(self, sample):
        if "jpg" not in sample.keys():
            # set fake label
            return "", torch.zeros(3, self.img_size, self.img_size), [None]

        keys = sample['__key__']
        images = sample['jpg']
        # transform the image
        img = self.t_train(images)
        # set fake caption
        caption = [None]
        return keys, img, caption

    def __len__(self):
        return 1


class ZipDataset(Dataset):
    def __init__(self, data_folder, img_size, shard_id):
        self.data_folder = sorted(glob.glob(data_folder + "*.zip"))[shard_id]
        self.transform = transforms.Compose([transforms.Resize(img_size, antialias=True),
                                             # transforms.CenterCrop((img_size, img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                                             ])

        self.samples = self.get_image_filenames(self.data_folder, extension=".png")
        self.labels = self.get_image_filenames(self.data_folder, extension=".json")[0]
        self.labels = eval(self.read_zip(self.labels))

    def get_image_filenames(self, zip_file_path, shuffle=False, sample=1, extension=".png"):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            filenames = [
                os.path.join(zip_file_path, file_info.filename)
                for file_info in zip_ref.infolist()
                if file_info.filename.endswith(extension)
            ]

            if shuffle:
                random.shuffle(filenames)

            if sample > 1:
                filenames = filenames[::sample]

        return filenames

    @staticmethod
    def read_zip(file_path):
        with zipfile.ZipFile(os.path.dirname(file_path), "r") as zip_ref:
            with zip_ref.open(os.path.basename(file_path)) as file:
                return file.read()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        keys = img_path.split("/")[-1]

        img = self.read_zip(img_path)
        img = Image.open(io.BytesIO(img)).convert("RGB")
        img = self.transform(img)

        text = self.labels[keys]["p"]
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        return keys, img, text


class FRONT_CAM(Dataset):
    def __init__(self, data_folder, img_size=(256, 256), shard_id=None, extension=".jpg"):

        # Collect all zip files recursively
        all_zips = sorted(glob.glob(os.path.join(data_folder, "**", "*.zip"), recursive=True))

        if shard_id is not None:
            # Use only one shard (zip file)
            self.zip_files = [all_zips[shard_id]]
        else:
            self.zip_files = all_zips

        self.extension = extension
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

        # Collect all (zip_path, inner_filename) pairs
        self.samples = []
        for zip_path in self.zip_files:
            with zipfile.ZipFile(zip_path, "r") as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(self.extension):
                        self.samples.append((zip_path, info.filename))
        print(self.samples[:10])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zip_path, inner_filename = self.samples[idx]

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(inner_filename) as file:
                    img = Image.open(io.BytesIO(file.read())).convert("RGB")

            img = self.transform(img)
            key = f"{zip_path}:{inner_filename}"
            return key, img, ""

        except (UnidentifiedImageError, OSError, zipfile.BadZipFile, KeyError) as e:
            print(f"[Warning] Skipped broken file: {zip_path}:{inner_filename} ({e})")

            # Option 1: return a blank tensor of same size
            dummy_img = torch.zeros((3, self.img_size[0], self.img_size[1]))
            return f"broken:{zip_path}:{inner_filename}", dummy_img, ""


class TgzDataset(Dataset):
    def __init__(self, data_folder, img_size, shard_id):
        self.root = data_folder
        self.shard_id = shard_id
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize(img_size, antialias=True),
                                             transforms.CenterCrop((img_size, img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                                             ])
        # get img path + caption
        self.labels = []
        with open(self.root + "train_anno_realease_repath.jsonl", 'r') as json_file:
            json_list = list(json_file)
        for j in json_list:
            # Decode the line to string and parse it as JSON
            data = json.loads(j)# .decode('utf-8'))
            if data["img_path"].startswith(f"./{shard_id:03d}"):
                img_path = data["img_path"]
                img_caption = data["Task2"]["Caption"] if "Caption" in data["Task2"].keys() else None
                if img_caption is not None:
                    self.labels.append([img_path, img_caption])

        # decompress the folder if not already decompressed
        self.img_tarfile = self.root + "imgs/" + f"{self.shard_id:03d}"

        self.tar_exist = os.path.exists(f"{self.img_tarfile}.tgz")

        if not os.path.isdir(self.img_tarfile):
            os.system(f"tar -xzf {self.img_tarfile}.tgz -C {self.root + 'imgs/'}")



    @staticmethod
    def get_image_filenames(tar_file_path, extension=".png"):
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            filenames = [
                os.path.join(tar_file_path, file_info.name)
                for file_info in tar.getmembers()
                if file_info.name.endswith(extension)
            ]
        return filenames

    def remove_set(self):
        if os.path.exists(self.img_tarfile):
            shutil.rmtree(self.img_tarfile)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        keys, text = self.labels[idx]

        try:
            img = Image.open(self.root + "imgs/" + f"{self.shard_id:03d}/" + keys.split("/")[-1].split("_")[-1]).convert("RGB")
            img = self.transform(img)
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        except:
            print("not found", keys.split("/")[-1].split("_")[-1])

            return keys, torch.ones(3, self.img_size, self.img_size), text.encode('utf-8', 'ignore').decode('utf-8')

        return keys, img, text


class Extractor:

    def __init__(self, args):
        self.args = args

        self.ae = self.get_network("vqgan-llama")
        self.dino = self.get_network("dino")
        self.t5_model, self.t5_tok = self.get_network("t5")
        self.clip_model, self.clip_tok = self.get_network("clip-l")
        self.llava_model, self.llava_processor = self.get_network("llava")

        self.patch_size = self.args.img_size // 16
        self.t_dino = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),  # [-1, 1] → [0, 1]
            transforms.Resize(224, antialias=True),  # Resize shorter side to 224
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_network(self, archi):
        if archi == "vqgan-llama":
            hf_hub_download(repo_id="peizesun/llamagen_t2i", filename="vq_ds16_t2i.pt",
                            local_dir=self.args.vqgan_folder)

            # create and load model
            model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
            checkpoint = torch.load(os.path.join(self.args.vqgan_folder, "vq_ds16_t2i.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model = model.eval()
            model = model.to(self.args.device)

            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module

        elif archi == "dino":
            model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.args.device).eval()
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "clip-b":
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.args.device).eval()
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "clip-l":
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.args.device).eval()
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "t5":
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
            model = T5EncoderModel.from_pretrained("google/flan-t5-xl").to(self.args.device)
            model = model.eval()
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "t5-xxl":
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
            model = T5EncoderModel.from_pretrained("google/flan-t5-xxl").to(self.args.device)
            model = model.eval()
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "llava":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

            model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                      torch_dtype=torch.float16,
                                                                      device_map=self.args.device)
            # remove the warning about pad_token_id
            model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
            model = model.eval()
            if self.args.compile:
                model = torch.compile(model)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "blip":
            pass

        elif archi == "siglipv2":
            pass

        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        if archi in ["t5", "t5-xxl", "clip-b", "clip-l"]:
            return model, tokenizer
        elif archi == "llava":
            return model, processor
        else:
            return model

    def extract_and_save(self, shard_id):
        dest_folder = os.path.join(self.args.dest_folder, f"{shard_id:06d}.tar")

        # Check if file exists
        if os.path.exists(dest_folder):
            try:
                # Try opening the tar file
                with tarfile.open(dest_folder, "r") as tar:
                    tar.getmembers()  # try to read all members
                print(f"{dest_folder} exists and is valid, skipping shard {shard_id}")
                return 0  # exit early, file is valid

            except (tarfile.TarError, EOFError) as e:
                # File is corrupted or incomplete
                print(f"{dest_folder} exists but is corrupted ({e}), overwriting")
        else:
            print(f"{dest_folder} does not exist, will create")

        if self.args.data in ["seg_any", "cc12m", "laion", "cc3m"]:
            shards = sorted(glob.glob(os.path.join(args.data_folder, "*.tar")))
            dataset = WebDataset(data_folder=self.args.data_folder,
                                 bsize=self.args.bsize, img_size=self.args.img_size,
                                 shard_id=shards[shard_id].split("/")[-1], data=self.args.data,
                                 )
            train_loader = DataLoader(dataset.dataset, num_workers=self.args.num_workers, batch_size=None)

        elif self.args.data == "diffusiondb":
            try:
                dataset = ZipDataset(data_folder=self.args.data_folder, img_size=self.args.img_size, shard_id=shard_id)
            except:
                return
            train_loader = DataLoader(dataset, num_workers=self.args.num_workers, batch_size=self.args.bsize)

        elif self.args.data == "MidjourneyDB":
            dataset = TgzDataset(data_folder=self.args.data_folder, img_size=self.args.img_size, shard_id=shard_id)
            if not dataset.tar_exist:
                print("tar file does not exist")
                return
            train_loader = DataLoader(dataset, num_workers=self.args.num_workers, batch_size=self.args.bsize)

        elif self.args.data == "FRONT_CAM":
            dataset = FRONT_CAM(data_folder=self.args.data_folder, img_size=(512, 1024), shard_id=shard_id)
            train_loader = DataLoader(dataset, num_workers=self.args.num_workers, batch_size=self.args.bsize)
        else:
            raise "wtf are these args.data?????"

        bar = tqdm(train_loader, leave=False) if self.args.is_master else train_loader
        tmp_dest_folder = os.path.join(self.args.dest_folder, f"{shard_id:06d}_tmp.tar")
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        nb_img = 0
        with wds.TarWriter(tmp_dest_folder) as dst:
            for keys, img, text in bar:

                B, C, H, W = img.size()

                with torch.no_grad():
                    # We use Llava to generate them
                    if self.args.data in ["seg_any", "laion", "FRONT_CAM"]:
                        pil_images = [tensor2pil(image) for image in img]
                        # Generate the code using LLava Model
                        inputs = self.llava_processor(
                            pil_images, [prompt] * img.size(0), return_tensors="pt"
                        ).to(self.args.device)

                        # autoregressive complete prompt
                        output = self.llava_model.generate(**inputs, max_new_tokens=120)
                        text = [
                            clean_answer(self.llava_processor.decode(output[i], skip_special_tokens=True))
                            for i in range(img.size(0))
                        ]

                    img = img.to(self.args.device)

                    # VQGAN encoding img to tokens
                    _, _, [_, _, code] = self.ae.encode(img)
                    code = code.reshape(B, H//16, W//16)
                    code = code.detach().cpu().numpy()

                    # DINOv2 features
                    clean_img = torch.stack([self.t_dino(img) for img in img]).to(self.args.device)
                    dino_emb = self.dino(clean_img).last_hidden_state
                    dino_emb = dino_emb.detach().cpu().numpy()

                    # CLIP text features
                    clip_text_inputs = self.clip_tok(
                        text=text, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                    ).to(self.args.device)

                    clip_text_emb = self.clip_model.text_model(**clip_text_inputs).last_hidden_state
                    clip_text_emb = clip_text_emb.detach().cpu().numpy()

                    # Text extraction using T5-large
                    t5_input = self.t5_tok(
                        text, truncation=True, padding="max_length", max_length=120, return_tensors="pt"
                    ).to(self.args.device)
                    t5_txt_emb = self.t5_model(**t5_input).last_hidden_state
                    t5_txt_emb = t5_txt_emb.detach().cpu().numpy()

                    if nb_img == 0:
                        print(code.shape, dino_emb.shape, clip_text_emb.shape, t5_txt_emb.shape, flush=True)
                        print(text, flush=True)

                nb_img += B
                if self.args.is_master:
                    bar.set_postfix(nb_img=nb_img)

                for i in range(len(keys)):
                    sample = {
                        "__key__": keys[i],
                        "txt": text[i],
                        "vq_feat.npy": code[i],
                        "dino_feat.npy": dino_emb[i],
                        "t5_txt_feat.npy": t5_txt_emb[i],
                        "clip_txt_feat.npy": clip_text_emb[i],
                    }

                    dst.write(sample)

        if self.args.data == "MidjourneyDB":
            dataset.remove_set()

        os.rename(tmp_dest_folder, dest_folder)
        print(f"✅ {dest_folder} extracted successfully")
        return nb_img


def main(args):
    extractor = Extractor(args)
    start = time.time()

    for i in range(args.range_start, args.range_end, 1):

        nb_img = extractor.extract_and_save(shard_id=i)
        # Clock time
        clock_time = (time.time() - start)
        if args.is_master:
            print(f" Time shard {i:05d}.tar: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s "
                  f"Number of extracted images {nb_img}")

    clock_time = (time.time() - start)
    if args.is_master:
        print(f" Total Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,                  default="laion", help="dataset name")
    parser.add_argument("--data_folder",  type=str,                  default="", help="data source")
    parser.add_argument("--dest_folder",  type=str,                  default="", help="data destination")
    parser.add_argument("--vqgan_folder", type=str,                  default="", help="path to the VQGAN folder")

    parser.add_argument("--range-start",  type=int, default=0,       help="start of shard id (include)")
    parser.add_argument("--range-end",    type=int, default=-1,      help="end of shard id (exclude)")
    parser.add_argument("--bsize",        type=int, default=32,      help="batch size")
    parser.add_argument("--img-size",     type=int, default=256,     help="image size")
    parser.add_argument("--num-workers",  type=int, default=0,       help="number of workers for loading")
    parser.add_argument("--compile",      action='store_true',       help="compile the network, pytorch 2.0")
    args = parser.parse_args()

    # https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.range_end == -1:
        args.range_end = args.range_start + 1
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"{world_size} GPU(s) found, launch multi-gpus training")
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:
        print(f"{world_size} GPU found")
        args.is_master = True
        args.is_multi_gpus = False
        main(args)


