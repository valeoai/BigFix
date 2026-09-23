import cv2
import yaml
from argparse import Namespace
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

# YAML configs shipped with the package (bigfix/config/)
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def compute_sharpness(pil_img):
    """
    Sharpness via Laplacian variance
    """
    gray = np.array(pil_img.convert("L"))
    gray = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gray = np.log1p(gray)
    return np.clip((gray - 1.0) / (5.0 - 1.0), 0.0, 1.0)


def compute_colorfulness(pil_img):
    """
    Hasler–Süsstrunk colorfulness metric
    """
    img = np.array(pil_img).astype(np.float32)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()

    img = float(
        np.sqrt(std_rg**2 + std_yb**2) +
        0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    )
    return np.clip(img / 100.0, 0.0, 1.0)


def compute_contrast(pil_img):
    """
    Contrast as grayscale standard deviation
    """
    gray = np.array(pil_img.convert("L")).astype(np.float32) / 255.0
    gray = float(gray.std())
    return np.clip(gray / 0.5, 0.0, 1.0)


class RandomCropWithCoords:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # original size BEFORE resize
        orig_w, orig_h = img.size

        # sample crop params
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.size, self.size)
        )

        img = TF.crop(img, i, j, h, w)

        meta = {
            "orig_size": (orig_h, orig_w),
            "crop": (i, j, h, w)  # top, left, height, width
        }

        return img, meta


class TrainTransformWithCoordsAndAesthetics:
    def __init__(self, img_size):
        self.resize = transforms.Resize(img_size)
        self.crop = RandomCropWithCoords(img_size)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __call__(self, img):
        # resize
        img = self.resize(img)

        # crop + coords
        img, meta = self.crop(img)

        # compute aesthetics (before tensor / normalization)
        meta["score"] = 0.4 * compute_sharpness(img) + 0.3 * compute_colorfulness(img) + 0.3 * compute_contrast(img)

        # to tensor + normalize
        img = self.norm(self.to_tensor(img))

        return img, meta


def resolve_config_path(config_path):
    """
    Locate a YAML config file.

    An existing path (absolute, or relative to the working directory) is used as is. Otherwise the
    name is looked up in the configs shipped with the package (bigfix/config/), so that
    "xlarge_txt2img_gpic.yaml" works from any directory once bigfix is installed.

    Args:
        config_path (str | os.PathLike): Path to a YAML file, or the name of a packaged config.

    Returns:
        Path: The resolved path of the YAML file.
    """
    path = Path(config_path)
    if path.is_file():
        return path
    packaged = CONFIG_DIR / path.name
    if packaged.is_file():
        return packaged
    available = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    raise FileNotFoundError(f"Config '{config_path}' not found. Packaged configs: {available}")


def load_args_from_file(config_path):
    """
    Load arguments from a YAML file and convert them into a Namespace.

    Args:
        config_path (str): Path to the YAML config file, or the name of a config shipped in bigfix/config/.

    Returns:
        Namespace: Arguments loaded as a Namespace object.
    """
    # Load YAML config file
    with open(resolve_config_path(config_path), "r") as file:
        config = yaml.safe_load(file)

    # Convert dictionary to Namespace
    args = Namespace(**config)
    return args

