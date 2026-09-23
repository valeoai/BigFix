import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, num_levels=21, learnable=False, f_factor=1):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = num_levels ** 3
        self.min_value = -1
        self.max_value = 1

        self.learnable = learnable
        if self.learnable:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Tanh())

        self.f_factor = f_factor

    def encode(self, x):
        # x: B, C, H, W
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.f_factor > 1:
            x = F.interpolate(x, scale_factor=1/self.f_factor, mode='bilinear', align_corners=False)
        x = torch.round(((x - self.min_value) / (self.max_value - self.min_value)) * (self.num_levels - 1))
        x = x[:, 0] + x[:, 1] * self.num_levels + x[:, 2] * self.num_levels ** 2

        return x.long().squeeze()

    def decode_code(self, code, to_rgb=False):
        b, h, w = code.size()
        x = torch.zeros(b, 3, h, w).to(code.device)  # Initialize with zeros

        # Decode color channels
        x[:, 0] = torch.fmod(code, self.num_levels)  # Blue channel
        x[:, 1] = torch.fmod(torch.div(code, self.num_levels), self.num_levels)  # Green channel
        x[:, 2] = torch.div(code, self.num_levels ** 2)  # Red channel

        x = ((x / (self.num_levels - 1)) * (self.max_value - self.min_value)) + self.min_value

        if self.f_factor > 1:
            x = F.interpolate(x, scale_factor=self.f_factor, mode='bilinear', align_corners=False)

        if to_rgb:
            x = self.forward(x)

        return torch.clip(x, self.min_value, self.max_value)

    def forward(self, x):
        if self.learnable:
            x = torch.clip(self.conv(x), -1, 1)
        return x


if __name__ == "__main__":
    from pathlib import Path
    import torch
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from PIL import Image

    # -------- Load a sample image shipped with the repository --------
    IMG_PATH = Path(__file__).resolve().parents[2] / "statics" / "its_just_a_frog_cie.png"

    img = Image.open(IMG_PATH).convert("RGB")

    # -------- Preprocess --------
    transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),                 # [0,1]
        T.Normalize([0.5]*3, [0.5]*3) # → [-1,1]
    ])

    img = transform(img)# .unsqueeze(0)  # (1, 3, H, W)

    # img = img.unsqueeze(0)  # (1, 3, H, W)

    ae = AutoEncoder(num_levels=24)
    code = ae.encode(img)
    code = code.unsqueeze(0)
    # print(img.size())
    # print(code.size())
    # exit()
    recon = ae.decode_code(code)

    print("Code shape:", code.shape)
    print("Recon shape:", recon.shape)

    # Visualize
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(((img + 1) / 2).permute(1,2,0))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(((recon[0] + 1) / 2).permute(1,2,0))
    plt.axis("off")

    plt.show()
