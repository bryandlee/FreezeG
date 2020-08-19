import argparse

import torch
from torchvision import utils
import torch.nn.functional as F

from model import Generator


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-d", "--degree", type=float, default=5)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("-n", "--n_sample", type=int, default=7)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_prefix", type=str, default="factor")
    parser.add_argument("factor", type=str)

    args = parser.parse_args()

    eigvec = torch.load(args.factor, map_location='cpu')["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    g = Generator(args.size, 512, 8).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    scale = 1
    for img_, img1_, img2_ in zip(img, img1, img2):
        
        imgs = torch.cat([img1_, img_, img2_], 0)
        imgs = F.interpolate(imgs, (256,256))

        min_, max_ = imgs.min(), imgs.max()
        imgs = ((imgs - (min_ + max_)/2)/(max_ - min_)) * 2
        
        
        grid = utils.save_image(
            imgs,
            f"sample/factors/{args.out_prefix}_index-{args.index}_degree-{args.degree}-{scale}.png",
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )
        scale += 1
