import argparse

import torch
from torchvision import utils
from model import Generator

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="truncation ratio"
    )
    parser.add_argument(
        "--ckpt1",
        type=str,
        default="550000.pt",
        help="path to the original model checkpoint",
    )
    parser.add_argument(
        "--ckpt2",
        type=str,
        default="face2met_10k.pt",
        help="path to the finetuned model checkpoint",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )    
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema1 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint1 = torch.load(args.ckpt1, map_location="cpu")

    g_ema1.load_state_dict(checkpoint1["g_ema"], strict=False)

    g_ema2 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint2 = torch.load(args.ckpt2, map_location="cpu")

    g_ema2.load_state_dict(checkpoint2["g_ema"], strict=False)   
    
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema1.mean_latent(4096)
    else:
        mean_latent = None

    with torch.no_grad():
        g_ema1.eval()
        g_ema2.eval()

        latents = g_ema1.get_latent(torch.randn(args.sample, args.latent, device=device))
        
        sample1, _ = g_ema1(
            [latents], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
        )

        sample2, _ = g_ema2(
            [latents], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
        )

        utils.save_image(
            torch.cat([sample1, sample2]),
            f"sample.png",
            nrow=args.sample,
            normalize=True,
            range=(-1, 1),
        )        

