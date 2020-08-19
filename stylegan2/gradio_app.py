import argparse
import torch
from torchvision import utils
import numpy as np
import random
import copy

from model import Generator

import gradio

torch.set_grad_enabled(False)


class Sampler:
    def __init__(self, args):
        
        self.device = args.device
        self.seed = random.randint(1,99999)

        source_ckpt = torch.load(args.source_ckpt, map_location=self.device)
        self.g1 = Generator(args.size, 512, 8).to(self.device)
        self.g1.load_state_dict(source_ckpt["g_ema"], strict=False)
        self.g1.eval()

        self.latent = None
        self.trunc = self.g1.mean_latent(4096)
        self.truncation = args.truncation

        target_ckpt = torch.load(args.target_ckpt, map_location=self.device)
        self.g2 = Generator(args.size, 512, 8).to(self.device)
        self.g2.load_state_dict(target_ckpt["g_ema"], strict=False)
        self.g2.eval()
        self.g2_forward_from = len(self.g2.to_rgbs) - target_ckpt['args'].finetune_loc + 1
        
        self.eigvec = torch.load(args.target_factors)["eigvec"].to(self.device)
        self.directions = self.eigvec[:,args.factors].unsqueeze(0)
        self.degrees = torch.zeros(len(args.factors)).to(self.device, dtype=torch.float32)

        self.img = None
        
    def create_sample(self, *inputs):
        
        seed = inputs[0]
        new_degrees = torch.tensor(inputs[1:], dtype=torch.float32).to(self.device)

        if seed != self.seed or ((new_degrees - self.degrees) != 0).any() or self.img is None:
            self.seed = seed
            
            self.degrees = new_degrees
                               
            torch.manual_seed(self.seed)
            self.latent = self.g1.get_latent(torch.randn(1, 512, device=self.device))
            
            source_outs = self.g1(
                    [self.latent],
                    truncation=self.truncation,
                    truncation_latent=self.trunc,
                    input_is_latent=True,
                    get_intermediate_layers=True)
            
            target_img, _ = self.g2(
                    [self.latent + self.directions@self.degrees],
                    truncation=self.truncation,
                    truncation_latent=self.trunc,
                    input_is_latent=True,
                    get_intermediate_layers=False,
                    forward_from=self.g2_forward_from,
                    out_and_skip=source_outs[self.g2_forward_from-1])
            
            out_img = torch.cat([source_outs[-1][-1], target_img], dim=3)
            
            out_img = 255*(0.5*out_img + 0.5)
            self.img = out_img.clamp_(0, 255).squeeze().permute(1, 2, 0).to('cpu', torch.uint8).numpy() 
        return self.img

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--n_factors", type=int, default=10)
    parser.add_argument("--finetune_loc", type=int, default=3)
    parser.add_argument("--source_ckpt", type=str, default='./checkpoint/ffhq.pt')
    parser.add_argument("--target_ckpt", type=str, default='./checkpoint/face2dog.pt')
    parser.add_argument("--target_factors", type=str, default='./checkpoint/face2dog_factor.pt')

    args = parser.parse_args()
    args.factors = list(range(args.n_factors))
    
    sampler = Sampler(args)

    gradio_inputs = [gradio.inputs.Slider(minimum=0, maximum=99999, step=1, default=sampler.seed, label='seed')]
    for i in range(args.n_factors):
        gradio_inputs.append(gradio.inputs.Slider(minimum=-5, maximum=5, step=0.2, default=0, label=str(i+1)))

    gradio.Interface(fn=sampler.create_sample, inputs=gradio_inputs, outputs=gradio.outputs.Image(), live=True, title='FreezeG').launch()