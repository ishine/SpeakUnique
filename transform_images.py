import torch
import os
from glob import glob
from torchvision.utils import make_grid
import PIL.Image as pilimg
from tqdm import tqdm
import sys
sys.path.insert(0, "cartoon_stylegan2")

from model import Generator
import argparse

parser = argparse.ArgumentParser(
    description="Put reference images into different"
)
parser.add_argument("--out_dir", type=str, required=True, help="directory where the images are stored")
parser.add_argument("--latents_dir", type=str, required=True, help="directory where the precomputed latents are stored")
parser.add_argument("--models", nargs="+", help="models to be projected to", required=True)
parser.add_argument('--create_overview', action='store_true')
parser.add_argument('--device', default='cuda')
parser.add_argument("--truncation", type=float, default=0.7, help="truncation")
args = parser.parse_args()

truncation = 0.7

# directory to save image
os.makedirs(args.out_dir, exist_ok=True)

print(args.models)

def imgs_to_file(imgs, filepath, nrow=1):
    grid = make_grid(
        imgs,
        nrow=nrow,
        normalize=True,
        range=(-1, 1),
    )

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = pilimg.fromarray(ndarr)
    im.save(filepath)


def load_model(model):
    network = torch.load(f'cartoon_stylegan2/networks/{model}.pt')
    generator = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    generator.load_state_dict(network["g_ema"], strict=False)
    trunc = generator.mean_latent(4096)
    return generator, trunc


swap = True
swap_layer_num = 2  # min:1, max:6, step:1

ref_model = 'ffhq256'
# reference_generator, ref_trunc = load_model(ref_model)
all_models = [ref_model] + args.models
latent_paths = sorted(glob(args.latents_dir + '/speaker*.pt'))

ref_generator, ref_trunc = load_model(ref_model)

with torch.no_grad():
    if args.create_overview:
        imgs = []
    # Generate samples for latent
    for idx, model in tqdm(enumerate(all_models)):
        generator, trunc = load_model(model)
        for latent_path in latent_paths:
            latent = torch.load(latent_path)
            _, save_swap_layer = ref_generator(
                [latent],
                input_is_latent=True,
                truncation=args.truncation,
                truncation_latent=ref_trunc,
                swap=swap,
                swap_layer_num=swap_layer_num,
                randomize_noise=False
            )
            img, _ = generator(
                [latent],
                input_is_latent=True,
                truncation=args.truncation,
                truncation_latent=trunc,
                swap=swap,
                swap_layer_num=swap_layer_num,
                swap_layer_tensor=save_swap_layer
            )
            if args.create_overview:
                imgs.append(img)
            else:
                dirname = f'{args.out_dir}/{model}'
                os.makedirs(dirname, exist_ok=True)
                filename = os.path.basename(latent_path).replace('.pt', '.jpg')
                filepath = f'{dirname}/{filename}'
                imgs_to_file(img, filepath)

    if args.create_overview:
        imgs_to_file(torch.cat(imgs, 0), f'{args.out_dir}/overview.png', nrow=len(latent_paths))
