import argparse
import math
import os

import sys
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


import sys
sys.path.insert(0, "cartoon_stylegan2")
import lpips
from model import Generator, Encoder

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    # -----------------------------------
    # Parser
    # -----------------------------------

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--e_ckpt", type=str, default=None, help="path to the encoder checkpoint"
    )
    parser.add_argument(
        "--custom_latent", type=str, default=None, help="path a custom latent to start from"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument("--vgg", type=float, default=1.0, help="weight of the vgg loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--project_name", required=True, type=str, help="name of the result project file"
    )
    parser.add_argument(
        "--file", required=True, help="path to image files to be projected"
    )
    parser.add_argument("--plot_every", type=int, default=0, help="create a plot every X times")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()


    # =============================================

    # -----------------------------------
    # Project Images to Latent spaces
    # -----------------------------------

    if args.file is None:
        exit()

    n_mean_latent = 10000

    # Load Real Images
    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    print(args.file)
    img = transform(Image.open(args.file).resize((256, 256)).convert("RGB"))
    imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    del transform

    # -------------
    # Generator
    # -------------

    g_ema = Generator(args.size, 512, 8).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()


    trunc = g_ema.mean_latent(4096).detach().clone()
    # -------------
    # Encoder
    # -------------

    if args.e_ckpt is not None:
        e_ckpt = torch.load(args.e_ckpt, map_location=device)

        encoder = Encoder(args.size, 512).to(device)
        encoder.load_state_dict(e_ckpt['e'])
        encoder.eval()

    # -------------
    # Latent vector
    # -------------

    assert sum([arg is not None for arg in [args.e_ckpt, args.custom_latent]]) < 2, ""

    if args.custom_latent is not None:
        latent_in = torch.load(args.custom_latent)
    if args.e_ckpt is not None:
        with torch.no_grad():
            latent_init = encoder(imgs)
        latent_in = latent_init.detach().clone()
    else:
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    # -------------
    # Noise
    # -------------

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    for noise in noises:
        noise.requires_grad = True

    # -------------
    # Loss
    # -------------

    # PerceptualLoss
    percept = lpips.PerceptualLoss(
        model="net-lin",
        net="vgg",
        use_gpu=device.startswith("cuda")
    )

    # Optimizer
    if args.e_ckpt is not None:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))

    # Training !

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)

        optimizer.param_groups[0]["lr"] = lr

        # fake image
        if args.e_ckpt is not None:
            img_gen, _ = g_ema([latent_in], input_is_latent=True,
                               truncation=args.truncation, truncation_latent=trunc,
                               randomize_noise=False)
        else:
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        #
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        # latent
        if args.e_ckpt is not None:
            latent_hat = encoder(img_gen)

        # Loss
        p_loss = percept(img_gen, imgs).sum()
        r_loss = torch.mean((img_gen - imgs) ** 2)
        mse_loss = F.mse_loss(img_gen, imgs)

        n_loss = noise_regularize(noises)

        if args.e_ckpt is not None:
            style_loss = F.mse_loss(latent_hat, latent_init)
            del latent_hat
            loss = args.vgg * p_loss + r_loss + style_loss + args.mse * mse_loss
        else:
            style_loss = 0.0
            loss = args.vgg * p_loss + r_loss + args.mse * mse_loss + args.noise_regularize * n_loss


        # update
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        noise_normalize_(noises)
        if args.plot_every > 0 and (i) % args.plot_every == 1:
            filename = f"{args.project_name}_iter_{i}.jpg"
            img_ar = make_image(img_gen)
            pil_img = Image.fromarray(img_ar[0])
            pil_img.save(filename)

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f}; "
                f"reconstruction: {r_loss:.4f}; "
                f"mse_img: {mse_loss.item():.4f}; mse_latent: {style_loss:.4f}; lr: {lr:.4f}; iteration: {i} |"
            )
        )
    os.makedirs('latents', exist_ok=True)
    filename = f"latents/{args.project_name}.pt"
    torch.save(latent_in[0].unsqueeze(0), filename)
