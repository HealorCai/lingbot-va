import argparse
import os
import re
import glob
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from diffusers.utils import export_to_video

from utils import (
    init_logger,
    logger,
)

from wan_va_server import VA_Server

def extract_idx(path):
    filename = os.path.basename(path)
    match = re.search(r"latents_(\d+)\.pt", filename)
    if match is None:
        raise ValueError(f"Invalid latent filename: {filename}")
    return int(match.group(1))

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    logger.info(f"****************************** Generate video ******************************")
    # model.generate()
    exp_name = args.exp_name
    latent_files = sorted(glob.glob(f"{args.save_root}/real/{exp_name}/*.pt"), key=extract_idx)
    latents = []
    for fp in latent_files:
        x = torch.load(fp, map_location="cpu")
        if not torch.is_tensor(x):
            raise TypeError(f"{fp} does not contain a tensor")
        latents.append(x)
    if len(latents) == 0:
        raise FileNotFoundError(f"No latent files found for {exp_name}")
    pred_latent = torch.cat(latents, dim=2)
    pred_latent = pred_latent.to(model.device if hasattr(model, "device") else next(model.parameters()).device)
    decoded_video = model.decode_visual_pred(pred_latent)
    os.makedirs(f"{args.save_root}/visual_pred", exist_ok=True)
    export_to_video(decoded_video, os.path.join(args.save_root, "visual_pred", f"{exp_name}.mp4"), fps=15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )

    args = parser.parse_args()
    args.exp_name='Pick the mug with thick sturdy handle, turn it, put it in the center, and transfer it to the metallic rack with smooth finish._20260228_102849'

    init_logger()
    run(args)

