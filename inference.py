# test_infer_to_netcdf.py
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
import gc
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
import xarray as xr
import numpy as np

from far.data import build_dataset
from far.trainers import build_trainer
from far.utils.logger_util import dict2str, set_path_logger

OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x))


def resume_checkpoint(accelerator: Accelerator, ckpt_root: str, train_pipeline) -> int:
    """
    Loads the latest accelerator state + EMA from FAR-style checkpoints.
    Returns the global_step inferred from the checkpoint dir name.
    """
    if not ckpt_root or not os.path.isdir(ckpt_root):
        accelerator.print("No checkpoint directory found; running with current weights.")
        return 0

    dirs = [d for d in os.listdir(ckpt_root) if d.startswith("checkpoint")]
    if not dirs:
        accelerator.print("No checkpoints found; running with current weights.")
        return 0
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1]

    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(os.path.join(ckpt_root, path))
    global_step = int(path.split("-")[1])

    # EMA (if used)
    if getattr(train_pipeline, "ema", None) is not None:
        accelerator.print(f"Resuming EMA from checkpoint {path}")
        ema_state = torch.load(
            os.path.join(ckpt_root, path, "ema.pth"),
            map_location="cpu",
            weights_only=True,
        )
        train_pipeline.ema.load_state_dict(ema_state)

    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to FAR config (YAML).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output NetCDF path. Default: <log_dir>/predictions_step-<global_step>.nc",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="(Optional) If your trainer supports swapping to EMA weights for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (sets per-rank seeds with offset).",
    )
    args = parser.parse_args()

    # Load config fully resolved
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=opt.get("mixed_precision"),
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            even_batches=False
            )
    )
    logger = get_logger("far", log_level="INFO")

    # Prepare experiment directories/loggers (no wandb)
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, opt, is_train=False)

    logger.info(accelerator.state)
    logger.info(dict2str(opt))

    # Seed
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Build trainer/model
    train_pipeline = build_trainer(opt["train"]["train_pipeline"])(**opt["models"], accelerator=accelerator)

    # Dataset / loader for sampling
    if opt["datasets"].get("sample"):
        sampleset_cfg = opt["datasets"]["sample"]
        sample_dataset = build_dataset(sampleset_cfg)
        sample_timestamps = sample_dataset.base.timestamps
        sample_lead_times = sample_dataset.base.output_timedeltas
    else:
        raise ValueError("Config has no datasets.sample section; nothing to run inference on.")

    dates = np.unique(sample_timestamps.date)

    # Prepare with accelerator
    train_pipeline.model = accelerator.prepare(train_pipeline.model)
    train_pipeline.model.eval()
    # set ema after prepare everything: sync the model init weight in ema
    train_pipeline.set_ema_model(ema_decay=opt['train'].get('ema_decay'))
    if opt['path']['pretrain_network']:
        global_step = resume_checkpoint(accelerator, os.path.join(opt['path']['pretrain_network'], 'models'), train_pipeline)
    else:
        logger.info('infer only pretrained model without load checkpoint!')
        global_step = 0

    dl = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=sampleset_cfg["batch_size_per_gpu"],
        shuffle=False,
        persistent_workers=False,
        num_workers=sampleset_cfg.get("num_worker_per_gpu", 4),
        pin_memory=False,
        timeout=120,
    )

    # Prepare with accelerator
    sample_dataloader = accelerator.prepare(dl)

    rank = accelerator.process_index
    out_root = Path(opt.get("eval_dir", "./eval_dir"))
    out_latents = out_root / 'latents'
    out_latents.mkdir(parents=True, exist_ok=True)
    zarr_path = out_root / f"tmp_asi_preds{rank}.zarr"
    zarr_path_latents = out_latents / f"tmp_latent_preds{rank}.zarr"

    time_encoding = {
        "units": "seconds since 1970-01-01T00:00:00",
        "dtype": "int64",
        "calendar": "standard",
    }
    encoding = {"time": time_encoding}

    def collector(pred_b, latent_b, idx_b):
        pred_b_np = (pred_b * 255).to(torch.uint8).numpy()
        latent_b_np = latent_b.to(torch.float32).numpy()
        idx_b_np = idx_b.numpy().astype('datetime64[ns]')

        da = xr.DataArray(
            pred_b_np,
            dims=("time", "sample", "lead_time", "c", "x", "y"),
            coords={
                "time": idx_b_np,
                "lead_time": sample_lead_times,
            },
            name="asi_preds",
            attrs={
                "description": "FAR model predictions",
                "dtype": "float32",
            },
        )
        da_latents = xr.DataArray(
            latent_b_np,
            dims=("time", "sample", "lead_time", "c", "x", "y"),
            coords={
                "time": idx_b_np,
                "lead_time": sample_lead_times,
            },
            name="asi_latents",
            attrs={
                "description": "FAR model predictions",
                "dtype": "float32",
            },
        )
        
        data_vars = {'asi_preds': da}

        if hasattr(train_pipeline.vae, 'irradiance_head'):
            N, S, T, C, H, W = latent_b.shape
            pred_latents_flat = latent_b.reshape(N*S*T, C, H, W)
            pred_latents_flat_scaled = 1 / train_pipeline.vae.config.scaling_factor * pred_latents_flat
            pred_irr = train_pipeline.vae.irradiance_head(pred_latents_flat_scaled.to(train_pipeline.vae.device))
            pred_irr = pred_irr.reshape(N, S, T)
            pred_irr_np = pred_irr.to(torch.float32).cpu().numpy()
            da_irr = xr.DataArray(
                pred_irr_np,
                dims=("time", "sample", "lead_time"),
                coords={
                    "time": idx_b_np,
                    "lead_time": sample_lead_times,
                },
                name="irr_preds",
                attrs={
                    "description": "FARSKY model irradiance predictions",
                },
            )
            data_vars['irr_preds'] = da_irr

        chunk_sizes = {
            "time": 2,
            "sample": -1,
            "lead_time": -1,
            "c": -1,
            "x": -1,
            "y": -1,
        }
        ds = xr.Dataset(data_vars=data_vars)
        ds = ds.chunk(chunk_sizes)
        da_latents = da_latents.chunk(chunk_sizes)
        if zarr_path.exists():
            ds.to_zarr(zarr_path, mode="a", append_dim="time")
        else:
            ds.to_zarr(zarr_path, mode="w", encoding=encoding)
        if zarr_path_latents.exists():
            da_latents.to_zarr(zarr_path_latents, mode="a", append_dim="time")
        else:
            da_latents.to_zarr(zarr_path_latents, mode="w", encoding=encoding)


    # Test!
    logger.info('***** Running testing *****')
    logger.info(f'begin evaluation step-{global_step}:')

    with torch.inference_mode():
        train_pipeline.sample(sample_dataloader, opt, wandb_logger=None, global_step=global_step, on_batch=collector)
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
