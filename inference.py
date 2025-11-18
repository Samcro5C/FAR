# test_infer_to_netcdf.py
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
import gc
from accelerate import Accelerator
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
    accelerator = Accelerator(mixed_precision=opt.get("mixed_precision"))
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
        global_step = resume_checkpoint(args, accelerator, os.path.join(opt['path']['pretrain_network'], 'models'), train_pipeline)
    else:
        logger.info('infer only pretrained model without load checkpoint!')
        global_step = 0

    for date in dates:
        from datetime import datetime
        # if date <= datetime(2019, 10, 20).date():
        #     continue
        date_mask = sample_timestamps.date == date
        idx_date = date_mask.nonzero()[0]
        ts_date = sample_timestamps[date_mask]
        ds_date = torch.utils.data.Subset(sample_dataset, idx_date)
        dl_date = dataloader = torch.utils.data.DataLoader(
            ds_date,
            batch_size=sampleset_cfg["batch_size_per_gpu"],
            shuffle=False,
            persistent_workers=False,
            num_workers=sampleset_cfg.get("num_worker_per_gpu", 4),
            pin_memory=False,
            timeout=120,
        )

        # Prepare with accelerator
        sample_dataloader = accelerator.prepare(dl_date)

        preds_list, preds_latents, indices = [], [], []
        def collector(pred_b, latent_b, idx_b):
            preds_latents.append(latent_b)
            preds_list.append(pred_b)                 # pred_b is CPU, detached
            indices.append(torch.as_tensor(idx_b))    # keep CPU

        # Test!
        logger.info('***** Running testing *****')

        logger.info(f'begin evaluation step-{global_step}:')

        with torch.inference_mode():
            train_pipeline.sample(sample_dataloader, opt, wandb_logger=None, global_step=global_step, on_batch=collector)

        accelerator.wait_for_everyone()

        preds   = torch.cat(preds_list, dim=0)   # (N_total, Ntraj, F, C, H, W)
        latents = torch.cat(preds_latents, dim=0)
        indices = torch.cat(indices, dim=0) # (N_total,)

        order = torch.argsort(indices)
        preds   = preds[order]
        latents = latents[order]
        indices = indices[order]

        # Build xarray DataArray
        da = xr.DataArray(
            preds.to(torch.float32).numpy(),
            dims=("time", "sample", "lead_time", "c", "x", "y"),
            coords={
                "time": ts_date.values,
                "lead_time": sample_lead_times,
            },
            name="asi_preds",
            attrs={
                "description": "FAR model predictions",
                "dtype": "float32",
            },
        )

        da_latents = xr.DataArray(
            latents.to(torch.float32).numpy(),
            dims=("time", "sample", "lead_time", "c", "x", "y"),
            coords={
                "time": ts_date.values,
                "lead_time": sample_lead_times,
            },
            name="asi_latents",
            attrs={
                "description": "FAR model predictions",
                "dtype": "float32",
            },
        )

        out_path = Path(opt.get('eval_dir', './eval_dir')) / date.strftime("%Y%m%d.nc")
        out_path_latents = Path(opt.get('eval_dir', './eval_dir')) / date.strftime("latents_%Y%m%d.nc")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        da.to_netcdf(out_path, engine="netcdf4")
        da_latents.to_netcdf(out_path_latents, engine="netcdf4")

        accelerator.print(f"Saved predictions to: {out_path.resolve()}")

        del sample_dataloader, dl_date, ds_date, da, preds, preds_list, indices
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
