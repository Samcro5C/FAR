import json
import os
from glob import glob

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm

from far.losses.lpips import LPIPS
from far.metrics.metric import VideoMetric
from far.models import build_model
from far.models.patch_discriminator import NLayerDiscriminator, calculate_adaptive_weight
from far.utils.ema_util import EMAModel
from far.utils.registry import TRAINER_REGISTRY
from far.utils.vis_util import log_paired_video


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


@TRAINER_REGISTRY.register()
class DCAEIrrTrainer:

    def __init__(
        self,
        accelerator,
        model_cfg,
        perceptual_weight=1.0,
        irradiance_weight=0.1,
        disc_weight=0,
        disc_start_iter=50001
    ):
        super(DCAEIrrTrainer, self).__init__()

        self.accelerator = accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        if model_cfg['vae'].get('from_config'):
            with open(model_cfg['vae']['from_config'], 'r') as fr:
                config = json.load(fr)
            self.model = build_model(
                model_cfg['vae']['type']).from_config(config)
        elif model_cfg['vae'].get('from_pretrained'):
            self.model = build_model(model_cfg['vae']['type']).from_pretrained(
                model_cfg['vae']['from_pretrained'])
        else:
            raise NotImplementedError
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.ema = None

        self.perceptual_weight = perceptual_weight
        self.irradiance_weight = irradiance_weight
        self.disc_weight = disc_weight
        self.disc_start_iter = disc_start_iter

        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().to(accelerator.device).eval()

        if self.disc_weight > 0:
            self.discriminator = NLayerDiscriminator()
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)

    def set_ema_model(self, ema_decay):
        logger = get_logger('far', log_level='INFO')

        if ema_decay is not None:
            self.ema = EMAModel(self.accelerator.unwrap_model(self.model), decay=ema_decay)
            logger.info(f'enable EMA training with decay {ema_decay}')

    def get_params_to_optimize(self, param_names_to_optimize):
        logger = get_logger('far', log_level='INFO')

        G_params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                G_params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')

        logger.info(
            f'#Trained Generator Parameters: {sum([p.numel() for p in G_params_to_optimize]) / 1e6} M'
        )

        D_params_to_optimize = []
        for name, param in self.discriminator.named_parameters():
            if param.requires_grad:
                D_params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')

        logger.info(
            f'#Trained Discriminator Parameters: {sum([p.numel() for p in D_params_to_optimize]) / 1e6} M'
        )

        return G_params_to_optimize, D_params_to_optimize

    def train_step_pixel_loss(self, batch, iters=-1):
        self.model.train()
        loss_dict = {}

        inputs = batch['video'].to(dtype=self.weight_dtype)
        inputs = inputs * 2 - 1

        # Prepare irradiance target aligned with flattened frames
        irr = batch['irradiance'].to(device=inputs.device, dtype=inputs.dtype)
        # irr expected shape: (B, T) or (B, T, 1) or (B,)
        if irr.dim() == 3 and irr.size(-1) == 1:
            irr = irr.squeeze(-1)  # (B, T)
        if inputs.dim() == 5:  # video
            b, t = inputs.shape[0], inputs.shape[1]
            inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')
            if irr.dim() == 2:          # (B, T)
                irr = rearrange(irr, 'b t -> (b t)')      # (B*T,)
            else:
                raise ValueError(f"Unexpected irradiance shape for video: {irr.shape}")
        else:
            # single image batch; irr should be (B,) or (B,1)
            if irr.dim() == 2 and irr.size(-1) == 1:
                irr = irr.squeeze(-1)
            if irr.dim() != 1:
                raise ValueError(f"Unexpected irradiance shape for images: {irr.shape}")

        # Forward: DCAEWithIrradiance returns (recon, irr_pred)
        reconstructions, irr_pred = self.model(inputs, return_dict=False)

        # reconstruction loss
        rec_loss = F.l1_loss(inputs, reconstructions)
        loss_dict['rec_loss'] = rec_loss

        perceptual_loss = torch.tensor(0.0, device=inputs.device)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions)
            loss_dict['perceptual_loss'] = perceptual_loss

        # irradiance loss (use L1 as a robust default)
        irr_pred = irr_pred.squeeze(-1)          # (B*T,)
        irr_loss = F.l1_loss(irr_pred, irr) * self.irradiance_weight
        loss_dict['irr_loss'] = irr_loss

        total_loss = rec_loss + perceptual_loss + irr_loss
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def train_step_gan_loss(self, batch, iters=-1):
        self.model.train()
        loss_dict = {}

        # train generator
        for p in self.discriminator.parameters():
            p.requires_grad = False

        for p in self.accelerator.unwrap_model(self.model).encoder.parameters():
            p.requires_grad = False

        inputs = batch['video'].to(dtype=self.weight_dtype)
        inputs = inputs * 2 - 1

        irr = batch['irradiance'].to(device=inputs.device, dtype=inputs.dtype)
        if irr.dim() == 3 and irr.size(-1) == 1:
            irr = irr.squeeze(-1)

        if inputs.dim() == 5:
            b, t = inputs.shape[0], inputs.shape[1]
            inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')
            if irr.dim() == 2:
                irr = rearrange(irr, 'b t -> (b t)')
            elif irr.dim() == 1 and irr.numel() == b:
                irr = irr.repeat_interleave(t)
            else:
                raise ValueError(f"Unexpected irradiance shape for video: {irr.shape}")
        else:
            if irr.dim() == 2 and irr.size(-1) == 1:
                irr = irr.squeeze(-1)
            if irr.dim() != 1:
                raise ValueError(f"Unexpected irradiance shape for images: {irr.shape}")

        reconstructions, irr_pred = self.model(inputs, return_dict=False)

        # reconstruction loss
        rec_loss = F.l1_loss(inputs, reconstructions)
        loss_dict['rec_loss'] = rec_loss

        perceptual_loss = torch.tensor(0.0, device=inputs.device)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions)
            loss_dict['perceptual_loss'] = perceptual_loss

        # irradiance loss
        irr_pred = irr_pred.squeeze(-1)
        irr_loss = F.l1_loss(irr_pred, irr) * self.irradiance_weight
        loss_dict['irr_loss'] = irr_loss

        # generator GAN loss
        logits_fake = self.discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake)
        d_weight = self.disc_weight * calculate_adaptive_weight(
            rec_loss + perceptual_loss, g_loss,
            last_layer=self.accelerator.unwrap_model(self.model).get_last_layer()
        )
        g_loss = d_weight * g_loss
        loss_dict['g_loss'] = g_loss

        total_loss_g = rec_loss + perceptual_loss + irr_loss + g_loss
        loss_dict['total_loss_g'] = total_loss_g

        # train discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True

        logits_fake = self.discriminator(reconstructions.detach())
        logits_real = self.discriminator(inputs.detach())

        total_loss_d = hinge_d_loss(logits_real, logits_fake)
        loss_dict['total_loss_d'] = total_loss_d

        loss_dict['total_loss'] = total_loss_g + total_loss_d
        return loss_dict

    def train_step(self, batch, iters=-1):
        if iters < self.disc_start_iter:
            return self.train_step_pixel_loss(batch, iters)
        else:
            return self.train_step_gan_loss(batch, iters)

    @torch.no_grad()
    def sample(self, val_dataloader, opt, wandb_logger=None, global_step=0):
        model = self.accelerator.unwrap_model(self.model)

        if self.ema is not None:
            self.ema.store(model)
            self.ema.copy_to(model)

        model.eval()

        # --- accumulators for dataset-level irradiance metrics ---
        sum_abs_err = torch.tensor(0.0, device=self.accelerator.device)
        sum_sq_err  = torch.tensor(0.0, device=self.accelerator.device)
        sum_bias_err= torch.tensor(0.0, device=self.accelerator.device)
        numel_total = torch.tensor(0.0, device=self.accelerator.device)


        for batch_idx, item in enumerate(tqdm(val_dataloader)):

            gt_video = 2 * item['video'] - 1
            gt_video = rearrange(gt_video, 'b t c h w -> (b t) c h w')

            # Model now returns (recon, irradiance_pred) when return_dict=False
            recon_flat, irr_pred_flat = model(gt_video, return_dict=False)

            # --- video reconstruction logging (unchanged behavior) ---
            recon_video = rearrange(
                recon_flat, '(b t) c h w -> b 1 t c h w', b=item['video'].shape[0]
            )
            recon_video = (recon_video + 1) / 2

            gt_video_vis = rearrange(
                gt_video, '(b t) c h w -> b 1 t c h w', b=item['video'].shape[0]
            )
            gt_video_vis = (gt_video_vis + 1) / 2

            log_paired_video(
                sample=recon_video,
                gt=gt_video_vis,
                context_frames=opt['val']['sample_cfg']['context_length'],
                save_suffix=item['index'],
                save_dir=os.path.join(opt['path']['visualization'], f'iter_{global_step}'),
                wandb_logger=wandb_logger,
                wandb_cfg={
                    'namespace': 'eval_vis',
                    'step': global_step,
                }
            )

            # --- irradiance logging (optional) ---
            if 'irradiance' in item:
                # pred: (B*T, 1) -> (B, T)
                irr_pred = irr_pred_flat.squeeze(-1)
                irr_pred = rearrange(irr_pred, '(b t) -> b t', b=item['video'].shape[0])

                irr_gt = item['irradiance'].to(device=irr_pred.device, dtype=irr_pred.dtype)

                # Normalize GT shape to (B, T)
                if irr_gt.dim() == 3 and irr_gt.size(-1) == 1:
                    irr_gt = irr_gt.squeeze(-1)          # (B, T)
                if irr_gt.dim() == 1:
                    # Per-clip irradiance -> broadcast across T explicitly
                    irr_gt = irr_gt[:, None].expand_as(irr_pred)

                # Hard fail if shapes don't match (prevents silent broadcasting bugs)
                if irr_gt.shape != irr_pred.shape:
                    raise ValueError(f"Irradiance shape mismatch: gt={irr_gt.shape}, pred={irr_pred.shape}")

                diff = irr_pred - irr_gt  # (B, T)

                # Distributed-safe: gather across processes before accumulating
                diff_all = self.accelerator.gather(diff.detach())

                sum_abs_err += diff_all.abs().sum()
                sum_sq_err  += (diff_all ** 2).sum()
                sum_bias_err+= diff_all.sum()
                numel_total += diff_all.numel()

        # --- compute dataset-level metrics once ---
        if numel_total.item() > 0:
            irr_mae  = sum_abs_err / numel_total
            irr_rmse = torch.sqrt(sum_sq_err / numel_total)
            irr_mbe = sum_bias_err / numel_total

            if wandb_logger is not None and self.accelerator.is_main_process:
                wandb_logger.log(
                    {
                        'eval/irr_mae': irr_mae.item(),
                        'eval/irr_rmse': irr_rmse.item(),
                        'eval/irr_mbe': irr_mbe.item(),
                    },
                    step=global_step
                )
        if self.ema is not None:
            self.ema.restore(model)

    def read_video_folder(self, video_dir, num_trajectory):
        video_path_list = sorted(glob(os.path.join(video_dir, '*.mp4')))
        video_list = []
        for video_path in video_path_list:
            try:
                video = EncodedVideo.from_path(video_path, decode_audio=False)
                video = video.get_clip(start_sec=0.0, end_sec=video.duration)['video']
                video_list.append(video)
            except:
                print(f'error when opening {video_path}')

        videos = torch.stack(video_list)
        videos = rearrange(videos, 'b c (n f) h w -> b n f c h w', n=num_trajectory)

        videos = videos / 255.0
        videos_sample, videos_gt = torch.chunk(videos, 2, dim=-1)

        # filter out context frame
        videos_sample = videos_sample
        videos_gt = videos_gt
        return videos_sample, videos_gt

    def eval_performance(self, opt, global_step=0):
        logger = get_logger('far', log_level='INFO')
        sample_dir = os.path.join(opt['path']['visualization'], f'iter_{global_step}')
        logger.info(f'begin evaluate {sample_dir}')

        video_metric = VideoMetric(metric=opt['val']['eval_cfg']['metrics'], device=self.accelerator.device)

        videos_sample, videos_gt = self.read_video_folder(sample_dir, num_trajectory=1)
        logger.info(f'evaluating: sample of shape {videos_sample.shape}, gt of shape {videos_gt.shape}')
        result_dict = video_metric.compute(videos_sample.contiguous(), videos_gt.contiguous(), context_length=opt['val']['sample_cfg']['context_length'])
        return result_dict
