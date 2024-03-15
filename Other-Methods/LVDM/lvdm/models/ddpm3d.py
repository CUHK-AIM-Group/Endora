import os
import time
import random
import itertools
from functools import partial
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_only
from lvdm.models.modules.distributions import normal_kl, DiagonalGaussianDistribution
from lvdm.models.modules.ema import LitEma
from lvdm.models.modules.util import make_beta_schedule, extract_into_tensor, noise_like, random_temporal_masking
from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, check_istarget
from lvdm.utils.saving_utils import log_txt_as_img

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


def split_video_to_clips(video, clip_length, drop_left=True):
    video_length = video.shape[2]
    shape = video.shape
    if video_length % clip_length != 0 and drop_left:
        video = video[:, :, :video_length // clip_length * clip_length, :, :]
        print(f'[split_video_to_clips] Drop frames from {shape} to {video.shape}')
    nclips = video_length // clip_length
    clips = rearrange(video, 'b c (nc cl) h w -> (b nc) c cl h w', cl=clip_length, nc=nclips)
    return clips

def merge_clips_to_videos(clips, bs):
    nclips = clips.shape[0] // bs
    video = rearrange(clips, '(b nc) c t h w -> b c (nc t) h w', nc=nclips)
    return video

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in pixel space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 video_length=None,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 n_classes=101,
                 *args, **kwargs
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.channels = channels
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.conditioning_key = conditioning_key # also register conditioning_key in diffusion
        self.n_classes = n_classes
        
        self.temporal_length = video_length if video_length is not None else unet_config.params.temporal_length
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik) or (ik.startswith('**') and ik.split('**')[-1] in k):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        channels = self.channels
        video_length = self.total_length
        size = (batch_size, channels, video_length, *self.image_size)
        return self.p_sample_loop(size,
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True, mask=None):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        if mask is not None:
            assert(mean is False)
            assert(loss.shape[2:] == mask.shape[2:]) #thw need be the same
            loss = loss * mask
        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3, 4])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
    def get_condition_validate(self, prompt):
        """ text embd
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        c = self.get_learned_conditioning(prompt)
        bs = c.shape[0]
        
        return c
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
        
    def training_epoch_end(self, outputs):
        
        if (self.current_epoch == 0) or self.resume_new_epoch == 0:
            self.epoch_start_time = time.time()
            self.current_epoch_time = 0
            self.total_time = 0
            self.epoch_time_avg = 0
        else:
            self.current_epoch_time = time.time() - self.epoch_start_time
            self.epoch_start_time = time.time()
            self.total_time += self.current_epoch_time
            self.epoch_time_avg = self.total_time / self.current_epoch
        self.resume_new_epoch += 1
        epoch_avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        self.log('train/epoch/loss', epoch_avg_loss, logger=True, on_epoch=True)
        self.log('train/epoch/idx', self.current_epoch, logger=True, on_epoch=True)
        self.log('train/epoch/time', self.current_epoch_time, logger=True, on_epoch=True)
        self.log('train/epoch/time_avg', self.epoch_time_avg, logger=True, on_epoch=True)
        self.log('train/epoch/time_avg_min', self.epoch_time_avg / 60, logger=True, on_epoch=True)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c t h w -> b n c t h w')
        denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
        denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, 
                   plot_diffusion_rows=True, plot_denoise_rows=True, **kwargs):
        """ log images for DDPM """
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        if 'fps' in batch:
            log['fps'] = batch['fps']

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            x_start = x[:n_row]

            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(x_start)
                    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                    diffusion_row.append(x_noisy)

            log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            if plot_denoise_rows:
                log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 shift_factor=0.0,
                 split_clips=True,
                 downfactor_t=None,
                 clip_length=None,
                 latent_frame_strde=None,
                 only_model=False,
                 video_length=None,
                 total_length=None,
                 lora_args={},
                 *args, **kwargs):
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.resume_new_epoch = 0
        self.video_length = video_length
        self.totoal_length = total_length
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.cond_stage_config = cond_stage_config
        self.first_stage_config = first_stage_config
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.restarted_from_ckpt = False
        self.shift_factor = shift_factor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
        self.split_clips = split_clips
        self.downfactor_t = downfactor_t
        self.clip_length = clip_length
        self.latent_frame_strde = latent_frame_strde
        
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
            print(f"std={z.flatten().std()}")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if config is None:
            self.cond_stage_model = None
            return
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        z = self.scale_factor * (z + self.shift_factor)
        return z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_condition(self, batch, x, bs, force_c_encode, k, cond_key):
        is_conditional = self.model.conditioning_key is not None # crossattn
        if is_conditional:
            if cond_key is None:
                cond_key = self.cond_stage_key 
            
            # get condition batch of different condition type
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'txt']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    raise NotImplementedError
            else:
                xc = x
            
            # get learned condition.
            # can directly skip it: c = xc
            if self.cond_stage_config is not None and (not self.cond_stage_trainable or force_c_encode):
                if isinstance(xc, torch.Tensor):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
            else:
                c = xc

            # process c
            if bs is not None:
                c = c[:bs]

        else:
            c = None
            xc = None
            
        return c, xc

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        # get input imgaes
        x = super().get_input(batch, k) # k=first_stage_key

        if bs is not None:
            x = x[:bs]
        
        x = x.to(self.device)
        x_ori = x
        b, _, t, h, w = x.shape
        
        # encode video frames x to z via a 2D encoder
        if self.encoder_type == "2d":
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if self.encoder_type == "2d":
            z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        
        if self.latent_frame_strde:
            z = z[:, :, ::4, :, :]
            assert(z.shape[2] == self.temporal_length), f'z={z.shape}, self.temporal_length={self.temporal_length}'
        
        c, xc = self.get_condition(batch, x, bs, force_c_encode, k, cond_key)
        out = [z, c]
        
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            if isinstance(xc, torch.Tensor) and xc.dim() == 4:
                xc = rearrange(xc, '(b t) c h w -> b c t h w', b=b, t=t)
            out.append(xc)
        
        return out
    
    @torch.no_grad()
    def decode(self, z, **kwargs,):
        z = 1. / self.scale_factor * z - self.shift_factor
        results = self.first_stage_model.decode(z,**kwargs)
        return results
    
    @torch.no_grad()
    def overlapped_decode(self, z, max_z_t=None, overlap_t=2, 
                          predict_cids=False, force_not_quantize=False):
        if max_z_t is None:
            max_z_t = z.shape[2]
        assert(max_z_t > overlap_t)
        max_x_t = max_z_t * 4
        drop_r = overlap_t // 2
        drop_l = overlap_t - drop_r
        drop_r_x = drop_r * 4
        drop_l_x = drop_l * 4
        start = 0
        end = max_z_t
        zs = []
        while start <= z.shape[2]:
            zs.append(z[:, :, start:end, :, :])
            start += (max_z_t-overlap_t)
            end += (max_z_t-overlap_t)
        reses = []
        for i, z_ in enumerate(zs):
            if i == 0:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, :max_x_t-drop_r_x, :, :]
            elif i == len(zs)-1:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, drop_l_x:, :, :]
            else:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, drop_l_x:max_x_t-drop_r_x, :, :]
            reses.append(res)
        results = torch.cat(reses, dim=2)
        return results
    
    @torch.no_grad()
    def decode_first_stage_2DAE(self, z, decode_bs=16, return_cpu=True, **kwargs):
        b, _, t, _, _ = z.shape
        z = rearrange(z, 'b c t h w -> (b t) c h w')
        if decode_bs is None:
            results = self.decode(z, **kwargs)
        else:
            z = torch.split(z, decode_bs, dim=0)
            if return_cpu:
                results = torch.cat([self.decode(z_, **kwargs).cpu() for z_ in z], dim=0)
            else:
                results = torch.cat([self.decode(z_, **kwargs) for z_ in z], dim=0)
        results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t).contiguous()
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, decode_bs=16, return_cpu=True, 
                           bs=None, decode_single_video_allframes=False, max_z_t=None,
                           overlapped_length=0, 
                           **kwargs):
        b, _, t, _, _ = z.shape

        if self.encoder_type == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE(z, decode_bs=decode_bs, return_cpu=return_cpu, **kwargs)
        if decode_single_video_allframes:
            z = torch.split(z, 1, dim=0) # tuple of zs
            cat_dim = 0
        elif max_z_t is not None:
            if self.encoder_type == "3d":
                z = torch.split(z, max_z_t, dim=2) # tuple of zs
                cat_dim = 2
            if self.encoder_type == "2d":
                z = torch.split(z, max_z_t, dim=0) # tuple of zs
                cat_dim = 0
        elif (self.split_clips and self.downfactor_t is not None) or ((self.clip_length is not None) and (self.downfactor_t is not None) and (z.shape[2] > self.clip_length // self.downfactor_t) \
            and self.encoder_type == "3d"):
            split_z_t = self.clip_length // self.downfactor_t
            print(f'split z ({z.shape}) to length={split_z_t} clips')
            z = split_video_to_clips(z, clip_length=split_z_t, drop_left=True) # tensor
            if bs is not None and z.shape[0] > bs:
                print(f'split z ({z.shape}) to bs={bs}')
                z = torch.split(z, bs, dim=0) # tuple of zs
                cat_dim = 0
        
        torch.cuda.empty_cache()
        if isinstance(z, tuple):
            zs = [self.decode(z_, **kwargs).cpu() for z_ in z]
            results = torch.cat(zs, dim=cat_dim)
        elif isinstance(z, torch.Tensor):
            results = self.decode(z, **kwargs)
        else:
            raise ValueError
        
        if self.split_clips and self.downfactor_t is not None:
            results = merge_clips_to_videos(results, bs=b)
        return results
    
    @torch.no_grad()
    def encode_first_stage_2DAE(self, x, encode_bs=16):
        b, _, t, _, _ = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if encode_bs is None:
            results = self.first_stage_model.encode(x)
        else:
            x = torch.split(x, encode_bs, dim=0)
            zs = []
            for x_ in x:
                encoder_posterior = self.first_stage_model.encode(x_)
                z = self.get_first_stage_encoding(encoder_posterior).detach()
                zs.append(z)
            results = torch.cat(zs, dim=0)
        results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            results = self.first_stage_model.encode(x)
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        else:
            results = self.first_stage_model.encode(x)
        return results

    def shared_step(self, batch, **kwargs):
        """ shared step of LDM.
        If learned condition, c is raw condition (e.g. text)
        Encoding condition is performed in below forward function.
        """
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None, skip_qsample=False, x_noisy=None, cond_mask=None, **kwargs,):
        if not skip_qsample:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        else:
            assert(x_noisy is not None)
            assert(noise is not None)
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        if self.logvar.device != self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, 
                        return_x0=False, score_corrector=None, corrector_kwargs=None, 
                        unconditional_guidance_scale=1., unconditional_conditioning=None,
                        uc_type=None,):
        t_in = t
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_out = self.apply_model(x, t_in, c)
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                model_out_uncond, model_out = self.apply_model(x_in, t_in, c_in).chunk(2)
            elif isinstance(c, dict):
                model_out = self.apply_model(x, t, c)
                model_out_uncond = self.apply_model(x, t, unconditional_conditioning)
            else:
                raise NotImplementedError
            if uc_type is None:
                model_out = model_out_uncond + unconditional_guidance_scale * (model_out - model_out_uncond)
            else:
                if uc_type == 'cfg_original':
                    model_out = model_out + unconditional_guidance_scale * (model_out - model_out_uncond)
                elif uc_type == 'cfg_ours':
                    model_out = model_out + unconditional_guidance_scale * (model_out_uncond - model_out)
                else:
                    raise NotImplementedError

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                 unconditional_guidance_scale=1., unconditional_conditioning=None,
                 uc_type=None,):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                                       unconditional_guidance_scale=unconditional_guidance_scale, 
                                       unconditional_conditioning=unconditional_conditioning,
                                       uc_type=uc_type,)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, 
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None,):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning,
                                uc_type=uc_type)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, 
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.total_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, 
                                  mask=mask, x0=x0,)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, shape=None, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            if shape is None:
                shape = (self.channels, self.total_length, *self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps, batch_size,
                                                        shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)
        return samples, intermediates
    
    @torch.no_grad()
    def log_condition(self, log, batch, xc, x, c, cond_stage_key=None):
        """ 
        xc: oringinal condition before enconding. 
        c: condition after encoding.
        """
        if x.dim() == 5:
            txt_img_shape = [x.shape[3], x.shape[4]]
        elif x.dim() == 4:
            txt_img_shape = [x.shape[2], x.shape[3]]
        else:
            raise ValueError
        if self.model.conditioning_key is not None: #concat-time-mask
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif cond_stage_key in ["caption", "txt"]:
                log["conditioning_txt_img"] = log_txt_as_img(txt_img_shape, batch[cond_stage_key], size=x.shape[3]//25)
                log["conditioning_txt"] = batch[cond_stage_key]
            elif cond_stage_key == 'class_label':
                try:
                    xc = log_txt_as_img(txt_img_shape, batch["human_label"], size=x.shape[3]//25)
                except:
                    xc = log_txt_as_img(txt_img_shape, batch["class_name"], size=x.shape[3]//25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)
            if isinstance(c, dict) and 'mask' in c:
                log['mask'] =self.mask_to_rgb(c['mask'])
        return log
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., unconditional_guidance_scale=1.0, 
                   cond_noisy_level=None, 
                   c=None, 
                   **kwargs):
        """ log images for LatentDiffusion """
        use_ddim = ddim_steps is not None
        log = dict()

        # get input
        z, c, x, xrec, xc = self.get_input(batch, 
                                           k=self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N,
                                           cond_key=None,
                                           )
        
        N = min(z.shape[0], N)
        n_row = min(x.shape[0], n_row)

        if unconditional_guidance_scale != 1.0:
            prompts = N * [""]
            uc = self.get_condition_validate(prompts)
        else:
            uc = None

        log["inputs"] = x
        log["reconstruction"] = xrec
        log = self.log_condition(log, batch, xc, x, c, 
                                 cond_stage_key=self.cond_stage_key,
        )
        if cond_noisy_level is not None:
            noise = torch.randn_like(z)
            c['noisy_cond'] = self.q_sample(x_start=z, t=cond_noisy_level, noise=noise)
        
        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        params = list(self.model.parameters())
        
        # append cond model paras
        # ------------------------------------------------------------------
        def add_cond_model(cond_model, params):
            if isinstance(params[0], dict):
                # parameter groups
                params.append({"params": list(cond_model.parameters())})
            else:
                # parameter list: [torch.nn.parameter.Parameter]
                params = params + list(cond_model.parameters())
            return params
        
        if self.cond_stage_trainable:
            # print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = add_cond_model(self.cond_stage_model, params)
        # ------------------------------------------------------------------
        
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        # --------------------------------------------------------------------------------
        opt = torch.optim.AdamW(params, lr=lr)
        
        # lr scheduler
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        
        return opt
    
    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @torch.no_grad()
    def mask_to_rgb(self, x):
        x = x * 255
        x = x.int()
        return x

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        print('Successfully initialize the diffusion model !')
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, s=None, mask=None, **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, **kwargs)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        elif self.conditioning_key == 'resblockcond':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc, **kwargs)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == 'hybrid-time':
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, **kwargs)
        elif self.conditioning_key == 'concat-time-mask':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask, **kwargs)
        elif self.conditioning_key == 'concat-adm-mask':
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask, **kwargs)
        elif self.conditioning_key == 'crossattn-adm':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=s, **kwargs)
        elif self.conditioning_key == 'hybrid-adm-mask':
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask, **kwargs)
        elif self.conditioning_key == 'hybrid-time-adm': # adm means y, e.g., class index
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm, **kwargs)
        else:
            raise NotImplementedError()

        return out

class FrameInterpPredLatentDiffusion(LatentDiffusion):
    def __init__(self, noisy_cond=False, max_noise_level=None, 
                 rand_temporal_mask=True, p_interp=None, p_pred=None, 
                 n_prevs=[1], mergenoisy=True, mask_cond_loss=False, 
                 uncondnoS=False, interp_frame_stride=None, concat_mask_on_input=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noisy_cond = noisy_cond
        self.max_noise_level = max_noise_level
        self.n_prevs = n_prevs
        self.rand_temporal_mask = rand_temporal_mask
        self.p_interp = p_interp
        self.p_pred = p_pred
        if self.rand_temporal_mask:
            assert((self.p_interp is not None) and (self.p_pred is not None))
        self.mergenoisy = mergenoisy # debug purpose
        self.mask_cond_loss = mask_cond_loss
        self.uncondnoS = uncondnoS
        self.interp_frame_stride = interp_frame_stride
        self.concat_mask_on_input = concat_mask_on_input
        if concat_mask_on_input:
            assert(self.conditioning_key in ['concat-adm-mask', 'hybrid-adm-mask', 'concat-mask', 'hybrid-mask'])

    @torch.no_grad()
    def add_frames_condition(self, input, c):
        device = input.device
        input_shape = input.shape

        # get a mask indicating condition positions
        if self.rand_temporal_mask:
            mask = random_temporal_masking(input_shape, self.p_interp, self.p_pred, device, n_prevs=self.n_prevs, interp_frame_stride=self.interp_frame_stride)
        else:
            raise NotImplementedError
        
        is_uncond = (mask.all() == 0)
        
        # get a random timestep for condition
        if self.noisy_cond:
            assert(self.max_noise_level is not None)
            s = torch.randint(0, self.max_noise_level, (input_shape[0],), device=device).long()
            if self.uncondnoS and is_uncond:
                s = None
        else:
            s = None

        # make conditional frames
        if not is_uncond:
            cond_idx = []
            for i in range(mask.shape[2]):
                if mask[:, :, i, :, :].all() == 1:
                    cond_idx.append(i)
            cond_frames = input[:, :, cond_idx, :, :] # for both prediction & interpolation
            if self.noisy_cond:
                noise = torch.randn_like(cond_frames)
                cond_frames = self.q_sample(x_start=cond_frames, t=s, noise=noise)
        else:
            cond_frames = None
        
        cond = {}
        cond['original_cond'] = c
        cond['mask'] = mask
        cond['s'] = s
        cond['noisy_cond'] = cond_frames
        return cond
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None, **kwargs):
        if return_original_cond and return_first_stage_outputs: # for logging images
            z, c, x, xrec, xc = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
            out_list = [z, c, x, xrec, xc]
        else:
            z, c = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
            out_list = [z, c]
        cond = self.add_frames_condition(z, c)
        out_list[1] = cond
        return out_list

    def apply_model(self, x_noisy, t, cond, **kwargs):
        """ serves as a pre forward hook """
        if self.concat_mask_on_input:
            assert(cond['mask'] is not None)
            cond['mask'] = cond['mask'].to(x_noisy.device) # TEMP code. maybe slower
            cond['c_concat'] = [cond['mask']]
        else:
            cond['c_concat'] = None

        if 'original_cond' in cond:
            original_cond = cond['original_cond']
            if isinstance(original_cond, torch.Tensor):
                cond['c_crossattn'] = [original_cond]
            elif isinstance(original_cond, list):
                raise NotImplementedError
                cond['c_crossattn'] = original_cond
            elif isinstance(original_cond, dict):
                # merge two cond dict
                cond = {**cond, **original_cond} # original_cond: 'c_crossattn', 'temporal_context'
            else:
                raise NotImplementedError
            
        if 'noisy_cond' in cond and cond['noisy_cond'] is not None: # mainly used for sampling?
            # put condition on input
            T_cond = cond['noisy_cond'].shape[2]
            if self.p_pred != 0.0 and self.p_interp == 0: # only prediction
                x_noisy[:, :, :T_cond, :, :] = cond['noisy_cond']
            elif self.p_pred == 0.0 and self.p_interp != 0: # only interpolation
                if T_cond == 2:
                    x_noisy[:, :, 0, :, :] = cond['noisy_cond'][:, :, 0, :, :]
                    x_noisy[:, :, -1, :, :] = cond['noisy_cond'][:, :, 1, :, :]
                else:
                    for i in range(T_cond):
                        x_noisy[:, :, i*self.interp_frame_stride, :, :] = cond['noisy_cond'][:, :, i, :, :]

            elif self.p_pred != 0.0 and self.p_interp != 0: # both interpolatin & prediction
                raise NotImplementedError
        return self.model(x_noisy, t, **cond)

    def p_losses(self, x_start, cond, t, noise=None):
        """ different noise level on x & cond """
        mask = cond['mask']
        s = cond['s']
        
        noise_t = default(noise, lambda: torch.randn_like(x_start))
        noise_s = default(noise, lambda: torch.randn_like(x_start))
        if self.noisy_cond and s is not None:
            x_s = self.q_sample(x_start, t=s, noise=noise_s)
        else:
            x_s = x_start
        x_t = self.q_sample(x_start, t=t, noise=noise_t)
        if s is None:
            assert(mask.all() == 0)
        if self.mergenoisy:
            x_noisy = x_t * (1-mask) + x_s * mask
            noise = noise_t * (1-mask) + noise_s * mask
        else:
            x_noisy = x_t
            noise = noise_t
        cond_mask = mask if self.mask_cond_loss else None
        return super().p_losses(x_start, cond, t, noise=noise, skip_qsample=True, x_noisy=x_noisy, 
                                cond_mask=cond_mask)
        