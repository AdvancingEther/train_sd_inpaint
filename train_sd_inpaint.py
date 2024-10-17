import argparse
import logging
import math
import os

import random
import shutil
from pathlib import Path
import pickle
import cv2
from PIL import Image
import itertools

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        required=True,
        help="The path to the training dataset.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        required=False,
        help="The path to the validation dataset.",
    )
    parser.add_argument(
        "--normal_data_path",
        type=str,
        default=None,
        required=False,
        help="The path to the normal dataset for validation.",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Whether to save the models every validation epoch."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-control-inpaint-finetuned",
        help="The path of the output directory where the trained model and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd-inpaint-finetune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="the name of the run if you choose to log with wandb.",
    )
    parser.add_argument(
        "--num_log_samples",
        type=int,
        default=16,
        required=False,
        help="The number of samples used to log the intermediate generated results during validation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        required=False,
        help="The number of inference steps of the finetuned StableDiffusionInpaintPipeline during validation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        required=False,
        help="The guidance scale of the finetuned StableDiffusionInpaintPipeline during validation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Sanity checks
    if args.train_data_path is None or args.val_data_path is None:
        raise ValueError("You must specify a training data directory and a validation data directory.")
    
    return args


class FinetuneDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.init_imgs = data["pixel values"]
        self.prompts = data["class labels"]
        self.masks = data["mask labels"]
        
        self.input_ids = self.tokenizer(
            self.prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
    
    def __len__(self):
        return len(self.init_imgs)
    
    def __getitem__(self, index):
        example = {}
        
        init_img = torch.cat([self.init_imgs[index][None]]*3, dim=0)
        example['init_img'] = init_img
        
        mask = self.masks[index][None, None].to(dtype=torch.float32)
        example["mask"] = mask
        
        masked_img = init_img[None] * (mask < 0.5)
        example["masked_img"] = masked_img
        
        example["input_id"] = self.input_ids[index]
        
        return example


def prepare_log_datasets(args):
    def prepare_log_images(args, data_path):
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        init_imgs = data["pixel values"]
        prompts = data["class labels"]
        masks = data["mask labels"]
        
        random_instance = random.Random()
        random_instance.seed(42 if args.seed is None else args.seed)
        indices = random_instance.sample(range(len(init_imgs)), args.num_log_samples)
        
        init_imgs = [init_imgs[id] for id in indices]
        prompts = [prompts[id] for id in indices]
        masks = [masks[id] for id in indices]
        
        log_dataset = {
            "init_imgs": init_imgs,
            "prompts": prompts,
            "masks": masks,
        }
        
        return log_dataset
    
    train_log_dataset = prepare_log_images(args, args.train_data_path)
    norm_log_dataset = prepare_log_images(args, args.normal_data_path)
    if args.val_data_path != None:
        val_log_dataset = prepare_log_images(args, args.val_data_path)
        return train_log_dataset, val_log_dataset, norm_log_dataset
    else:
        return train_log_dataset, None, norm_log_dataset


def log_validation(
    pipeline,
    train_log_dataset,
    val_log_dataset,
    norm_log_dataset,
    args,
    accelerator,
):
    def prepare_log_results(
        pipeline,
        log_dataset,
        args,
        accelerator,
    ):
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
        formatted_images = []
        for i in range(len(log_dataset["init_imgs"])):
            init_img = (log_dataset["init_imgs"][i] + 1) / 2
            init_img = torch.cat([init_img[None]]*3, dim=0)
            mask = log_dataset["masks"][i]
            mask = mask[None]
            
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=log_dataset["prompts"][i],
                    image=init_img,
                    mask_image=mask,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            
            mask = mask.numpy().squeeze()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            init_img = (init_img * 255).permute(1, 2, 0).byte().numpy()
            init_img = np.ascontiguousarray(init_img)
            cv2.drawContours(init_img, contours, -1, (255,0,0), 1)
            formatted_images.append(
                wandb.Image(init_img, caption=f"{i+1}-initial image")
            )
            
            result = np.array(result, dtype=np.uint8)
            cv2.drawContours(result, contours, -1, (255,0,0), 1)
            formatted_images.append(
                wandb.Image(result, caption=f"{i+1}-inpainted image")
            )
        
        return formatted_images
    
    train_formatted_images = prepare_log_results(pipeline, train_log_dataset, args, accelerator)
    norm_formatted_images = prepare_log_results(pipeline, norm_log_dataset, args, accelerator)
    if val_log_dataset is not None:
        val_formatted_images = prepare_log_results(pipeline, val_log_dataset, args, accelerator)
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "training": train_formatted_images,
                        "validation-Hc": val_formatted_images,
                        "validation-normal": norm_formatted_images,
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    else:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "training": train_formatted_images,
                        "validation-normal": norm_formatted_images,
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")


def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scheduler, tokenizer, models and create wrapper for stable diffusion
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    
    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    params_to_optimize = (
        itertools.chain(
            unet.parameters(), text_encoder.parameters()
        )
        if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Get the training dataset and dataloader
    train_dataset = FinetuneDataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
    )
    
    def collate_fn(examples):
        input_ids = torch.stack([example["input_id"] for example in examples])
        
        init_imgs = torch.stack([example["init_img"] for example in examples])
        init_imgs = init_imgs.to(memory_format=torch.contiguous_format).float()
        
        masks = torch.stack([example["mask"] for example in examples])
        masks = masks.to(memory_format=torch.contiguous_format).float()
        
        masked_imgs = torch.stack([example["masked_img"] for example in examples])
        masked_imgs = masked_imgs.to(memory_format=torch.contiguous_format).float()
        
        batch = {
            "input_ids": input_ids,
            "init_imgs": init_imgs,
            "masks": masks,
            "masked_imgs": masked_imgs,
        }
        
        return batch
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        
        if args.wandb_run_name is not None:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.run.name = args.wandb_run_name
    
    # Get the logging datasets
    train_log_dataset, val_log_dataset, norm_log_dataset = prepare_log_datasets(args)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["init_imgs"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_imgs"].reshape(batch["init_imgs"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            mask, size=(args.resolution // 8, args.resolution // 8)
                        )
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if epoch % args.validation_interval == 0:
                logger.info("Running validation... ")
                
                if args.train_text_encoder:
                    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        unet=unwrap_model(unet),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                else:
                    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unwrap_model(unet),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                
                log_validation(
                    pipeline,
                    train_log_dataset,
                    val_log_dataset,
                    norm_log_dataset,
                    args,
                    accelerator,
                )
                
                if args.save_models:
                    pipeline.save_pretrained(args.output_dir)
                
                del pipeline
                torch.cuda.empty_cache()
    
    accelerator.end_training()


if __name__ == "__main__":
    main()