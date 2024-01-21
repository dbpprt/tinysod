import logging
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2

from accelerate import Accelerator
from accelerate.local_sgd import LocalSGD
from src.data import CollateFunction
from src.modeling.mlp import MLP
from src.modeling.tiny_vit_sam import TinyViT
from src.utils.config import instantiate


logger = logging.getLogger(__name__)


def load_model(image_size: int) -> nn.Module:
    encoder = TinyViT(
        img_size=image_size,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    tinysam = torch.load("./.checkpoints/tinysam.pth", map_location="cpu")
    encoder_keys = [key for key in tinysam.keys() if key.startswith("image_encoder")]
    encoder_weights = {key[len("image_encoder.") :]: tinysam[key] for key in encoder_keys}
    encoder.load_state_dict(encoder_weights)

    for p in encoder.parameters():
        p.requires_grad = False

    # return nn.Sequential(encoder, MLP(1024, 128, 1024, 16))
    return nn.Sequential(encoder, MLP(128, 128, 128, 16))


def main(params: Namespace) -> None:
    accelerator = Accelerator()
    # if args.with_tracking:
    #     accelerator = Accelerator(
    #         cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="all", project_dir=args.project_dir
    #     )
    # else:
    #     accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)

    logger.info(f"Loading dataset configuration {params.data_config}")
    data_config = OmegaConf.load(params.data_config)
    train_dataset = ConcatDataset(datasets=[instantiate(dataset) for dataset in data_config.train])
    eval_dataset = ConcatDataset(datasets=[instantiate(dataset) for dataset in data_config.eval])
    logger.info(
        f"Loaded train dataset with {len(train_dataset)} samples and eval dataset with {len(eval_dataset)} samples"
    )

    train_collate_fn = CollateFunction(
        image_size=params.image_size,
        transforms=[
            v2.RandomResizedCrop(size=params.image_size, scale=(0.5, 1.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # v2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        ],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=2, drop_last=True, shuffle=True, collate_fn=train_collate_fn
    )

    eval_collate_fn = CollateFunction(
        image_size=params.image_size,
        transforms=[
            v2.CenterCrop(size=params.image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=2, drop_last=False, shuffle=False, collate_fn=eval_collate_fn
    )

    # TODO: make config
    # lr = 2e-5
    lr = 1

    model = load_model(image_size=params.image_size)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr / 25)

    # Instantiate learning rate scheduler
    lr_scheduler = OneCycleLR(
        optimizer=optimizer, max_lr=lr, epochs=params.num_epochs, steps_per_epoch=len(train_dataloader)
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    for epoch in range(params.num_epochs):
        model.train()

        with LocalSGD(
            accelerator=accelerator,
            model=model,
            local_sgd_steps=params.local_sgd_steps,
            enabled=params.local_sgd_steps is not None,
        ) as local_sgd:
            for step, batch in enumerate(train_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                # batch.to(accelerator.device)
                # New code #
                # We use the new `accumulate` context manager to perform gradient accumulation
                # We also currently do not support TPUs nor advise it as bugs were found on the XLA side when running our tests.
                with accelerator.accumulate(model):
                    x, y = batch
                    y_hat = model(x)
                    loss = F.binary_cross_entropy_with_logits(y_hat, y / 255)
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    optimizer.zero_grad()
                    # LocalSGD-specific line
                    local_sgd.step()

                if step % params.print_freq == 0:
                    accelerator.print(
                        f"[Training] Epoch: {epoch} | Step {step}/{len(train_dataloader)} - Loss: {loss:.2f} - LR: {optimizer.param_groups[0]['lr']:.7f}"
                    )


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--data_config", required=True, help="Dataset configuration file.")

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Size of the input images.",
    )

    parser.add_argument(
        "--local_sgd_steps",
        type=int,
        default=None,
        help="Number of local SGD steps to perform. If None, local SGD is disabled.",
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Number of steps between printing training progress.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevant project information",
    )
