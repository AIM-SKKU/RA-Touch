import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import ra_touch.util.misc as misc

from torch.utils.data import DataLoader
from tvl_enc import tvl
from tvl_enc.tvl import ModalityType
from ra_touch.tg_retriever import TGRetriever
from ra_touch.data.retriever_dataset import Dataset
from ra_touch.util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Tactile-Guided Retriever', add_help=False)
    parser.add_argument('--checkpoint_path', default='/path/to/pretrained', type=str, help='path to checkpoint from pretrain stage')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')
    parser.add_argument('--gpu', type=lambda s: [int(x) for x in s.split(',')], help='GPU ids to use for distributed training')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/finetune/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--log_name', default=None, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--tactile_model', type=str, default='resnet18', choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "resnet18"], 
                        help="Tactile encoder model")

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Tactile related 
    parser.add_argument("--crop_tacvis", action="store_true", default=False, help="enable cropping for tacvis dataset, otherwise uses CLIP augmentation")
    parser.add_argument("--subtract_background", type=str, default=None, 
                        help="Subtract tactile by [mean, median] of the background tactile image", 
                        choices=[None, "mean", "median", "background"])
    parser.add_argument("--augment_rgb", action="store_true", default=False, help="enable augmentation for rgb images")
    parser.add_argument("--augment_tactile", action="store_true", default=False, help="enable augmentation for tactile images")
    parser.add_argument("--random_drop", action="store_true", default=False, help="randomly drop tactile or vision modality")

    return parser


def train(args):
    misc.init_distributed_mode(args)
    
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.determinstic = True
    
    retriever = TGRetriever(
        feature_dim=768,
        output_dim=768,
        device=device,
    ).to(device)
    
    modality_types = [
        getattr(ModalityType, modality_name.upper()) 
        for modality_name in ["vision", "text", "tactile"]
    ]
    tvl_encoder = tvl.TVL(tactile_model=args.tactile_model, active_modalities=modality_types)
    state_dict = torch.load(args.checkpoint_path, map_location='cpu')['model']
    miss_keys, unexpected_keys = tvl_encoder.load_state_dict(state_dict, strict=False)
    print(f"Missing tvl_encoder keys: {miss_keys}, unexpected tvl_encoder keys: {unexpected_keys}")

    if args.distributed:
        retriever = torch.nn.parallel.DistributedDataParallel(retriever, device_ids=[args.gpu],
                                                              find_unused_parameters=True)
        local_rank = int(os.environ["LOCAL_RANK"])
        tvl_encoder = tvl_encoder.to(f"cuda:{local_rank}")
        tvl_encoder.eval()
        model_without_ddp = retriever.module

    # Training detail
    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    dataset_train = Dataset(
        args.data_config, max_words=args.max_words,
        crop_tacvis=args.crop_tacvis, subtract_background=args.subtract_background,
        augment_rgb=args.augment_rgb, augment_tactile=args.augment_tactile,
        random_drop=args.random_drop,
    )
    sampler_train = torch.utils.data.DistributedSampler(dataset_train) if args.distributed else None
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    # Tensorboard 로그를 위한 SummaryWriter 초기화 (log_dir이 지정된 경우)
    log_writer = None
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir, comment=args.log_name if args.log_name is not None else "")
    
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        retriever.train()
        metric_logger = misc.MetricLogger(delimiter="  ")
        optimizer.zero_grad()
        
        for data_iter_step, dataset_item in enumerate(metric_logger.log_every(data_loader_train, 10, f'Epoch: [{epoch}]')):
            for k, v in dataset_item.items():
                if isinstance(v, list):
                    v = v[0]
                if k != 'labels':
                    dataset_item[k] = v.to(device, non_blocking=True).squeeze()
            
            with torch.no_grad() as no_grad:
                feats = tvl_encoder(dataset_item)
            
            query_emb, losses = retriever(vis_feat=feats[ModalityType.VISION],
                                          tac_feat=feats[ModalityType.TACTILE],
                                          gt_feat=feats[ModalityType.TEXT])
            tot_loss = sum([v for k, v in losses.items() if k.endswith("_loss")])
            loss_value = torch.tensor(tot_loss.item())
            if not torch.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            loss_scaler(tot_loss, optimizer, parameters=retriever.parameters(), update_grad=True)
            optimizer.zero_grad()
            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss_value)
            for k, v in losses.items():
                metric_logger.update(**{f"{k}": v.item()})
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # logging on Tensorboard
            if log_writer is not None:
                global_step = epoch * len(data_loader_train) + data_iter_step
                log_writer.add_scalar('train_loss', loss_value.item(), global_step)
                log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=retriever, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()}, 'epoch': epoch}
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    # 학습 종료 후 로그 기록 종료
    if log_writer is not None:
        log_writer.close()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.log_name is not None: 
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(args)
