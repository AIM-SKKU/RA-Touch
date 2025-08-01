import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import ra_touch.util.misc as misc
from ra_touch.util.misc import NativeScalerWithGradNormCount as NativeScaler
from ra_touch.util.misc import save_code_snapshot
from ra_touch.llama.llama_adapter import LLaMA_adapter

from ra_touch.data.dataset import FinetuneDataset
from ra_touch.engine import train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama-2-7b', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='ra_touch/llama-2', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
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


    parser.add_argument('--tactile_model', type=str, default='vit_base_patch16_224', 
                        choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224",], 
                        help="Tactile encoder model")

    # distributed training parameters
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Tactile related LoRA 
    parser.add_argument("--active_modality_names", type=str, nargs="+", default=["vision", "tactile"], help="List of active modalities.")
    parser.add_argument("--lora_modality_names", type=str, nargs="+", default=[], help="List of active modalities.")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint.")
    parser.add_argument("--crop_tacvis", action="store_true", default=False, help="enable cropping for tacvis dataset, otherwise uses CLIP augmentation")
    parser.add_argument("--subtract_background", type=str, default=None, 
                        help="Subtract tactile by [mean, median] of the background tactile image", 
                        choices=[None, "mean", "median", "background"])
    parser.add_argument("--augment_rgb", action="store_true", default=False, help="enable augmentation for rgb images")
    parser.add_argument("--augment_tactile", action="store_true", default=False, help="enable augmentation for tactile images")
    parser.add_argument("--random_drop", action="store_true", default=False, help="randomly drop tactile or vision modality")

    # Retreiaval related
    parser.add_argument('--phase', type=str, default='retrieval', help="Phase of the dataset")
    parser.add_argument('--embedding_path', default=None, type=str, help="Path to the retrieval embeddings")
    parser.add_argument('--top_k', default=5, type=int, help="Top k retrieval")
    parser.add_argument('--retrieval_method', type=str, default='txt2img', choices=['img2img', 'tac2img', 'txt2img', 'img2txt','tac2txt', 'txt2txt'], help="Retrieval method") 
    parser.add_argument('--retriever_weight', type=str, default='', help='path to the query generator model')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, args=args, load=True)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(model_without_ddp, args.pretrained_path)

    dataset_train = FinetuneDataset(
        args.data_config, max_words=args.max_words, 
        tokenizer_path=llama_tokenzier_path, 
        crop_tacvis=args.crop_tacvis, subtract_background=args.subtract_background,
        augment_rgb=args.augment_rgb, augment_tactile=args.augment_tactile,
        random_drop=args.random_drop,
    )
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        save_code_snapshot('code_snaps', args.log_dir)
    else:
        log_writer = None


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.log_name is not None: 
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
