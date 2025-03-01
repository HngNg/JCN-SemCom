import os
import csv
import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True

import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision

parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='ffhq',
                    choices=['kodak', 'CLIC21', 'ffhq'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                    help='SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='96',
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='13',
                    help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base',
                    choices=['small', 'base', 'large'], help='SwinJSCC model size')
args = parser.parse_args()

class config():
    seed = 42
    pass_channel = True
    CUDA = False
    device = torch.device("mps")
    norm = False
    # logger
    print_step = 1
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = os.path.join(workdir, 'Log_{}.log'.format(filename))
    samples = os.path.join(workdir, 'samples')
    models = os.path.join(workdir, 'models')
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 100

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "./Dataset/CIFAR10/"
        test_data_dir = "./Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 20
        image_dims = (3, 256, 256)
        base_path = "Dataset/HR_Image_dataset/"
        if args.testset == 'kodak':
            test_data_dir = ["/media/D/Dataset/kodak/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = ["/media/D/yangke/SwinJSCC/data/ffhq/"]

        train_data_dir = [os.path.join(base_path, 'clic2020/**'),
                          os.path.join(base_path, 'clic2021/train'),
                          os.path.join(base_path, 'clic2021/valid'),
                          os.path.join(base_path, 'clic2022/val'),
                          os.path.join(base_path, 'DIV2K_train_HR'),
                          os.path.join(base_path, 'DIV2K_valid_HR')]
        batch_size = 16
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size =='large':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )

if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to("mps")
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to("mps")

def load_weights(model_path):
    pretrained = torch.load(model_path, map_location=torch.device('mps'))
    net.load_state_dict(pretrained, strict=True)
    del pretrained

# Note: We now pass the epoch and csv_writer to train_one_epoch.
def train_one_epoch(args, epoch, csv_writer, csv_file):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    # (Here we show the CIFAR10 branch; similar changes would apply to the other branch.)
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to("mps")
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / torch.log(torch.tensor(10.)))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % len(train_loader)) / len(train_loader) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % len(train_loader)}/{len(train_loader)}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                # Build a dictionary of values for the CSV row:
                csv_row = {
                    "epoch": epoch,
                    "step": global_step % len(train_loader),
                    "process": process,
                    "time": elapsed.val,
                    "loss": losses.val,
                    "loss_avg": losses.avg,
                    "CBR": cbrs.val,
                    "CBR_avg": cbrs.avg,
                    "SNR": snrs.val,
                    "SNR_avg": snrs.avg,
                    "PSNR": psnrs.val,
                    "PSNR_avg": psnrs.avg,
                    "MSSSIM": msssims.val,
                    "MSSSIM_avg": msssims.avg,
                    "Lr": cur_lr,
                }
                csv_writer.writerow(csv_row)
                csv_file.flush()
                for meter in metrics:
                    meter.clear()
    else:
        # Similar logging is applied for the DIV2K branch.
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to("mps")
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / torch.log(torch.tensor(10.)))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % len(train_loader)) / len(train_loader) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % len(train_loader)}/{len(train_loader)}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                csv_row = {
                    "epoch": epoch,
                    "step": global_step % len(train_loader),
                    "process": process,
                    "time": elapsed.val,
                    "loss": losses.val,
                    "loss_avg": losses.avg,
                    "CBR": cbrs.val,
                    "CBR_avg": cbrs.avg,
                    "SNR": snrs.val,
                    "SNR_avg": snrs.avg,
                    "PSNR": psnrs.val,
                    "PSNR_avg": psnrs.avg,
                    "MSSSIM": msssims.val,
                    "MSSSIM_avg": msssims.avg,
                    "Lr": cur_lr,
                }
                csv_writer.writerow(csv_row)
                csv_file.flush()
                for meter in metrics:
                    meter.clear()
    for meter in metrics:
        meter.clear()

if __name__ == '__main__':
    # Create working directories if they do not exist.
    os.makedirs(config.workdir, exist_ok=True)
    os.makedirs(config.samples, exist_ok=True)
    os.makedirs(config.models, exist_ok=True)

    # Configure logger (assuming logger_configuration sets up a FileHandler for config.log)
    logger = logger_configuration(config, save_log=False)
    logger.info(config.__dict__)
    
    # Create CSV log file with name based on current date and time.
    csv_log_filename = os.path.join(config.workdir, 'Log_{}.csv'.format(config.filename))
    csv_file = open(csv_log_filename, mode='w', newline='')
    fieldnames = ["epoch", "step", "process", "time", "loss", "loss_avg", "CBR", "CBR_avg", "SNR", "SNR_avg",
                  "PSNR", "PSNR_avg", "MSSSIM", "MSSSIM_avg", "Lr"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    seed_torch()
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    model_path = "SwinJSCC_w/SA&RA/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    # load_weights(model_path)
    net = net.to("mps")
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // len(train_loader)
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args, epoch, csv_writer, csv_file)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=os.path.join(config.models, '{}_EP{}.model'.format(config.filename, epoch + 1)))
                # For demonstration, break out after saving once.
                break
    else:
        # In test mode, you might implement CSV logging similarly.
        pass

    csv_file.close()
