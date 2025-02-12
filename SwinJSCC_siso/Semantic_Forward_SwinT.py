#!/usr/bin/env python
# encoding: utf-8
"""Example codes for https://arxiv.org/abs/2310.07987"""

import csv
import os
import copy
import warnings
import imageio
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

from loss.distortion import MS_SSIM

import torch
from torch import nn
# from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as datasets
import torch.optim as optim
from net.network import SwinJSCC
# from data.datasets import get_loader
from utils import AverageMeter, logger_configuration, seed_torch

from SemanticNN_ import img2bin, bin2img, SemanticNN

import LDPC

warnings.filterwarnings("ignore")

epoch_len = 1
batch_size = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
print("device:", device)

seed = None
rng = np.random.RandomState(seed)


# === Set device to MPS if available, otherwise fallback to CPU ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not available, using CPU")

# =============================================================================
#  Argument Parser
# =============================================================================
parser = argparse.ArgumentParser(description="SwinJSCC")
parser.add_argument("--training", action="store_true", help="training or testing")
parser.add_argument(
    "--trainset",
    type=str,
    default="DIV2K",
    choices=["CIFAR10", "DIV2K"],
    help="train dataset name",
)
parser.add_argument(
    "--testset",
    type=str,
    default="ffhq",
    choices=["kodak", "CLIC21", "ffhq"],
    help="specify the testset for HR models",
)
parser.add_argument(
    "--distortion-metric",
    type=str,
    default="MS-SSIM",
    choices=["MSE", "MS-SSIM"],
    help="evaluation metrics",
)
parser.add_argument(
    "--model",
    type=str,
    default="SwinJSCC_w/_SAandRA",
    choices=[
        "SwinJSCC_w/o_SAandRA",
        "SwinJSCC_w/_SA",
        "SwinJSCC_w/_RA",
        "SwinJSCC_w/_SAandRA",
    ],
    help="SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet",
)
parser.add_argument(
    "--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh"]
)
parser.add_argument("--mimo", action="store_true", help="MIMO or not")
parser.add_argument("--C", type=str, default="96", help="bottleneck dimension")
parser.add_argument(
    "--multiple-snr", type=str, default="10", help="random or fixed snr"
)
parser.add_argument(
    "--model_size",
    type=str,
    default="base",
    choices=["small", "base", "large"],
    help="SwinJSCC model size",
)
args = parser.parse_args()



# =============================================================================
#  Configuration
# =============================================================================
class config:
    seed = 42
    pass_channel = True
    CUDA = False  # We're not using CUDA; we're using MPS or CPU
    device = device  # Use the device we selected above
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = "./history/{}".format(filename)
    log = workdir + "/Log_{}.log".format(filename)
    samples = workdir + "/samples"
    models = workdir + "/models"
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    # if args.trainset == "CIFAR10":
    #     save_model_freq = 5
    #     image_dims = (3, 32, 32)
    #     train_data_dir = "/media/D/Dataset/CIFAR10/"
    #     test_data_dir = "/media/D/Dataset/CIFAR10/"
    #     batch_size = 128
    #     downsample = 2
    #     channel_number = int(args.C)
    #     encoder_kwargs = dict(
    #         img_size=(image_dims[1], image_dims[2]),
    #         patch_size=2,
    #         in_chans=3,
    #         embed_dims=[128, 256],
    #         depths=[2, 4],
    #         num_heads=[4, 8],
    #         C=channel_number,
    #         window_size=2,
    #         mlp_ratio=4.0,
    #         qkv_bias=True,
    #         qk_scale=None,
    #         norm_layer=nn.LayerNorm,
    #         patch_norm=True,
    #     )
    #     decoder_kwargs = dict(
    #         img_size=(image_dims[1], image_dims[2]),
    #         embed_dims=[256, 128],
    #         depths=[4, 2],
    #         num_heads=[8, 4],
    #         C=channel_number,
    #         window_size=2,
    #         mlp_ratio=4.0,
    #         qkv_bias=True,
    #         qk_scale=None,
    #         norm_layer=nn.LayerNorm,
    #         patch_norm=True,
    #     )
    # elif args.trainset == "DIV2K":
    if args.trainset == "DIV2K":
        save_model_freq = 100
        image_dims = (3, 256, 256)
        base_path = "Dataset/HR_Image_dataset/"
        if args.testset == "kodak":
            test_data_dir = ["Dataset/kodak/"]
        elif args.testset == "CLIC21":
            test_data_dir = ["Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == "ffhq":
            test_data_dir = ["yangke/SwinJSCC/data/ffhq/"]

        train_data_dir = [
            base_path + "/clic2020/**",
            base_path + "/clic2021/train",
            base_path + "/clic2021/valid",
            base_path + "/clic2022/val",
            base_path + "/DIV2K_train_HR",
            base_path + "/DIV2K_valid_HR",
        ]
        batch_size = 16
        downsample = 4
        if args.model == "SwinJSCC_w/o_SAandRA" or args.model == "SwinJSCC_w/_SA":
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == "small":
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 2, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == "base":
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 6, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == "large":
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 18, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )


# # =============================================================================
# #  Create the MS-SSIM metric and send it to the selected device
# # =============================================================================
# if args.trainset == "CIFAR10":
#     CalcuSSIM = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).to(
#         config.device
#     )
# else:
#     CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)
CalcuSSIM = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).to(device)


def scale_8bit_weight(x):
    # Multiply each element by a weight that cycles through [8,7,...,1]
    n = x.size()[1]  # sequence length
    w = range(8, 0, -1)  # weights: 8, 7, ..., 1
    for i in range(n):
        x[0, i] = x[0, i] * w[i % 8]
    return x


def data_tf(x):
    # Transform function for CIFAR10: resize and normalize image.
    x = x.resize((96, 96), 2)  # resize image to 96x96
    x = np.array(x, dtype="float32") / 255  # scale to [0,1]
    x = (x - 0.5) / 0.5  # normalize to [-1,1]
    x = x.transpose((2, 0, 1))  # change to channel-first format
    x = torch.from_numpy(x)
    return x


def merge_images(sources, targets, k=10):
    # Merge two sets of images side-by-side for visualization.
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h : (i + 1) * h, (j * 2) * h : (j * 2 + 1) * h] = s
        merged[:, i * h : (i + 1) * h, (j * 2 + 1) * h : (j * 2 + 2) * h] = t
    return (
        merged.transpose(1, 2, 0) / 2 + 0.5
    )  # reverse normalization and change channel order


def to_data(x):
    """Converts a PyTorch tensor to a numpy array."""
    # if torch.cuda.is_available():
    #     y = x.cpu()
    # else:
    #     y = x
    y = x.cpu()
    return y.data.numpy()


def calc_exinfo(Lp1, La1, g1, n1, n, k):
    # Calculate extrinsic information for each group by subtracting previously
    # provided side information (La1) from the current LLRs (Lp1).
    X = torch.zeros_like(Lp1)
    for gi in range(g1 - 1):
        si = gi * k  # start index for information bits
        ei = (gi + 1) * k  # end index for information bits
        sni = gi * n  # start index for this group in the codeword
        eni = gi * n + k  # end index corresponding to information bits
        X[sni:eni] = Lp1[sni:eni] - La1[si:ei]
    # Process the last (possibly shorter) group
    sni = (g1 - 1) * n
    eni = (g1 - 1) * (n - k) + n1
    X[sni:eni] = Lp1[sni:eni] - La1[(g1 - 1) * k :]
    return X


def LDPC_enc(G, X1):
    # Encode the bitstream X1 with the LDPC generator matrix G.
    n1 = X1.size()[1]
    n, k = G.shape  # n: code length, k: info bits length
    g1 = int(np.ceil(n1 / k))  # number of groups
    C1 = torch.zeros([n * g1, 1])
    for gi in range(g1):
        X_g = torch.zeros([k, 1])  # prepare group with zero padding if needed
        si = gi * k
        ei = min((gi + 1) * k, n1)
        X_g[0 : ei - si, 0] = X1[0, si:ei]
        # Encode each group using LDPC
        C1[gi * n : (gi + 1) * n] = torch.tensor(LDPC.encode(G, X_g))
    return C1


def LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La, maxiter):
    # Perform one LDPC decoding iteration for each group.
    # Optionally, use extrinsic (side) information La.
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lp = Lp1[si:ei]
        if La is None:
            La1 = None  # no extrinsic info provided
        else:
            # Prepare extrinsic info for the current group.
            ski = gi * k
            if gi < g1 - 1:
                La1 = torch.zeros(1, n)
                eki = (gi + 1) * k
                La1[0, :k] = La[0, ski:eki]
            else:
                # For the last group, assume default positive LLRs for padded bits.
                La1 = torch.ones(1, n)
                La1[0, : n1 - ski] = La[0, ski:n1]
        # Decode the group with or without the extrinsic info.
        Lp1[si:ei] = LDPC.decode_LLR(Lp, **DEC_para1, La=La1, maxiter=maxiter)
    return Lp1


def hard_decision(Lp2, g1, n1, n, k):
    # Convert LLRs to hard bit decisions (0 or 1) for each group.
    X = torch.zeros([1, n1], dtype=int)
    for gi in range(g1 - 1):
        si = gi * k
        ei = (gi + 1) * k
        sni = gi * n
        eni = gi * n + k
        # Decide bit=1 if LLR < 0, else 0.
        X[0, si:ei] = torch.tensor((Lp2[sni:eni] < 0).T)
    X[0, (g1 - 1) * k :] = torch.tensor(
        (Lp2[(g1 - 1) * n : (g1 - 1) * (n - k) + n1] < 0).T
    )
    return X


def LDPC_dec_init(H, Y1, snr1, g1, n):
    # Initialize LDPC decoding by computing initial LLRs based on the noisy received signal.
    Lc1 = torch.zeros_like(Y1)
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lc1[si:ei], DEC_para1 = LDPC.decoder_init(H, Y1[si:ei], snr1)
    return np.array(Lc1), DEC_para1


def save_img(img, path):
    # Save an image to disk.
    imageio.imwrite(path, Image.fromarray(np.uint8(img * 255)))


def E_distance(x, y):
    # Compute the mean squared error between two images.
    x1 = np.array(x.detach().cpu())
    y1 = np.array(y)
    return ((x1 - y1) ** 2).sum() / x1.size

# ----- Swin-T JSCC -----

def load_weights(model_path):
    # Remap storage to config.device (either mps or cpu)
    pretrained = torch.load(model_path, map_location=device)
    net.load_state_dict(pretrained, strict=False)
    del pretrained

def swintjscc(input, SNR, rate = 96, j = 1):
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]

    # for batch_idx, (input, label) in enumerate(test_data):
    # === Send input to the device ===
    # input is the image tensor (w shape of sth like [1, 3, 96, 96]) 
    input = input.to(device) 
    # Recon_image is the output image
    recon_image, CBR, SNR_out, mse, loss_G = net(input, SNR, rate)

    cbrs.update(CBR)
    snrs.update(SNR_out)

    if mse.item() > 0:
        psnr = 10 * (torch.log10(255.0 * 255.0 / mse))
        psnrs.update(psnr.item())
        msssim = (
            1
            - CalcuSSIM(input, recon_image.clamp(0.0, 1.0))
            .mean()
            .item()
        )
        msssims.update(msssim)
    else:
        psnrs.update(100)
        msssims.update(100)

    log_str = " | ".join(
        [
            f"Time {elapsed.val:.3f}",
            f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
            f"SNR {snrs.val:.1f}",
            f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
            f"MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})",
        ]
    )
    logger.info(log_str)
    writer.writerow([snrs.val, cbrs.val, psnrs.val, msssims.val])
    return recon_image # image tensor, e.g. [1, 3, 96, 96]



def sf_relay(x, swt_x, snr1, rho):
    # Main simulation function implementing the semantic relay system.
    # This function encodes an image into two bitstreams, sends them over noisy channels,
    # then iteratively performs joint LDPC decoding with extrinsic (side) information exchange.
    # Note: swt_x represent the torch tensor of the image that went through
    # Swin-T JSCC Encoder and Decoder
    # ----- Preparation -----
    n = 900  # LDPC codeword length
    # snr2 = 20  # Fixed SNR for the semantic channel
    d_v = 2  # Number of parity-check equations per bit
    d_c = 3  # Number of bits per parity-check equation

    imgdir = f"images/snr{snr1}-rho{rho:g}"
    os.makedirs(imgdir, exist_ok=True)

    # Convert the original image to a binary bitstream.
    X1 = img2bin(x)  # Original bitstream
    # Simulate message corruption by flipping bits with probability rho.
    E = rng.binomial(1, rho, X1.shape)
    X2_bits = (X1 + E) % 2  # Corrupted bitstream

    # Recover the corrupted image and pass it through the semantic encoder.
    X2_img = bin2img(X2_bits).reshape([batch_size, 3, 96, 96]).to(device)

    # X2_img has the shape of [1, 3, 96, 96]
    X2 = swintjscc(X2_img)

    # ----- Modification related to Swin-T starting from here -----

    # This X2 is turned into binary already
    # X2 = semantic_coder.enc(X2_img)
    # n2 = X2.size()[1]

    # LDPC PHY channel: prepare for encoding and transmission.
    n1 = X1.size()[1]
    H, G = LDPC.make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
    n, k = G.shape  # n: code length, k: information bits length

    # ----- Simulate 2 ways of transmission from this point-----
    # 2 process are still decrete from each other at this point

    # Divide the bitstreams into groups for LDPC encoding.
    g1 = int(np.ceil(n1 / k))  # Groups for original bitstream
    # g2 = int(np.ceil(n2 / k))  # Groups for semantic encoded bitstream

    # LDPC encoding of both bitstreams.
    C1 = LDPC_enc(G, X1)
    # C2 = LDPC_enc(G, X2)

    # Simulate transmission by adding Gaussian noise.
    Y1 = LDPC.add_gaussian_noise(C1, snr1, seed=seed)
    # Y2 = LDPC.add_gaussian_noise(C2, snr2, seed=seed)

    # Initialize LDPC decoding: get initial LLRs and decoder parameters.
    Lc1, DEC_para1 = LDPC_dec_init(H, Y1, snr1, g1, n)
    # Lc2, DEC_para2 = LDPC_dec_init(H, Y2, snr2, g2, n)

    # Set up copies of the LLRs for joint and independent decoding.
    Lp1 = copy.deepcopy(Lc1)
    # Lp2 = copy.deepcopy(Lc2)
    Lp1s = copy.deepcopy(Lc1)
    # Lp2s = copy.deepcopy(Lc2)

    # Initialize extrinsic information placeholders.
    La1 = None  # Extrinsic info from semantic branch to LDPC decoder.
    # La2 = None  # Extrinsic info from LDPC branch to semantic decoder.

    for i in range(8):  # Joint iterative decoding loop.
        print(
            f"--------------------- LDPC joint dec [{i:d}] -----------------------------"
        )
        # ----- Joint Decoding: Use extrinsic information from the previous iteration -----
        Lp1 = LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La=La1, maxiter=1)
        # Lp2 = LDPC_dec_LLR(Lp2, DEC_para2, g2, n2, n, k, La=La2, maxiter=1)
        # ----- Independent Decoding (without extrinsic info) for comparison -----
        Lp1s = LDPC_dec_LLR(Lp1s, DEC_para1, g1, n1, n, k, La=None, maxiter=1)
        # Lp2s = LDPC_dec_LLR(Lp2s, DEC_para2, g2, n2, n, k, La=None, maxiter=1)

        # Hard decision outputs from the current LLRs.
        X1_hat = hard_decision(
            Lp1, g1, n1, n, k
        )  # Joint decoding result for original bitstream.
        X1s_hat = hard_decision(Lp1s, g1, n1, n, k)  # Independent decoding result.
        j1 = LDPC.BER(X1, X1_hat)
        s1 = LDPC.BER(X1, X1s_hat)
        print(f"BER s: {s1:g}, j: {j1:g}")

        # Process the semantic branch: decode its bits to reconstruct an image.
        # X2 = hard_decision(Lp2, g2, n2, n, k)
        # X2 = semantic_coder.dec(X2)

        # Prepare images for visualization - Turn torch arr to np arr.
        X2_data = to_data(X2.reshape([batch_size, 3, 96, 96]))

        X1_data = to_data(bin2img(X1_hat).reshape([batch_size, 3, 96, 96]))
        X1s_data = to_data(bin2img(X1s_hat).reshape([batch_size, 3, 96, 96]))

        # Save side-by-side images for visual evaluation.
        merged = merge_images(to_data(x), X2_data)
        save_img(merged, f"{imgdir:s}/origin-semantic-{e:d}-{i:d}.png")

        merged = merge_images(X1s_data, X1_data)
        ed1s = E_distance(x, X1s_data)
        ed1 = E_distance(x, X1_data)
        ed2 = E_distance(x, X2_data)
        print(f"EDs: {ed1s:g}, EDj: {ed1:g}, ED2: {ed2:g}")
        save_img(
            merged,
            os.path.join(
                "%s/%d-%d-BER=%.9f-ED1s=%.9f-ED1=%.9f.png"
                % (imgdir, e, i, j1, ed1s, ed1)
            ),
        )

        # --- Until this point, data of 2 ways of transmission are still separated ---

        # ---------------- Extrinsic Information Exchange Process ----------------
        # (1) Compute extrinsic information from the LDPC decoder branch.
        # if La1 is None:
        #     ex_info1 = Lp1
        # else:
        #     ex_info1 = calc_exinfo(torch.tensor(Lp1), La1.t(), g1, n1, n, k)

        # # (2) Convert the semantic decoder's output (reconstructed image) back to a bitstream
        # # and then map the bits to LLR-like values (0 -> 1 and 1 -> -1).
        # # ex_info2 = (img2bin(X2) * -2 + 1)

        # Lp1_max = Lp1.max()
        # # Lp2_max = Lp2.max()

        # # (3) Generate extrinsic information for the LDPC branch:
        # # Process semantic branch output via a fusion function and scale it.
        # La1 = LDPC.fc(ex_info2, rho / (i + 1), LLR_limit=50)
        # La1 = scale_8bit_weight(La1) * (
        #     10 ** ((-5 + i * (1 - rho) * 2 - snr1 / 2 - 3) / 10)
        # )
        # # Explanation: With lower snr1, the system relies more on the semantic branch info.

        # # (4) Generate extrinsic information for the semantic branch:
        # # Process the extrinsic info from the LDPC branch and convert it through the semantic encoder.
        # ex_fc1 = LDPC.fc(ex_info1, rho / (i + 1), LLR_limit=50)
        # ex_fc1 = hard_decision(ex_fc1, g1, n1, n, k)
        # ex_fc1 = bin2img(ex_fc1).reshape([batch_size, 3, 96, 96])
        # La2 = scale_8bit_weight(semantic_coder.enc(ex_fc1) * -2 + 1) * (
        #     10 ** ((rho * (rho * 1000 + 10 * snr1) + 8 * i) / 10)
        # )
        # # Explanation: Higher snr1 and rho yield more extrinsic info from the LDPC branch to help the semantic decoder.

        # La1_max = La1.max()
        # La2_max = La2.max()
        # print(
        #     f"Max Lp1: {Lp1_max:g}, ex_info2: {ex_info2.max():g}, La1: {La1_max:g}, Lp2: {Lp2_max:g},La2: {La2_max:g}"
        # )

        # # Optionally clip/scaling LLRs if they exceed thresholds.
        # if Lp1_max > 200:
        #     Lp1 = Lp1 * (200 / Lp1_max)
        # if Lp2_max > 300:
        #     Lp2 = Lp2 * (300 / Lp2_max)
        # # Update the hard decision for the next iteration.
        # X1_hat = hard_decision(Lp1, g1, n1, n, k)

        # # Log various metrics to a CSV file for later analysis.
        # with open(f"images/snr{snr1:d}-rho{rho:g}.csv", mode="a", newline="") as file:
        #     writer = csv.writer(file)
        #     data = [e, i, s1, j1, ed1s, ed1, ed2, Lp1_max, La1_max, Lp2_max, La2_max]
        #     writer.writerow(data)

    # Return the final reconstructed image from the joint LDPC decoder.
    return bin2img(X1_hat).reshape([batch_size, 3, 96, 96])


if __name__ == "__main__":

    # Configurations for 

    seed_torch()
    logger = logger_configuration(config, save_log=False)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    # model_path = r"SwinJSCC w/SA&RA/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    model_path = os.path.join(".", "SwinJSCC_w", "SA&RA", "SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model")

    load_weights(model_path)
    # === Move network to the selected device ===
    net = net.to(config.device)
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    # train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0

    # -------------------------------------

    # Initialize the semantic coder and load pretrained weights if available.
    semantic_coder = SemanticNN(16, 20, 1, "mps")

    file_path = "semantic_coder.pkl"
    if os.path.exists(file_path):
        semantic_coder.load_state_dict(torch.load(file_path))
    semantic_coder.to(device)

    # Load CIFAR10 dataset (train and test sets).
    train_set = datasets.CIFAR10("./data", train=True, transform=data_tf, download=True)
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_set = datasets.CIFAR10("./data", train=False, transform=data_tf, download=True)
    test_data = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    # Main simulation loop: iterate over epochs and batches.
    for e in range(epoch_len):
        counter = 0
        for im, _ in test_data:
            print("Epoch %d-%d:" % (e, counter))
            im = Variable(im)
            im = im.to(device)

            # Iterate over different corruption probabilities (rho) and SNR values.
            for rho in [0.05, 0.15, 0.35, 0]:
                for snr1 in range(-5, 20, 5):
                    print(
                        f"===================== rho={rho:g}, snr={snr1:d} ===================="
                    )
                    os.makedirs("images/", exist_ok=True)
                    fname = f"images/snr{snr1:d}-rho{rho:g}.csv"
                    if not os.path.exists(fname):
                        with open(fname, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            data = [
                                "epoch",
                                "iter_round",
                                "BERs",
                                "BERj",
                                "EDs",
                                "EDj",
                                "ED_semantic",
                                "Lp1_max",
                                "La1_max",
                                "Lp2_max",
                                "La2_max",
                            ]
                            writer.writerow(data)

                    # Run the simulation function (which includes extrinsic information exchange).
                    sf_relay(copy.deepcopy(im), snr1, rho)

            counter += 1
            if counter >= 32:
                break
