import csv
import os
import warnings
import imageio
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

from loss.distortion import MS_SSIM

import torch
from torch import nn
from torchvision import datasets, transforms

from utils import logger_configuration, seed_torch

from Semantic_SwinT_cifar10 import SwinJSCCInferenceCIFAR10
from Semantic_SwinT import SwinJSCCInference
from SemanticNN_ import SemanticNN

import LDPC

warnings.filterwarnings("ignore")

# results_path = 'results/test_results.csv'

# with open(results_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['SNR', 'CBR', 'PSNR', 'MSSSIM'])

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
    default="CLIC21",
    choices=["kodak", "CLIC21", "ffhq"],
    help="specify the testset for HR models",
)
parser.add_argument(
    "--distortion-metric",
    type=str,
    default="MSE",
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
parser.add_argument(
    "--user",
    type=str,
    default="Bob",
    choices=["Bob", "Eve"],
    help="User at receiver",
)
args = parser.parse_args()



# =============================================================================
#  Configuration
# =============================================================================
class config():
    seed = 42
    pass_channel = True
    CUDA = False
    device = torch.device("mps")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000


    save_model_freq = 100
    image_dims = (3, 256, 256)
    base_path = "Dataset/HR_Image_dataset/"
    test_data_dir = ["Dataset/HR_Image_dataset/clic2021/test/"]

    train_data_dir = [base_path + '/clic2020/**',
                        base_path + '/clic2021/train',
                        base_path + '/clic2021/valid',
                        base_path + '/clic2022/val',
                        base_path + '/DIV2K_train_HR',
                        base_path + '/DIV2K_valid_HR']
    batch_size = 16
    downsample = 4
    channel_number = 96
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



# # =============================================================================
# #  Create the MS-SSIM metric and send it to the selected device
# # =============================================================================
CalcuSSIM = MS_SSIM(window_size=3, data_range=1.0, levels=4, channel=3).to(device)


def scale_8bit_weight(x):
    n = x.size()[1]  # sequence length
    w = range(8, 0, -1)  # np.power(2, range(7, -1, -1))
    for i in range(n):
        x[0, i] = x[0, i] * w[i % 8]
    return x


def img2bin(x1):
    x = x1.reshape(1, -1)  # convert to vector
    x = (x / 2 + 0.5) * 255  # inverse of regularization
    n = x.size()[1]  # sequence length
    y = torch.zeros([1, n * 8], dtype=int)
    for i in range(n):
        x2 = bin(int(min(max(x[0, i].item(), 0), 255)))[2:].zfill(8)
        # print(bin(int(x[0, i].item())),x2)
        for j in range(8):
            y[0, i * 8 + j] = int(x2[j])
    return y


def bin2img(y):
    n = int(y.size()[1] / 8)  # sequence length
    x = torch.zeros([1, n], dtype=torch.float)
    for i in range(n):
        arr = np.array(y[0, i * 8: (i + 1) * 8])
        y2 = ''.join(str(i) for i in arr)
        for j in range(8):
            x[0, i] = int(y2, 2)  # bin to digital
    x = (x / 255. - 0.5) * 2  # regularization again
    return x


def data_tf(x):
    x = x.resize((96, 96), 2)  # shape of x: (96, 96, 3)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
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
    """Converts variable to numpy."""
    # if torch.cuda.is_available():
    #     y = x.cpu()
    # else:
    #     y = x
    y=x.cpu()
    return y.data.numpy()


def calc_exinfo(Lp1, La1, g1, n1, n, k):
    X = torch.zeros_like(Lp1)
    for gi in range(g1 - 1):
        si = gi * k
        ei = (gi + 1) * k
        sni = gi * n
        eni = gi * n + k
        X[sni:eni] = Lp1[sni:eni] - La1[si:ei]
    sni = (g1 - 1) * n
    eni = (g1 - 1) * (n - k) + n1
    X[sni:eni] = Lp1[sni:eni] - La1[(g1 - 1) * k:]
    return X


def LDPC_enc(G, X1):
    n1 = X1.size()[1]
    n, k = G.shape  # n: code length, k: information bits length
    g1 = int(np.ceil(n1 / k))  # divide into groups
    C1 = torch.zeros([n * g1, 1])
    for gi in range(g1):
        X_g = torch.zeros([k, 1])  # padding "0" at the end of the last group
        si = gi * k
        ei = min((gi + 1) * k, n1)
        X_g[0:ei - si, 0] = X1[0, si:ei]
        C1[gi * n:(gi + 1) * n] = torch.tensor(LDPC.encode(G, X_g))
    return C1


def LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La, maxiter):
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lp = Lp1[si:ei]
        if La is None:
            La1 = None
        else:
            ski = gi * k
            if gi < g1 - 1:
                La1 = torch.zeros(1, n)
                eki = (gi + 1) * k
                La1[0, :k] = La[0, ski:eki]
            else:
                La1 = torch.ones(1, n)  # last bits are all 0，LLR should be positive
                La1[0, :n1 - ski] = La[0, ski:n1]
        Lp1[si:ei] = LDPC.decode_LLR(Lp, **DEC_para1, La=La1, maxiter=maxiter)
    return Lp1


def hard_decision(Lp2, g1, n1, n, k):
    X = torch.zeros([1, n1], dtype=int)
    for gi in range(g1 - 1):
        si = gi * k
        ei = (gi + 1) * k
        sni = gi * n
        eni = gi * n + k
        X[0, si:ei] = torch.tensor((Lp2[sni:eni] < 0).T)
    X[0, (g1 - 1) * k:] = torch.tensor((Lp2[(g1 - 1) * n:(g1 - 1) * (n - k) + n1] < 0).T)
    return X


def LDPC_dec_init(H, Y1, snr1, g1, n):
    Lc1 = torch.zeros_like(Y1)
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lc1[si:ei], DEC_para1 = LDPC.decoder_init(H, Y1[si:ei], snr1)
    return np.array(Lc1), DEC_para1


def save_img(img, path):
    imageio.imwrite(path, Image.fromarray(np.uint8(img * 255)))


def E_distance(x, y):
    x1 = np.array(x.detach().cpu())
    y1 = np.array(y)
    return ((x1 - y1) ** 2).sum() / x1.size


def sf_relay(x, snr1, rho, result_dir):
    # Main simulation function implementing the semantic relay system.
    # This function encodes an image into two bitstreams, sends them over noisy channels,
    # then iteratively performs joint LDPC decoding with extrinsic (side) information exchange.
    # ----- Preparation -----
    n = 900  # LDPC codeword length
    d_v = 2  # Number of parity-check equations per bit
    d_c = 3  # Number of bits per parity-check equation

    imgdir = result_dir + f"snr{snr1}-rho{rho:g}"
    os.makedirs(imgdir, exist_ok=True)
    print("x.shape: ", x.shape)

    # Convert the original image to a binary bitstream.
    X1 = img2bin(x) 

    X2_img = x.reshape([batch_size, 3, 96, 96]).to(device)

    # Reconstruct the image using the semantic encoder/decoder
    X2, CBR, _, msssim = simulator.infer(x, snr1)
    X2 = X2.reshape([batch_size, 3, 96, 96]).to(device)

    # LDPC PHY channel: prepare for encoding and transmission.
    n1 = X1.size()[1]
    H, G = LDPC.make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
    n, k = G.shape  # n: code length, k: information bits length

    # Divide the bitstreams into groups for LDPC encoding.
    g1 = int(np.ceil(n1 / k))  # Groups for original bitstream

    # LDPC encoding of both bitstreams.
    C1 = LDPC_enc(G, X1)

    # Simulate transmission by adding Gaussian noise.
    if (args.user == "Eve"):
        noise_factor = 100
        Y1 = LDPC.add_gaussian_noise(C1, snr1, seed=seed) + noise_factor
    else:
        Y1 = LDPC.add_gaussian_noise(C1, snr1, seed=seed)

    # Initialize LDPC decoding: get initial LLRs and decoder parameters.
    Lc1, DEC_para1 = LDPC_dec_init(H, Y1, snr1, g1, n)
    Lp1 = Lc1
    Lp1s = Lc1

    # Initialize extrinsic information placeholders.
    La1 = None  # Extrinsic info from semantic branch to LDPC decoder.

    for i in range(5):  # Joint iterative decoding loop. No. loopsf is originally 8 
        print(
            f"--------------------- LDPC joint dec [{i:d}] -----------------------------"
        )
        # ----- Joint Decoding: Use extrinsic information from the previous iteration -----
        Lp1 = LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La=La1, maxiter=1)
        # ----- Independent Decoding (without extrinsic info) for comparison -----
        Lp1s = LDPC_dec_LLR(Lp1s, DEC_para1, g1, n1, n, k, La=None, maxiter=1)

        # Hard decision outputs from the current LLRs.
        X1_hat = hard_decision(
            Lp1, g1, n1, n, k
        )  # Joint decoding result for original bitstream.
        X1s_hat = hard_decision(Lp1s, g1, n1, n, k)  # Independent decoding result.
        j1 = LDPC.BER(X1, X1_hat)
        s1 = LDPC.BER(X1, X1s_hat)
        print(f"BER s: {s1:g}, j: {j1:g}")
        
        #PSNR
        squared_difference = torch.nn.MSELoss(reduction='none')
        mse = squared_difference(x * 255., X2.clamp(0., 1.) * 255.)
        mse_val = mse.mean()
        psnr = 10 * (torch.log(255. * 255. / mse_val) / np.log(10))

        # Prepare images for visualization - Turn torch arr to np arr.
        X2_data = to_data(X2.reshape([batch_size, 3, 96, 96]))
        X1_data = to_data(bin2img(X1_hat).reshape([batch_size, 3, 96, 96]))
        X1s_data = to_data(bin2img(X1s_hat).reshape([batch_size, 3, 96, 96]))

        # Save side-by-side images for visual evaluation.
        merged = merge_images(to_data(x.reshape([batch_size, 3, 96, 96])), X2_data)
        save_img(merged, f"{imgdir:s}/origin-semantic-{e:d}-{i:d}.png")

        merged = merge_images(X1s_data, X1_data)
        ed1s = E_distance(x, X1s_data)
        ed1 = E_distance(x, X1_data)
        ed2 = E_distance(x, X2_data)
        print(f"EDs: {ed1s:g}, EDj: {ed1:g}, ED2: {ed2:g}")
        print(f"PSNR: {psnr:g}")
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
        if La1 is None:
            ex_info1 = Lp1
        else:
            ex_info1 = calc_exinfo(torch.tensor(Lp1, dtype=torch.float32).to(device), La1.t().to(device), g1, n1, n, k)
            

        # (2) Convert the semantic decoder's output (reconstructed image) back to a bitstream
        # and then map the bits to LLR-like values (0 -> 1 and 1 -> -1).
        ex_info2 = (img2bin(X2) * -2 + 1)

        Lp1_max = Lp1.max()
        # Lp2_max = Lp2.max()
        Lp2_max = -1

        # (3) Generate extrinsic information for the LDPC branch:
        # Process semantic branch output via a fusion function and scale it.
        La1 = LDPC.fc(ex_info2, rho / (i + 1), LLR_limit=50)
        La1 = scale_8bit_weight(La1) * (
            10 ** ((-5 + i * (1 - rho) * 2 - snr1 / 2 - 3) / 10)
        )
        # Explanation: With lower snr1, the system relies more on the semantic branch info.

        # (4) Generate extrinsic information for the semantic branch:
        # Process the extrinsic info from the LDPC branch and convert it through the semantic encoder.
        ex_fc1 = LDPC.fc(ex_info1, rho / (i + 1), LLR_limit=50)
        ex_fc1 = hard_decision(ex_fc1, g1, n1, n, k)
        ex_fc1 = bin2img(ex_fc1).reshape([batch_size, 3, 96, 96])
        La2 = scale_8bit_weight(semantic_coder.enc(ex_fc1) * -2 + 1) * (
            10 ** ((rho * (rho * 1000 + 10 * snr1) + 8 * i) / 10)
        )
        # Explanation: Higher snr1 and rho yield more extrinsic info from the LDPC branch to help the semantic decoder.

        La1_max = La1.max()
        La2_max = La2.max()
        print(
            f"Max Lp1: {Lp1_max:g}, ex_info2: {ex_info2.max():g}, La1: {La1_max:g}, Lp2: {Lp2_max:g},La2: {La2_max:g}"
        )

        # Optionally clip/scaling LLRs if they exceed thresholds.
        if Lp1_max > 200:
            Lp1 = Lp1 * (200 / Lp1_max)
        # if Lp2_max > 300:
        #     Lp2 = Lp2 * (300 / Lp2_max)
        # Update the hard decision for the next iteration.
        X1_hat = hard_decision(Lp1, g1, n1, n, k)

        # Log various metrics to a CSV file for later analysis.
        csv_path = result_dir + f"snr{snr1:d}-rho{rho:g}.csv"
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            data = [e, i, s1, j1, ed1s, ed1, ed2, psnr.item(), Lp1_max, La1_max, Lp2_max, La2_max]
            writer.writerow(data)


    # Return the final reconstructed image from the joint LDPC decoder.
    return bin2img(X1_hat).reshape([batch_size, 3, 96, 96])


if __name__ == "__main__":

    seed_torch()
    logger = logger_configuration(config, save_log=False)
    # logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    # train_loader, test_loader = get_loader(args, config)
    # data_tf = transforms.Compose([
    # transforms.Resize((96, 96)),  # resize to 96x96
    # transforms.ToTensor()
    # ])
    data_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    test_set = datasets.CIFAR10('./data', train=False, transform=data_tf, download=True)
    
    # Adjust batch_size as you like; e.g. 16, 32, etc.
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    global_step = 0
    
    # model_path = "SwinJSCC_w/SA&RA/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    if (args.trainset == "CIFAR10"):
        logger.info("Using CIFAR10 checkpoint.")
        model_path = "./SwinJSCC_w/ model/2025-02-28 23:23:04_EP5.model"
        simulator = SwinJSCCInferenceCIFAR10(model_path, device="mps", save_log=True, user=args.user)
    else:
        logger.info("Using DIV2K checkpoint.")
        model_path = "./SwinJSCC_w/SA&RA/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
        simulator = SwinJSCCInference(model_path, device="mps", save_log=True, user=args.user)

    if (args.user == "Eve"):
        result_dir = "./history/images_psnr_eve/"
    else:
        result_dir = "./history/images_psnr_test/"

    # -------------------------------------

    # Initialize the semantic coder and load pretrained weights if available.
    semantic_coder = SemanticNN(16, 20, 1, "mps")

    file_path = "semantic_coder.pkl"
    if os.path.exists(file_path):
        semantic_coder.load_state_dict(torch.load(file_path))
    semantic_coder.to(device)

    # Main simulation loop: iterate over epochs and batches.
    for e in range(epoch_len):

        counter = 0
        print("Length of test_data:", len(test_loader))
        for batch_idx, (im, label) in enumerate(test_loader):

            print("Epoch %d-%d:" % (e, counter))
            # im = Variable(im)
            im = im.to(device)


        # for batch_idx, batch in enumerate(test_loader):

            # Iterate over different corruption probabilities (rho) and SNR values.
            for rho in [0.05, 0.35, 1, 0]:
            # for rho in [1, 0]:
                for snr1 in range(-5, 10):
                # for snr1 in range(5, 6):
                # for snr1 in range (10,11):
                    print(
                        f"===================== rho={rho:g}, snr={snr1:d} ===================="
                    )
                    os.makedirs(result_dir, exist_ok=True)
                    fname = result_dir + f"snr{snr1:d}-rho{rho:g}.csv"
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
                                "PSNR",
                                "Lp1_max",
                                "La1_max",
                                "Lp2_max",
                                "La2_max",
                            ]
                            writer.writerow(data)

                    # Run the simulation function (which includes extrinsic information exchange).
                    sf_relay(im, snr1, rho, result_dir=result_dir)
                    # X2, CBR, psnr, msssim = simulator.infer(im.reshape([batch_size, 3, 96, 96]).to(device), snr1)

            counter += 1
            if counter >= 32:
                break

        # ------------------------------- Below code is to test a single image -------------------------------


        # index = 42
        # im, label = test_set[index]
        # im = im.unsqueeze(0)
        
        # print("Epoch %d-%d:" % (e, counter))
        # # im = Variable(im)
        # im = im.to(device)

        # # Iterate over different corruption probabilities (rho) and SNR values.
        # for rho in [0.05, 0.35, 1, 0]:
        #     snr_list =  [-5, 0, 5]
        #     for snr1 in snr_list:
        #     # for snr1 in range (10,11):
        #         print(
        #             f"===================== rho={rho:g}, snr={snr1:d} ===================="
        #         )
                # os.makedirs(result_dir, exist_ok=True)
                # fname = result_dir + f"snr{snr1:d}-rho{rho:g}.csv"
        #         if not os.path.exists(fname):
        #             with open(fname, mode="a", newline="") as file:
        #                 writer = csv.writer(file)
        #                 data = [
        #                     "epoch",
        #                     "iter_round",
        #                     "BERs",
        #                     "BERj",
        #                     "EDs",
        #                     "EDj",
        #                     "ED_semantic",
        #                     "PSNR",
        #                     "Lp1_max",
        #                     "La1_max",
        #                     "Lp2_max",
        #                     "La2_max",
        #                 ]
        #                 writer.writerow(data)

        #         # Run the simulation function (which includes extrinsic information exchange).
        #         sf_relay(im, snr1, rho)
        #         # X2, CBR, psnr, msssim = simulator.infer(im.reshape([batch_size, 3, 96, 96]).to(device), snr1)

        # counter += 1
        # if counter >= 32:
        #     break
        # #Take the first image only
        # break
