import torch
import torch.nn as nn
import argparse
from datetime import datetime
import time
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

from net.network import SwinJSCC
from loss.distortion import MS_SSIM
from utils import logger_configuration

class config():
    seed = 42
    pass_channel = True
    CUDA = False
    device = torch.device("mps")
    norm = False
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    save_model_freq = 100
    image_dims = (3, 256, 256)  # Default to 256×256 if used in training
    train_data_dir = []
    test_data_dir = []
    batch_size = 1
    downsample = 4
    channel_number = None

    # Example for the 'base' model_size on 256×256
    encoder_kwargs = dict(
        img_size=(256, 256),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    decoder_kwargs = dict(
        img_size=(256, 256),
        embed_dims=[320, 256, 192, 128],
        depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )


class SwinJSCCInference:
    def __init__(self, model_path, device="mps", args=None, save_log=True):
        self.device = torch.device(device)
        if args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--trainset', type=str, default='DIV2K')
            parser.add_argument('--testset', type=str, default='ffhq')
            parser.add_argument('--distortion-metric', type=str, default='MSE')
            parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA')
            parser.add_argument('--channel-type', type=str, default='awgn')
            parser.add_argument('--C', type=str, default='96')
            parser.add_argument('--multiple-snr', type=str, default='10')
            parser.add_argument('--model_size', type=str, default='base')
            parser.add_argument('--training', action='store_false')
            args = parser.parse_args([])
        self.args = args

        self.config = config()
        self.config.device = self.device
        self.logger = logger_configuration(self.config, save_log=save_log)
        self.logger.info("Initializing SwinJSCCInference...")

        # Setup distortion metric
        if self.args.trainset == 'CIFAR10':
            self.CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(self.device)
        else:
            self.CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(self.device)

        # Initialize network
        self.net = SwinJSCC(self.args, self.config)
        self.load_weights(model_path)
        self.net.to(self.device)
        self.net.eval()

        self.logger.info("Model loaded and set to evaluation mode.")

    def load_weights(self, model_path):
        self.logger.info(f"Loading model weights from {model_path}")
        pretrained = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(pretrained, strict=True)
        del pretrained
        self.logger.info("Weights loaded successfully.")

    def infer(self, image_tensor, SNR_value=None, rate_value=None):
        """
        Simulate the semantic transmission on a single image and log the outputs.
        If input is 96x96, pad it to 256x256, then remove padding from the output
        and compute PSNR/MS-SSIM with the original 96x96 image.
        """
        _, _, H, W = image_tensor.shape

        # (1) SNR / rate defaults
        if SNR_value is None:
            SNR_value = int(self.args.multiple_snr.split(",")[0])
        if rate_value is None:
            rate_value = int(self.args.C.split(",")[0])

        # Keep a copy of the original 96x96 if we need it for metrics
        orig_image_96 = None

        # # (2) Conditional padding
        # if (H, W) == (96, 96):
        #     self.logger.info("Input is 96x96, padding to 256x256.")
        #     # Save original for final metrics
        #     orig_image_96 = image_tensor.clone()

        #     # Create a padded canvas of 256x256
        #     # We'll place the 96x96 in the top-left corner (you could center it if you prefer).
        #     padded_image = torch.zeros((1, 3, 256, 256), device=image_tensor.device)

        #     # Copy the original 96x96 region into the top-left corner of the 256x256 canvas
        #     padded_image[:, :, :96, :96] = orig_image_96
        #     # Now the "image_tensor" that we feed to the network is this padded version
        #     image_tensor = padded_image
        # elif (H, W) == (256, 256):
        #     self.logger.info("Input is 256x256; proceeding as is.")
        # elif (H, W) == (768, 768):
        #     self.logger.info("Input is 768x768; proceeding as is.")
        # else:
        #     msg = (f"Unsupported input resolution {H}x{W}. "
        #         "Please use 96x96, 256x256, or 768x768.")
        #     self.logger.error(msg)
        #     raise ValueError(msg)

        # (3) Move the (possibly padded) input to the device
        image_tensor = image_tensor.to(self.device)
        # self.logger.info(f"Starting inference with SNR={SNR_value}, rate={rate_value}")
        start_time = time.time()

        # (4) Forward pass
        with torch.no_grad():
            # net(...) returns (recon_image, CBR, SNR_out, mse, loss_G)
            recon_image, CBR, SNR_out, net_mse, loss_G = self.net(image_tensor, SNR_value, rate_value)

        elapsed = time.time() - start_time

        # (5) If we padded the input to 256x256, remove the padding from the output
        #     The network's output is 256x256, so slice out the top-left 96x96 region
        #     (or whichever region you padded to).
        if orig_image_96 is not None:
            recon_image_96 = recon_image[:, :, :96, :96]
        else:
            recon_image_96 = recon_image

        # (6) Compute final metrics between the original 96x96 and the 96x96 reconstruction
        #     If we did not pad, just compare the input with the output directly.
        if orig_image_96 is not None:
            ref_image = orig_image_96.to(self.device)
        else:
            ref_image = image_tensor  # already 256x256 or 768x768

        # Compute MSE in [0,1] range
        # mse_val = torch.mean((ref_image - recon_image_96.clamp(0., 1.)).pow(2))
        
        #PSNR
        squared_difference = torch.nn.MSELoss(reduction='none')
        mse = squared_difference(ref_image * 255., recon_image_96.clamp(0., 1.) * 255.)
        mse_val = mse.mean()
        psnr = 10 * (torch.log(255. * 255. / mse_val) / np.log(10))

        if mse_val.item() > 0:
            # psnr = 10 * (torch.log(255. * 255. / mse_val) / np.log(10))
            msssim_val = 1 - self.CalcuSSIM(ref_image, recon_image_96.clamp(0., 1.)).mean().item()
        else:
            psnr = 100.0
            msssim_val = 100.0

        # (7) Logging
        self.logger.info(
            "Inference done | Time: {:.3f}s | CBR: {:.4f} | SNR: {:.1f} "
            "| PSNR: {:.3f} | MS-SSIM: {:.3f}".format(
                elapsed, CBR, SNR_out,
                psnr.item() if isinstance(psnr, torch.Tensor) else psnr,
                msssim_val
            )
        )
        # print(
        #     f"Inference Time: {elapsed:.3f}s, "
        #     f"CBR: {CBR:.4f}, "
        #     f"SNR: {SNR_out:.1f}, "
        #     f"PSNR: {psnr:.3f}, "
        #     f"MS-SSIM: {msssim_val:.3f}"
        # )
        # print("Final output shape:", recon_image_96.shape)

        # (8) Return the unpadded reconstruction (96x96) if applicable
        return recon_image_96, CBR, psnr.item() if isinstance(psnr, torch.Tensor) else psnr, msssim_val

if __name__ == '__main__':
    # 1) Prepare the CIFAR10 test set (96x96)
    data_tf = transforms.Compose([
        transforms.Resize((96, 96)),  # resize to 96x96
        transforms.ToTensor()
    ])
    test_set = datasets.CIFAR10('./data', train=False, transform=data_tf, download=True)
    
    # Adjust batch_size as you like; e.g. 16, 32, etc.
    batch_size = 1
    test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 2) Path to your pretrained model
    model_path = "SwinJSCC_w/SA&RA/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"

    # 3) Instantiate the inference class
    #    Make sure you have already defined or imported SwinJSCCInference above.
    simulator = SwinJSCCInference(model_path, device="mps", save_log=True)

    # 4) Run inference on each batch of the test data
    #    Each batch is shape [batch_size, 3, 96, 96].
    for batch_idx, (images, labels) in enumerate(test_data):
        for snr_ in range (-5, 2):
            # Pass the entire batch (images) to simulator.infer()
            # The class will automatically resize/pad to 256x256 if needed (per your logic).
            output_images = simulator.infer(images, SNR_value=snr_)

            # If you want to do something with 'output_images' (e.g. save them or compute metrics),
            # you can do that here.

            # if batch_idx % 10 == 0:
                # print(f"Processed batch {batch_idx}/{len(test_data)}")