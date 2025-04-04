import copy
import numpy as np

import torch
from torch import nn

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

class SemanticNN(nn.Module):
    def __init__(self, out_ch=16, epoch_len=20, batch_size=16, device="cpu"):
        # Initialize the semantic encoder-decoder network.
        super(SemanticNN, self).__init__()
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.device = device


        # Encoder layers
        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        # Decoder (transpose convolution) layers
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=2, stride=1, padding=0)

    def enc(self, x):
        # Semantic encoder: compress the image into a latent representation.
        out = self.conv1(x.to(self.device))
        out = self.conv2(out)
        out = self.conv3(out)

        # Scale and quantize the feature map
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = torch.div(out, out_max)

        # Quantize: scale to 256 levels, convert to int, then back to float.
        out_tmp = torch.mul(out_tmp, 256)
        out_tmp = out_tmp.clone().type(torch.int)
        out_tmp = out_tmp.clone().type(torch.float32)
        out_tmp = torch.div(out_tmp, 256)

        out = torch.mul(out_tmp, out_max)

        # Convert quantized features to a binary bitstream.
        out = img2bin(out)
        return out

    def dec(self, x):
        # Semantic decoder: reconstruct the image from the binary bitstream.
        out = bin2img(x)
        out = out.reshape([self.batch_size, 16, 23, 23])  # reshape into feature map dimensions

        out = out.to(self.device)
        out = self.tconv3(out)
        out = self.tconv4(out)
        out = self.tconv5(out)

        # Scale and quantize the output image
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = torch.div(out, out_max)

        # Quantize the output
        out_tmp = torch.mul(out_tmp, 256)
        out_tmp = out_tmp.clone().type(torch.int)
        out_tmp = out_tmp.clone().type(torch.float32)
        out_tmp = torch.div(out_tmp, 256)

        out = torch.mul(out_tmp, out_max)
        return out

    def forward(self, x):
        return x