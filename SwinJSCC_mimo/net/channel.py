import torch.nn as nn
import numpy as np
import torch

class Channel(nn.Module):
    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        self.n_tx = 2  # Number of transmit antennas
        self.n_rx = 2  # Number of receive antennas
        self.coherence_time = args.coherence_time if hasattr(args, "coherence_time") else 100
        self.current_step = 0
        
        # Initialize the channel matrix H
        self.update_channel()
        
        if config.logger:
            config.logger.info(
                "【Channel】: Built {} MIMO channel with {} TX and {} RX antennas, SNR {} dB.".format(
                    args.channel_type, self.n_tx, self.n_rx, args.multiple_snr
                )
            )

    def update_channel(self):
        """Updates the channel matrix H assuming perfect CSI"""
        # Generate Rayleigh fading channel matrix
        real_part = torch.randn(self.n_rx, self.n_tx, dtype=torch.complex64, device=self.device)
        imag_part = torch.randn(self.n_rx, self.n_tx, dtype=torch.complex64, device=self.device)
        self.H = (real_part + 1j * imag_part) / np.sqrt(2)
        
        # Pre-compute channel inverse for detection (using perfect CSI)
        self.H_inv = torch.linalg.pinv(self.H)

    def gaussian_noise_layer(self, input_layer, noise_std):
        noise_dist = torch.distributions.Normal(0.0, noise_std / np.sqrt(2))
        noise_real = noise_dist.rsample(input_layer.shape).to(input_layer.device)
        noise_imag = noise_dist.rsample(input_layer.shape).to(input_layer.device)
        return input_layer + (noise_real + 1j * noise_imag)

    def complex_normalize(self, x, power):
        L = x.shape[0] // 2
        x_complex = x[:L] + 1j * x[L:]
        pwr = torch.mean(torch.abs(x_complex) ** 2)
        tx_power = power / self.n_tx
        out = np.sqrt(tx_power) * x_complex / torch.sqrt(pwr)
        return torch.cat([torch.real(out), torch.imag(out)]), pwr

    def forward(self, input, chan_param, avg_pwr=False):
        # Update channel if coherence time has elapsed
        self.current_step += 1
        if self.current_step >= self.coherence_time:
            self.update_channel()
            self.current_step = 0

        # Preserve input shape and flatten
        orig_shape = input.shape
        input_flat = input.reshape(-1)

        # Power normalization
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input_flat / torch.sqrt(avg_pwr * 2)
            pwr = avg_pwr
        else:
            channel_tx, pwr = self.complex_normalize(input_flat, power=1)

        # Convert to complex representation
        L = channel_tx.shape[0] // 2
        complex_signal = channel_tx[:L] + 1j * channel_tx[L:]
        symbols_per_antenna = L // self.n_tx
        tx_signals = complex_signal.reshape(symbols_per_antenna, self.n_tx)
        print(f"tx_signals shape: {tx_signals.shape}, dtype: {tx_signals.dtype}")

        # Apply MIMO channel effect
        rx_signals = torch.matmul(tx_signals.to(self.device), self.H.T)
        rx_power = torch.mean(torch.abs(rx_signals) ** 2)

        # Add noise
        snr_linear = 10 ** (chan_param / 10)
        noise_std = torch.sqrt(rx_power / snr_linear)
        rx_noisy = self.gaussian_noise_layer(rx_signals, noise_std)

        # Direct channel inversion using perfect CSI
        detected_signal = torch.matmul(rx_noisy, self.H_inv.T)

        # Power normalization of detected signal
        detected_power = torch.mean(torch.abs(detected_signal) ** 2)
        detected_signal = detected_signal * torch.sqrt(1.0 / detected_power)

        # Convert back to real-valued representation
        combined_signal = detected_signal.reshape(-1)
        channel_output = torch.cat([torch.real(combined_signal), torch.imag(combined_signal)])

        # Restore original dimensions and apply final scaling
        channel_output = channel_output.reshape(orig_shape)
        scaling_factor = torch.sqrt(pwr if not avg_pwr else avg_pwr)
        
        return channel_output * scaling_factor