from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

# ----------------- Some Blocks -----------------
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ----------------- JEPA2D -----------------
class Encoder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate the number of ResBlocks
        self.num_blocks = config.num_block
        layers = []
        in_channel = 2
        out_channel = config.out_channel    # initial output channel
        num_resblock = config.num_resblock

        # [B, 2, 65, 65] -> [B, _, 33, 33] -> [B, _, 17, 17] -> [B, _, 9, 9] -> [B, _, 5, 5]
        for i in range(self.num_blocks):
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
            )

            for j in range(num_resblock):
                layers.append(
                    ResBlock(out_channel, out_channel)
                )

            in_channel = out_channel
            out_channel *= 2

        layers.append(nn.Conv2d(in_channel, 1, kernel_size=1))  # (B, 1, emb_w, emb_w)
        self.encoder = nn.Sequential(*layers)   
        
        
    def forward(self, x):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, emb_w, emb_w)
        """

        x = self.encoder(x)
        return x

class Predictor2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.emb_w = 65
        for i in range(config.num_block):
            self.emb_w = int((self.emb_w + 1) // 2)  # 65 -> 33 -> 17 -> 9 -> 5
        
        self.emb_dim = self.emb_w * self.emb_w

        self.fc = nn.Linear(2, self.emb_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(2, self.hidden_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channel, 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

    def forward(self, states, actions):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, emb_w, emb_w)

        Output: (B, 1, emb_w, emb_w)
        """
        B, _, emb_h, emb_w = states.shape
        assert self.emb_dim == emb_h * emb_w, f"emb_dim {self.emb_dim} is not equal to emb_h * emb_w {emb_h * emb_w}"

        action_emb = self.fc(actions).view(B, 1, emb_h, emb_w)  # (B, 1, emb_w, emb_w)
        x = torch.cat((states, action_emb), dim=1)  # (B, 2, emb_w, emb_w)

        x = self.conv(x)  # (B, 1, emb_w, emb_w)
        return x

# ----------------------------- helper ------------------------------------------
def off_diagonal(x):
    # Return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    """
    Args:
        x, y: [B, D] - Flattened embedding from encoder and predictor
    """

    # Invariance loss (MSE)
    repr_loss = F.mse_loss(x, y)

    # Variance loss
    std_x = torch.sqrt(x.var(dim=0) + eps)
    std_y = torch.sqrt(y.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    # Covariance loss
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() / x.shape[1] + off_diagonal(cov_y).pow_(2).sum() / y.shape[1]

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


class JEPA2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.encoder = Encoder2D(config)
        self.predictor = Predictor2D(config)

    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, emb_w, emb_w], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, emb_w, emb_w], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T > 1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states)  # (B*T, 1, emb_w, emb_w)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, emb_w, emb_w)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, emb_w, emb_w)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, emb_w, emb_w)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, emb_w, emb_w)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions)  # (B*(T-1), 1, emb_w, emb_w)
            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, emb_w, emb_w)

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0])  # (B, 1, emb_w, emb_w) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  # [B, T, emb_dim]
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")

    # --------------------------------- implementation of mse --------------------------------- 
    def loss_mse(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]

        Output:
            loss: scalar
        """

        loss = F.mse_loss(enc_states[:, 1:], pred_states[:, :-1], reduction='mean')
        return loss

    # --------------------------------- implementation of vicreg --------------------------------- 
    def loss_vicreg(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]

        Output:
            loss: scalar
        """
        B, T, _, H, W = enc_states.shape
        x = enc_states[:, 1:].reshape(B * (T - 1), -1)
        y = pred_states[:, :-1].reshape(B * (T - 1), -1)

        return vicreg_loss(x, y)

    # --------------------------------- implementation of regularization --------------------------------- 
    def loss_R(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]

        Output:
            loss: scalar
        """

        # TODO
        pass

# ---------------- Models -------------------
MODELS: dict = {
    # "Model_Name": Model_Class
    "JEPA2D": JEPA2D
    # add models here
}
