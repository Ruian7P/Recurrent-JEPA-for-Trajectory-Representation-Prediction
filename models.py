from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math
import copy



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




# ----------------- Some Helpers -----------------
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.dropout2 = nn.Dropout2d(dropout)

        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

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



class CBAMChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class CBAMSpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        compressed = torch.cat([max_out, mean_out], dim=1)
        scale = torch.sigmoid(self.compress(compressed))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = CBAMChannelGate(in_channels, reduction)
        self.spatial_gate = CBAMSpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x



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
        dropout = config.dropout

        # [B, 2, 65, 65] -> [B, _, 33, 33] -> [B, _, 17, 17] -> [B, _, 9, 9] -> [B, _, 5, 5]
        for i in range(self.num_blocks):
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
            )

            for j in range(num_resblock):
                layers.append(
                    ResBlock(out_channel, out_channel, dropout)
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
        dropout = config.dropout

        self.conv = nn.Sequential(
            nn.Conv2d(2, self.hidden_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
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
    

class Regularizer2D(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.emb_w= int(math.sqrt(repr_dim))  # Calculate the side of the 2D embedding
        self.action_reg_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 2D conv layer
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Output single channel
            nn.Flatten(),  # Flatten to prepare for linear mapping
            nn.Linear(self.emb_w * self.emb_w, 2),  # Map to action_dim
        )

    def forward(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T-1, 1, emb_w, emb_w]
            pred_states: [B, T-1, 1, emb_w, emb_w]
        Output: 
            predicted_actions: [B(T-1), 2]
        """
        # Calculate embedding differences
        embedding_diff = pred_states - enc_states  # [B, T-1, 1, emb_w, emb_w]
        embedding_diff = embedding_diff.view(-1, 1, self.emb_w, self.emb_w)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # [B(T-1), 2]
        return predicted_actions


class JEPA2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.teacher_forcing = config.teacher_forcing

        self.encoder = Encoder2D(config)
        self.predictor = Predictor2D(config)
        self.emb_w = 65
        for i in range(config.num_block):
            self.emb_w = int((self.emb_w + 1) // 2)  # 65 -> 33 -> 17 -> 9 -> 5

        self.repr_dim = self.emb_w * self.emb_w

        self.regularizer = Regularizer2D(self.repr_dim)
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, emb_w, emb_w], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, emb_w, emb_w], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
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
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, emb_w, emb_w)

            self.enc_states = enc_states
            self.pred_states = pred_states 

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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:])  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff
        return loss


# ----------------- JEPA2D2 -----------------  

class Encoder2D2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate the number of ResBlocks
        self.num_blocks = config.num_block

        in_channel = 2
        out_channel = config.out_channel    # initial output channel
        num_resblock = config.num_resblock
        dropout = config.dropout

        # [B, 2, 65, 65] -> [B, _, 32, 32] -> [B, _, 16, 16] -> [B, _, 8, 8] -> [B, _, 4, 4]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
        )
        
        self.cbam = CBAM(out_channel, reduction= 8)

        self.project = nn.Conv2d(out_channel, 4, kernel_size = 1)
        
        
    def forward(self, x):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, emb_w, emb_w)
        """

        x = self.encoder(x)
        x = self.cbam(x)
        x = self.project(x)

        return x
            

class Predictor2D2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.emb_w = 16
        
        self.emb_dim = self.emb_w * self.emb_w

        self.fc = nn.Linear(2, self.emb_dim)
        dropout = config.dropout

        self.conv = nn.Sequential(
            nn.Conv2d(2, self.hidden_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
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
    

class Regularizer2D2(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.emb_w= int(math.sqrt(repr_dim))  # Calculate the side of the 2D embedding
        self.action_reg_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 2D conv layer
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Output single channel
            nn.Flatten(),  # Flatten to prepare for linear mapping
            nn.Linear(self.emb_w * self.emb_w, 2),  # Map to action_dim
        )

    def forward(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T-1, 1, emb_w, emb_w]
            pred_states: [B, T-1, 1, emb_w, emb_w]
        Output: 
            predicted_actions: [B(T-1), 2]
        """
        # Calculate embedding differences
        embedding_diff = pred_states - enc_states  # [B, T-1, 1, emb_w, emb_w]
        embedding_diff = embedding_diff.view(-1, 1, self.emb_w, self.emb_w)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # [B(T-1), 2]
        return predicted_actions


class JEPA2D2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.teacher_forcing = config.teacher_forcing

        self.encoder = Encoder2D2(config)
        self.predictor = Predictor2D2(config)
        self.emb_w = 16

        self.repr_dim = self.emb_w * self.emb_w

        self.regularizer = Regularizer2D2(self.repr_dim)
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, emb_w, emb_w], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, emb_w, emb_w], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
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
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, emb_w, emb_w)

            self.enc_states = enc_states
            self.pred_states = pred_states 

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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:])  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss


    
# ----------------- JEPA -----------------
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate the number of ResBlocks
        self.num_blocks = config.num_block

        layers = []
        in_channel = 2
        out_channel = config.out_channel    # initial output channel
        num_resblock = config.num_resblock
        self.repr_dim = out_channel * 2 ** (self.num_blocks - 1)  # final output channel
        dropout = config.dropout

        # [B, 2, 65, 65] -> [B, 16, 33, 33] -> [B, 32, 17, 17] -> [B, 64, 9, 9] -> [B, 128, 5, 5]
        for i in range(self.num_blocks):
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
            )

            for j in range(num_resblock):
                layers.append(
                    ResBlock(out_channel, out_channel, dropout)
                )

            in_channel = out_channel
            out_channel *= 2

        self.encoder = nn.Sequential(*layers)   # [B, repr_dim, w, w]

        # [B, repr_dim, w, w] -> [B, repr_dim]
        self.pool = nn.AdaptiveAvgPool2d(1) # [B, repr_dim, 1, 1]
        self.fc = nn.Linear(self.repr_dim, self.repr_dim)
        
        
    def forward(self, x):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, repr_dim)
        """

        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, -1)  # [B, 1, repr_dim]
        return x
            

class Predictor(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr_dim = config.out_channel * 2 ** (config.num_block - 1)  # final output channel

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.repr_dim + 2, self.repr_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(self.repr_dim * 2, self.repr_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.repr_dim, self.repr_dim)
        # )

        self.action_fc = nn.Linear(2, self.repr_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.repr_dim + self.repr_dim, int(self.repr_dim)),
            nn.ReLU(),
            nn.Linear(int(self.repr_dim), self.repr_dim)
        )



    def forward(self, states, actions):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, repr_dim)

        Output: (B, 1, repr_dim)
        """
        # x = torch.cat((states, actions), dim= -1) # (B, 1, repr_dim + 2)
        # x = x.squeeze(1)  # (B, repr_dim + 2)
        # x = self.mlp(x)  # (B, repr_dim)
        # x = x.unsqueeze(1)  # (B, 1, repr_dim) 

        self.emb_action = self.action_fc(actions.squeeze(1))  # (B, repr_dim)
        x = torch.cat((states.squeeze(1), self.emb_action), dim= -1) # (B, repr_dim * 2)
        x = self.mlp(x)
        x = x.unsqueeze(1)
        return x

    
class Regularizer(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_reg_net = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, 2)
        )

    def forward(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T-1, 1, repr_dim]
            pred_states: [B, T-1, 1, repr_dim]
        Output: 
            predicted_actions: [B(T-1), 2]
        """
        # Calculate embedding differences
        embedding_diff = pred_states - enc_states  # [B, T-1, 1, repr_dim]
        embedding_diff = embedding_diff.view(-1, self.repr_dim)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # [B(T-1), 2]
        return predicted_actions


class JEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.teacher_forcing = config.teacher_forcing

        self.encoder = Encoder(config)
        self.predictor = Predictor(config)

        self.repr_dim = config.out_channel * 2 ** (config.num_block - 1)  # final output channel

        self.regularizer = Regularizer(self.repr_dim)


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, repr_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, repr_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states)  # (B*T, 1, repr_dim)
            _, _, repr_dim = enc_states.shape

            enc_states = enc_states.view(B, T, 1, repr_dim) # (B, T, 1, repr_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, repr_dim), device = enc_states.device)  # (B, T, 1, repr_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :].contiguous() # (B, T-1, 1, repr_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, repr_dim)  # (B*(T-1), 1, repr_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions)  # (B*(T-1), 1, repr_dim)
            next_states = next_states.view(B, T - 1, 1, repr_dim)
            pred_states[:, 1:] = next_states   # (B, T, 1, emb_w, emb_w)

            self.enc_states = enc_states
            self.pred_states = pred_states

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]    # T-1

            enc_states = self.encoder(states[:, 0])  # (B, 1, repr_dim) s_0
            _, _, repr_dim = enc_states.shape

            h = enc_states  # [B, 1, repr_dim]
            h = h.unsqueeze(1)  # [B, 1, 1, repr_dim]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action)  # -> [B, 1, repr_dim]
                h = h.unsqueeze(1)  # [B, 1, 1, repr_dim]
                pred_states.append(h)
                

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, repr_dim]
            pred_states = pred_states.view(B, T, repr_dim)  # [B, T, emb_dim]
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
    def loss_mse(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, repr_dim]
            pred_states: [B, T, 1, repr_dim]

        Output:
            loss: scalar
        """

        loss = F.mse_loss(enc_states[:, 1:], pred_states[:, :-1], reduction='mean')
        return loss


    def loss_vicreg(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, repr_dim]
            pred_states: [B, T, 1, repr_dim]

        Output:
            loss: scalar
        """
        B, T, _, repr_dim = enc_states.shape
        x = enc_states[:, 1:].reshape(B * (T - 1), -1)
        y = pred_states[:, :-1].reshape(B * (T - 1), -1)

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)



    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, repr_dim]
            pred_states: [B, T, 1, repr_dim]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:])  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')

        # emb_action = self.predictor.emb_action
        # loss = (emb_action**2).mean()

        return loss

         
    def compute_loss(self, print_loss=False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff 

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss
            
    
            
# ----------------- JEPA2Dv1 -----------------
class Encoder2Dv1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate the number of ResBlocks
        self.num_blocks = config.num_block

        layers = []
        in_channel = 2
        out_channel = config.out_channel    # initial output channel
        num_resblock = config.num_resblock
        emb_w = 65
        dropout = config.dropout

        # [B, 2, 65, 65] -> [B, _, 33, 33] -> [B, _, 17, 17] -> [B, _, 9, 9] -> [B, _, 5, 5]
        for i in range(self.num_blocks):
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
            )

            for j in range(num_resblock):
                layers.append(
                    ResBlock(out_channel, out_channel, dropout)
                )

            in_channel = out_channel
            out_channel *= 2
            emb_w = (emb_w + 1) // 2

        
        self.encoder = nn.Sequential(*layers)   
        repr_dim  = out_channel // 2
        
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, repr_dim)
        """

        x = self.encoder(x) # (B, repr_dim, emb_w, emb_w)
        B, C, emb_h, emb_w = x.shape
        x = x.view(B, -1, C)  # (B, num_patches, repr_dim)
        x = x + emb_pos
        x = x.view(B, 1, -1, C) # (B, 1, num_patches, repr_dim)
        return x
            

class Predictor2Dv1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_w = 65
        for i in range(config.num_block):
            self.emb_w = int((self.emb_w + 1) // 2)  # 65 -> 33 -> 17 -> 9 -> 5
        
        self.num_patches = self.emb_w * self.emb_w
        self.repr_dim = config.out_channel * 2 **(config.num_block - 1)  # final output channel

        self.mlp = nn.Sequential(
            nn.Linear(self.repr_dim + 2, self.repr_dim),
            nn.LayerNorm(self.repr_dim),
            nn.ReLU(),
            nn.Linear(self.repr_dim, self.repr_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, repr_dim)

        Output: (B, 1, num_patches, repr_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Dv1(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim

        self.mlp = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, 2)
        )

    def forward(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """
        # Remove the singleton dim
        enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
        pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

        # Compute token-wise difference
        delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

        # Mean over tokens (per frame)
        delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

        # Flatten over batch and time
        B, Tm1, _ = delta_mean.shape
        delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

        # Predict action
        action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

        return action_pred



class JEPA2Dv1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.emb_w = 65
        for i in range(config.num_block):
            self.emb_w = int((self.emb_w + 1) // 2)  # 65 -> 33 -> 17 -> 9 -> 5

        self.num_patches = self.emb_w * self.emb_w

        repr_dim = config.out_channel * 2 ** (config.num_block - 1)  # final output channel


        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, repr_dim))  # [1, num_patches, repr_dim]

        self.encoder = Encoder2Dv1(config)
        self.predictor = Predictor2Dv1(config)


        self.regularizer = Regularizer2Dv1(repr_dim)

        self.repr_dim = repr_dim * self.num_patches  


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, repr_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, repr_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, repr_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, repr_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, repr_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, repr_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, repr_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, repr_dim)
            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, repr_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, repr_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:])  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss





# ----------------- JEPA2Dv2 -----------------
class Encoder2Dv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2

        self.proj = nn.Conv2d(2, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Dv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.emb_dim, self.emb_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Dv2(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Dv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Dv2(config)
        self.predictor = Predictor2Dv2(config)


        self.regularizer = Regularizer2Dv2(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss




# ----------------- JEPA2Da1 -----------------
class Encoder2Da1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2

        self.proj = nn.Conv2d(2, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Da1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel
        self.dropout = config.dropout
        self.num_heads = config.num_heads

        self.fc = nn.Sequential(
            nn.Linear(2, self.emb_dim//2),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_dim//2, self.emb_dim)
        )

        self.project = nn.Linear(self.emb_dim*2, self.emb_dim)

        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(self.dropout)
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)
          emb_pos: (1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, emb_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        states = states.view(B, num_patches, emb_dim)  # (B, num_patches, emb_dim)
        if emb_pos is not None:
            states = states + emb_pos
        
        action = actions.view(B, 2)  # (B, 2)
        emb_action = self.fc(action)  # (B, emb_dim)

        emb_action = emb_action.view(B, 1, emb_dim)  # (B, 1, emb_dim)
        emb_action = emb_action.expand(-1, num_patches, -1)  # (B, num_patches, emb_dim)

        states = torch.cat((states, emb_action), dim=-1) # (B, num_patches, emb_dim* 2)
        states = self.project(states)  # (B, num_patches, emb_dim)

        x = self.norm1(states)  # (B, num_patches, emb_dim)
        attention,_ = self.attention(x, x, x) # (B, num_patches, emb_dim)
        states = 1 *states + attention  # (B, num_patches, emb_dim)

        x = self.norm2(states)  # (B, num_patches, emb_dim)
        x = self.ffn(x)  # (B, num_patches, emb_dim)
        x = 1 * states + x  # (B, num_patches, emb_dim)

        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
    

class Regularizer2Da1(nn.Module):
    def __init__(self, repr_dim, config):
        super().__init__()
        self.repr_dim = repr_dim
        self.config = config
        self.r = self.config.r

        if self.r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif self.r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if self.r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if self.r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Da1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Da1(config)
        self.predictor = Predictor2Da1(config)


        self.regularizer = Regularizer2Da1(self.emb_dim, config)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")

            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:])  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")

        
        return loss   







# ----------------- JEPA2Dv2B -----------------
class Encoder2Dv2B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2

        self.proj = nn.Conv2d(2, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Dv2B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.emb_dim, self.emb_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Dv2B(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Dv2B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Dv2B(config)
        self.predictor = Predictor2Dv2B(config)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.regularizer = Regularizer2Dv2B(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """
        Update the target encoder with a moving average of the current encoder.
        """
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            # Update the target encoder
            with torch.no_grad():
                target_enc_states = self.target_encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
                target_enc_states = target_enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)
            self.target_enc_states = target_enc_states

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 


            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def loss_byol(self, enc_states, pred_states):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]

        Output:
            loss: scalar
        """
        B, T, _, H, W = enc_states.shape
        x = enc_states[:, :-1].reshape(B * (T - 1), -1)
        y = pred_states[:, 1:].reshape(B * (T - 1), -1)

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        # Compute the BYOL loss
        loss = 2 - 2 * (x * y).sum(dim=-1)  # [B*(T-1), 1]
        loss = loss.mean()

        return loss


    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        byol = self.loss_byol(self.target_enc_states, self.pred_states)
        loss = vicreg + reg * self.config.reg_coeff + byol * self.config.byol_coeff

        if print_loss:
            print(f"loss: {loss.item()}, vicreg: {vicreg.item()}, reg: {reg.item() * self.config.reg_coeff}, byol: {byol.item() * self.config.byol_coeff}")

        return loss



# ----------------- JEPA2Da2 -----------------
class Encoder2Da2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2
        self.num_heads = config.num_heads
        self.dropout = config.dropout


        self.proj = nn.Conv2d(2, self.out_channel, kernel_size = self.patch_size, stride = self.patch_size) 

        self.attention = nn.MultiheadAttention(
            embed_dim=self.out_channel,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_channel * 2, self.out_channel),
            nn.Dropout(self.dropout)
        )

        self.norm1 = nn.LayerNorm(self.out_channel)
        self.norm2 = nn.LayerNorm(self.out_channel)

        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        attention = self.attention(x, x, x)[0]  # (B, num_patches, emb_dim)
        x = self.norm1(x + attention)  # (B, num_patches, emb_dim)
        out = self.mlp(x)  # (B, num_patches, emb_dim)
        x = self.norm2(x + out)  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)
        return x
            

class Predictor2Da2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.emb_dim, self.emb_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Da2(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Da2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Da2(config)
        self.predictor = Predictor2Da2(config)


        self.regularizer = Regularizer2Da2(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss


# ----------------- JEPA2Dv3 -----------------
class Encoder2Dv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2
        self.dropout = config.dropout

        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size = 7, stride = 2, padding = 2), # (B, 2, 65, 65) -> (B, 16, 32, 32)
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout)
        )
            

        self.proj = nn.Conv2d(16, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.conv(x)  
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Dv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.emb_dim, self.emb_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Dv3(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Dv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Dv3(config)
        self.predictor = Predictor2Dv3(config)


        self.regularizer = Regularizer2Dv3(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss






# ----------------- JEPA2Dv4 -----------------
class Encoder2Dv4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2
        self.dropout = config.dropout

        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size = 7, stride = 2, padding = 2), # (B, 2, 65, 65) -> (B, 16, 32, 32)
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout)
        )
            
        self.cbam = CBAM(16, reduction = 8)
        self.proj = nn.Conv2d(16, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.conv(x)  
        x = self.cbam(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Dv4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channel = config.hidden_channel
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.emb_dim, self.emb_dim)
        )



    def forward(self, states, actions, emb_pos=None):
        """
        Args:
          actions: (B, 1, 2)
          states: (B, 1, num_patches, emb_dim)

        Output: (B, 1, num_patches, emb_dim)
        """
        B, _, num_patches, repr_dim = states.shape
        assert self.num_patches == num_patches, f"num_patches {self.num_patches} is not equal to num_patches {num_patches}"

        action_emb = actions.repeat(1, num_patches, 1)  # (B, num_patches, 2)
        action_emb = action_emb.view(B, 1, num_patches, 2)  # (B, 1, num_patches, 2)
        states = states.view(B, num_patches, repr_dim)  # (B, num_patches, repr_dim)
        states = states + emb_pos  # (B, num_patches, repr_dim)
        states = states.view(B, 1, num_patches, repr_dim)  # (B, 1, num_patches, repr_dim)
        x = torch.cat((states, action_emb), dim=-1) # (B, 1, num_patches, repr_dim + 2)
        x = x.squeeze(1)     # (B, num_patches, repr_dim + 2)
        x = x.view(B * num_patches, -1)  # (B * num_patches, repr_dim + 2)
        x = self.mlp(x)
        x = x.view(B, 1,  num_patches, -1)  # (B, 1, num_patches, repr_dim)
        
        return x
    

class Regularizer2Dv4(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Dv4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Dv4(config)
        self.predictor = Predictor2Dv4(config)


        self.regularizer = Regularizer2Dv4(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], 0)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, 0)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss


# ----------------- JEPA2Ds1 -----------------
class Encoder2Ds1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channel = config.out_channel   
        self.patch_size = config.patch_size
        self.num_patches = (65 // self.patch_size) ** 2

        self.proj = nn.Conv2d(2, out_channel, kernel_size = self.patch_size, stride = self.patch_size)  
        
        
    def forward(self, x, emb_pos=None):
        """
        Args: (B, 2, 65, 65)
        Output: (B, 1, num_patches, emb_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + emb_pos  # (B, num_patches, emb_dim)
        B, num_patches, emb_dim =x.shape
        x = x.view(B, 1, num_patches, emb_dim)  # (B, 1, num_patches, emb_dim)

        return x
            

class Predictor2Ds1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel
        self.repr_dim = self.emb_dim * self.num_patches

        self.gru = nn.GRU(self.repr_dim + 2, self.repr_dim, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.Linear(self.repr_dim, self.repr_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.repr_dim, self.repr_dim)
        )

    def forward(self, encodings, actions):
        """
        encodings: (B, T, 1, num_patches, emb_dim)
        actions: (B, T-1, 2)

        returns: (B, T, 1, num_patches, emb_dim)
        """
        B, T, _, num_patches, emb_dim = encodings.shape
        assert T == actions.shape[1] + 1

        inputs = []
        for t in range(T - 1):
            e = encodings[:, t].reshape(B, -1)  # (B, repr_dim)
            a = actions[:, t]                   # (B, 2)
            inputs.append(torch.cat([e, a], dim=-1))  # (B, repr_dim + 2)

        inputs = torch.stack(inputs, dim=1)  # (B, T-1, repr_dim + 2)
        output_seq, _ = self.gru(inputs)     # (B, T-1, repr_dim)
        output_seq = self.out_proj(output_seq)  # (B, T-1, repr_dim)

        output_seq = output_seq.view(B, T - 1, 1, num_patches, emb_dim)
        first_state = encodings[:, :1]
        return torch.cat([first_state, output_seq], dim=1)  # (B, T, 1, num_patches, emb_dim)

    
    

class Regularizer2Ds1(nn.Module):
    def __init__(self, repr_dim, r = 'diff'):
        super().__init__()
        self.repr_dim = repr_dim

        if r == 'diff':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )
        elif r == 'mean':
            self.mlp = nn.Sequential(
                nn.Linear(repr_dim * 2, repr_dim),
                nn.ReLU(),
                nn.Linear(repr_dim, 2)
            )


    def forward(self, enc_states, pred_states, r = 'diff'):
        """
        Args:
            enc_states: [B, T-1, 1, num_patches, repr_dim]
            pred_states: [B, T-1, 1, num_patches, repr_dim]
        Returns:
            predicted_actions: [B*(T-1), 2]
        """

        if r == 'diff':
            # Remove the singleton dim
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            # Compute token-wise difference
            delta = pred_states - enc_states  # [B, T-1, num_patches, repr_dim]

            # Mean over tokens (per frame)
            delta_mean = delta.mean(dim=2)  # [B, T-1, repr_dim]

            # Flatten over batch and time
            B, Tm1, _ = delta_mean.shape
            delta_flat = delta_mean.view(B * Tm1, self.repr_dim)  # [B*(T-1), repr_dim]

            # Predict action
            action_pred = self.mlp(delta_flat)  # [B*(T-1), 2]

            return action_pred
        
        if r == 'mean':
            enc_states = enc_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]
            pred_states = pred_states.squeeze(2)  # [B, T-1, num_patches, repr_dim]

            h = torch.cat((enc_states, pred_states), dim=-1) # [B, T-1, num_patches, 2*repr_dim]
            h = h.mean(dim=2)  # [B, T-1, 2*repr_dim]

            B, Tm1, D = h.shape
            h = h.view(B * Tm1, D)  # [B*(T-1), 2 *repr_dim]

            pred_action = self.mlp(h)  # [B*(T-1), 2]
            return pred_action



class JEPA2Ds1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.teacher_forcing = config.teacher_forcing
        self.num_patches = (65 // config.patch_size) ** 2
        self.emb_dim = config.out_channel

        self.emb_pos = nn.Parameter(torch.randn(1, self.num_patches, self.emb_dim))  # [1, num_patches, emb_dim]

        self.encoder = Encoder2Ds1(config)
        self.predictor = Predictor2Ds1(config)


        self.regularizer = Regularizer2Ds1(self.emb_dim, config.r)

        self.repr_dim = self.emb_dim * self.num_patches  
        self.noise = config.noise


    def forward(self, states, actions):
        """
        Args:
            actions: [B, T-1, 2]
            states: [B, T, 2, 65, 65]

        Output:
            enc_states: [B, T, 1, num_patches, emb_dim], T: (s_0, s'_1, ..., s'_T)
            pred_states: [B, T, 1, num_patches, emb_dim], T: (s_0, tilde{s_1}, ..., tilde{s_T})
        """

        self.actions = actions
        self.states = states
        B, T, C, H, W = states.shape
        
        if self.teacher_forcing and T >1:
            states = states.view(B * T, C, H, W) # (B*T, 2, 65, 65)
            
            enc_states = self.encoder(states, self.emb_pos)  # (B*T, 1, num_patches, emb_dim)
            _, _, emb_h, emb_w = enc_states.shape

            enc_states = enc_states.view(B, T, 1, emb_h, emb_w) # (B, T, 1, num_patches, emb_dim)

            # tensor for predictions
            pred_states = torch.zeros((B, T, 1, emb_h, emb_w), device = enc_states.device)  # (B, T, 1, num_patches, emb_dim)
            pred_states[:, 0] = enc_states[:, 0] # set the first state to the encoded state

            # inputs for predictor
            predictor_states = enc_states[:, :-1, :, :, :].contiguous() # (B, T-1, 1, num_patches, emb_dim)
            predictor_states = predictor_states.view(B * (T - 1), 1, emb_h, emb_w)  # (B*(T-1), 1, num_patches, emb_dim)
            actions = actions.view(B * (T - 1), 1, 2)   # (B*(T-1), 1, 2)

            next_states = self.predictor(predictor_states, actions, self.emb_pos)  # (B*(T-1), 1, num_patches, emb_dim)
            next_states = next_states + self.noise * torch.randn_like(next_states)  # Add noise

            next_states = next_states.view(B, T - 1, 1, emb_h, emb_w)
            pred_states[:, 1:] = next_states   # (B, T-1, 1, num_patches, emb_dim)

            self.enc_states = enc_states
            self.pred_states = pred_states 

            return enc_states, pred_states
        

        elif T == 1:    # for inference
            T = actions.shape[1]

            enc_states = self.encoder(states[:, 0], self.emb_pos)  # (B, 1, num_patches, emb_dim) s_0
            _, _, emb_h, emb_w = enc_states.shape

            h = enc_states  # [B, 1, H, W]
            h = h.unsqueeze(1)  # [B, 1, 1, H, W]
            pred_states = [h]

            for t in range(T):
                action = actions[:, t].unsqueeze(1)  # [B, 1, 2]
                h = self.predictor(h.squeeze(1), action, self.emb_pos)  # -> [B, 1, H, W]
                h = h.unsqueeze(1)  # [B, 1, 1, H, W]
                pred_states.append(h)

            T = T + 1
            pred_states = torch.cat(pred_states, dim=1)  # [B, T, 1, H, W]
            pred_states = pred_states.view(B, T, emb_h * emb_w)  
            return pred_states
        
        else:
            # TODO
            raise NotImplementedError("None Teacher Forcing is not implemented yet.")



            

        
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

        return vicreg_loss(x, y, self.config.sim_coeff, self.config.std_coeff, self.config.cov_coeff, self.config.eps)
    

    def loss_R(self, enc_states, pred_states, actions):
        """
        Args:
            enc_states: [B, T, 1, emb_h, emb_w]
            pred_states: [B, T, 1, emb_h, emb_w]
            actions: [B, T-1, 2]

        Output:
            loss: scalar
        """
        
        # Calculate the action regularization loss
        predicted_actions = self.regularizer(enc_states[:, :-1], pred_states[:, 1:], self.config.r)  # [B(T-1), 2]
        actions = actions.view(-1, 2)
        loss = F.mse_loss(predicted_actions, actions, reduction='mean')
        return loss
    

    def compute_loss(self, print_loss = False):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff

        if print_loss:
            print(f"loss: {loss.item():.4f}, vicreg: {vicreg.item():.4f}, reg: {reg.item() * self.config.reg_coeff:.4f}")
        return loss






# ---------------- Models -------------------
MODELS: dict = {
    # "Model_Name": Model_Class
    "JEPA2D": JEPA2D,       # CNN 2D embedding
    "JEPA2D2": JEPA2D2,     # JEPA2D + CBAM
    "JEPA": JEPA,           # CNN 1D embedding
    "JEPA2Dv1": JEPA2Dv1,   # CNN patch embedding 
    "JEPA2Dv2": JEPA2Dv2,   # Simple patch embedding (ViT-like) + new regularizations
    "JEPA2Dv2B": JEPA2Dv2B, # JEPA2Dv2 + BYOL
    "JEPA2Dv3": JEPA2Dv3,   # JEPA2Dv2 + conv
    "JEPA2Dv4": JEPA2Dv4,   # JEPA2Dv3 + CBAM
    "JEPA2Da1": JEPA2Da1,   # JEPA2Dv2 + self attention in predictor 
    "JEPA2Da2": JEPA2Da2,   # JEPA2Dv2 + self attention in encoder
    "JEPA2Ds1": JEPA2Ds1    # JEPA2Dv2 + GRU predictor
    # add models here
}

