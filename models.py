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

         
    def compute_loss(self):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff 
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
    

    def compute_loss(self):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff
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


        self.regularizer = Regularizer2Dv2(self.emb_dim)

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
    

    def compute_loss(self):
        """
        Compute the loss for the model.
        """
        # Compute the loss
        vicreg = self.loss_vicreg(self.enc_states, self.pred_states)
        reg = self.loss_R(self.enc_states, self.pred_states, self.actions)
        loss = vicreg + reg * self.config.reg_coeff
        return loss




# ---------------- Models -------------------
MODELS: dict = {
    # "Model_Name": Model_Class
    "JEPA2D": JEPA2D,
    "JEPA": JEPA,
    "JEPA2Dv1": JEPA2Dv1,
    "JEPA2Dv2": JEPA2Dv2
    # add models here
}
