import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import MODELS
from configs import ModelConfig, PATH, CONFIG_PATH, MODEL_PATH
from tqdm import tqdm
import os
from accelerate import Accelerator
import math
from emb_visual import log_embedding_statistics



def warmup_cosine_lr(epoch, total_epochs, warmup_epochs=10):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs # Linear warmup
    else:
        return 0.5 * (math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi) + 1)
    
def get_momentum(epoch, max_epoch):
    base_m = 0.99
    final_m = 0.996
    return final_m - (final_m - base_m) * (1 + math.cos(math.pi * epoch / max_epoch)) / 2


def train(config_path: str, model_path: str, resume: bool = False, log: bool = False):
    config = ModelConfig.parse_from_file(config_path)

    accelerator = Accelerator()
    device = accelerator.device
    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs

    if epochs == 15:
        warmup_epochs = 2
    elif epochs == 20:
        warmup_epochs = 3
    elif epochs == 30:
        warmup_epochs = 5
    elif epochs == 40:
        warmup_epochs = 7
    else:
        warmup_epochs = 10

    # print config
    print("Config:")
    print(config)

    # Dataset
    data_path = f"{PATH}/train"
    train_loader = create_wall_dataloader(data_path, probing=False, device=device, batch_size= batch_size, train=True)

    # Model
    model_name = config.Model
    model = MODELS[model_name]
    model = model(config).to(device)

    if resume:
        model.load_state_dict(torch.load(model_path, map_location=device))
        accelerator.print(f"Model loaded from {MODEL_PATH}.")
    else:
        accelerator.print("Training from scratch.")


    # delete log dir
    if log:
        logdir = "./logs"
        if os.path.exists(logdir):
            os.system(f"rm -rf {logdir}")
        os.makedirs(logdir, exist_ok=True)
        accelerator.print(f"Log directory: {logdir}")

    
    # check whether weight_decay exists in config
    if hasattr(config, "weight_decay"):
        weight_decay = config.weight_decay
    else:
        weight_decay = 0


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    # Scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs
    # )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_lr(epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # only for BYOL
    total_steps = len(train_loader) * epochs
    global_step = 0

    # Training loop
    for epoch in range(epochs):  # adjust as needed
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            states, actions = batch.states, batch.actions  # (B, T, C, H, W), (B, T-1, 2)
            optimizer.zero_grad()

            enc, pred = model(states, actions)  # Forward pass

            # Loss
            if idx == len(train_loader) - 1:
                loss = model.compute_loss(print_loss=True)
            else:
                loss = model.compute_loss()
            accelerator.backward(loss)  # Backward pass
            optimizer.step()

            epoch_loss += loss.item()

            if model_name == "JEPA2Dv2B":
                momentum = get_momentum(global_step, total_steps)
                model.update_target_encoder(momentum)

            global_step += 1

        scheduler.step()  
        accelerator.print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

        # Save model checkpoint
        if accelerator.is_main_process:
            if epoch % 1 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{epoch+1}.pth")


        # Log embedding statistics
        if log and accelerator.is_main_process:
            log_embedding_statistics(model, epoch, logdir=logdir)


    # save final model
    if accelerator.is_main_process:
        os.makedirs("models", exist_ok=True)
        model_name = CONFIG_PATH.split("/")[-1].split(".")[0]
        torch.save(model.state_dict(), f"models/{model_name}.pth")
        accelerator.print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", help= "Resume training from saved model"
    )
    parser.add_argument(
        "--config", type=str, default=CONFIG_PATH, help="Path to the config file"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to the model file"
    )
    parser.add_argument(
        "--log", action="store_true", help="Log embedding statistics"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.model, resume=args.resume, log=args.log)
