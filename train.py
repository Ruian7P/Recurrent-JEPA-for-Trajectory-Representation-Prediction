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



def train(config: ModelConfig, resume: bool = False):

    accelerator = Accelerator()
    device = accelerator.device
    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs

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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        accelerator.print(f"Model loaded from {MODEL_PATH}.")
    else:
        accelerator.print("Training from scratch.")


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    for epoch in range(epochs):  # adjust as needed
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            states, actions = batch.states, batch.actions  # (B, T, C, H, W), (B, T-1, 2)
            optimizer.zero_grad()

            enc, pred = model(states, actions)  # Forward pass

            # Loss
            loss = model.compute_loss()
            accelerator.backward(loss)  # Backward pass
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()  
        accelerator.print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

        # Save model checkpoint
        if accelerator.is_main_process:
            if epoch % 5 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{epoch+1}.pth")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.parse_from_file(CONFIG_PATH)
    train(config, resume=args.resume)
