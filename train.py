import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import MODELS
from configs import ModelConfig, PATH, CONFIG_PATH
from tqdm import tqdm
import os



def train(config: ModelConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs

    # Dataset
    data_path = f"{PATH}/train"
    train_loader = create_wall_dataloader(data_path, probing=False, device=device, batch_size= batch_size, train=True)

    # Model
    model_name = config.Model
    model = MODELS[model_name]
    model = model(config).to(device)


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    for epoch in range(epochs):  # adjust as needed
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            states, actions = batch.states, batch.actions  # (B, T, C, H, W), (B, T-1, 2)
            optimizer.zero_grad()

            enc, pred = model(states, actions)  # Forward pass

            # Loss
            loss = model.loss_mse(enc, pred)  
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()  
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")

        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{epoch}.pt")


    # save final model
    os.makedirs("models", exist_ok=True)
    model_name = CONFIG_PATH.split("/")[-1].split(".")[0]
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    print("Training complete!")


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config", type=str, required=True, help="Path to config YAML"
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     config = ModelConfig.parse_from_file(args.config)
#     train(config)

if __name__ == "__main__":
    config = ModelConfig.parse_from_file(CONFIG_PATH)
    train(config)
