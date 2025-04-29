import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def log_embedding_statistics(model, epoch, logdir="logs"):
    os.makedirs(logdir, exist_ok=True)

    with torch.no_grad():
        embeddings = model.enc_states.reshape(model.enc_states.shape[0], model.enc_states.shape[1], -1)  # (B, T, D)
        norms = torch.norm(embeddings, dim=-1).cpu().numpy()  # (B, T)
        norms = norms.flatten()

    avg_norm = np.mean(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    var_norm = np.var(norms)

    stats = np.array([epoch, avg_norm, min_norm, max_norm, var_norm])

    log_file = os.path.join(logdir, "embedding_stats.npy")

    if os.path.exists(log_file):
        old_stats = np.load(log_file)
        new_stats = np.vstack([old_stats, stats])
    else:
        new_stats = stats[None]

    np.save(log_file, new_stats)





import numpy as np
import matplotlib.pyplot as plt

def detect_collapse(log_path="logs/embedding_stats.npy", threshold=1e-3):
    log = np.load(log_path)  # [epochs, 5]

    epochs = log[:,0]
    avg_norm = log[:,1]
    min_norm = log[:,2]
    max_norm = log[:,3]
    var_norm = log[:,4]

    collapsed_epochs = np.where(var_norm < threshold)[0]

    if len(collapsed_epochs) > 0:
        print(f"❌ Collapse detected at epochs: {collapsed_epochs}")
    else:
        print(f"✅ No collapse detected. Model looks good.")

    # Plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, avg_norm, label="Avg Norm")
    plt.plot(epochs, min_norm, label="Min Norm")
    plt.plot(epochs, max_norm, label="Max Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Value")
    plt.title("Embedding Norms over Training")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, var_norm, label="Variance")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Collapse Threshold")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Variance")
    plt.title("Embedding Norm Variance")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    detect_collapse()
