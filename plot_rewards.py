import pandas as pd
import matplotlib.pyplot as plt
import os

# agents = ["ddpg", "td3", "sac"]
agents = ["td3"]

window = 10  # Smooth window size

metrics = {
    "reward": {"ylabel": "Reward", "title": "Episode Reward"},
    "rho_max": {"ylabel": "Max Rho", "title": "Max Line Loading per Episode"},
    "rho_mean": {"ylabel": "Mean Rho", "title": "Mean Line Loading per Episode"}
}

for agent in agents:
    filepath = f"metrics_log_{agent}.csv"
    if not os.path.exists(filepath):
        print(f" Log file for {agent} not found: {filepath}")
        continue

    df = pd.read_csv(filepath)

    for metric in metrics:
        if metric not in df.columns:
            print(f"  Metric '{metric}' not found in {filepath}")
            continue

        data = df[metric]
        smoothed = data.rolling(window=window).mean()

        plt.figure()
        plt.plot(data, label="Raw", linestyle="--", alpha=0.6)
        plt.plot(smoothed, label="Smoothed", linewidth=2)

        plt.title(f"{metrics[metric]['title']} ({agent.upper()})")
        plt.xlabel("Episode")
        plt.ylabel(metrics[metric]['ylabel'])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{agent}_{metric}.png")
        print(f"✅ Saved plot: {agent}_{metric}.png")
        plt.close()

# === Newly plotted loss curves ===
loss_metrics = {
    "q_loss": {"ylabel": "Q Loss", "title": "Critic Loss"},
    "pi_loss": {"ylabel": "Policy Loss", "title": "Actor Loss"}
}

for agent in agents:
    loss_path = f"loss_log_{agent}.csv"
    if not os.path.exists(loss_path):
        print(f"⚠️  Loss log not found for {agent}: {loss_path}")
        continue

    df = pd.read_csv(loss_path)

    for metric in loss_metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not in {loss_path}")
            continue

        data = df[metric]
        smoothed = data.rolling(window=window).mean()

        plt.figure()
        plt.plot(data, label="Raw", linestyle="--", alpha=0.6)
        plt.plot(smoothed, label="Smoothed", linewidth=2)
        plt.title(f"{loss_metrics[metric]['title']} ({agent.upper()})")
        plt.xlabel("Step")
        plt.ylabel(loss_metrics[metric]['ylabel'])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{agent}_{metric}.png")
        print(f" Saved plot: {agent}_{metric}.png")
        plt.close()
