import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from lightsim2grid.lightSimBackend import LightSimBackend
from L2RPN.Agents import DDPG, DDPGParams, TD3, TD3HParams
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from grid2op.Reward import GameplayReward
import torch.nn as nn
import torch.optim as optim

def evaluate_single(agent_name: str = "TD3", env_name: str = "l2rpn_case14_sandbox", max_episodes: int = 1004):
    env = TemplateEnvWrapper(env_name, backend=LightSimBackend(automatically_disconnect=True),
                             env_kwargs={'reward_class': GameplayReward},
                             rho_threshold=0.95, verbose=False)
    n_eps, ep_ids = env.get_env_size()
    OBS_DIM = len(env.reset()[0])
    ACT_DIM = 6
    ACT_LIMIT = 1.0

    if agent_name == "DDPG":
        agent_params = DDPGParams(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            pi_hidden_sizes=[400, 400],
            q_hidden_sizes=[400, 400],
            activation_function=nn.ReLU,
            act_limit=ACT_LIMIT,
            optim_pi=optim.Adam,
            optim_q=optim.Adam,
            lr_pi=3e-4,
            lr_q=3e-4,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            noise_scale=0.0,
            n_steps=1
        )
        agent = DDPG(agent_params)
    elif agent_name == "TD3":
        agent_params = TD3HParams(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            act_limit=ACT_LIMIT,
             pi_hidden=[128, 128],
            q_hidden=[128, 128],
            activation=nn.ReLU,
            optim_pi=optim.Adam,
            optim_q=optim.Adam,
            lr_pi=3e-4,
            lr_q=3e-4,
            device="cuda",
            gamma=0.99,
            tau=0.003,
            act_noise=0.01,
            target_noise=0.1,
            policy_delay=2,
            noise_clip=0.3,
            n_steps=1
        )
        agent = TD3(agent_params)
    else:
        raise ValueError("Unsupported agent name.")

    ckpt_path = f"checkpoints/{agent_name.lower()}_latest.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    agent.ac.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    print(f"✅ Loaded {agent_name} checkpoint from {ckpt_path}")

    results = {"episode": ep_ids[:max_episodes], "baseline": [], agent_name: []}

    for ep_id in ep_ids[:max_episodes]:
        # Baseline
        env.env.reset(options={"time serie id": ep_id})
        done, total_reward = False, 0
        while not done:
            action = env.env.action_space({})
            _, reward, done, _ = env.env.step(action)
            total_reward += reward
        results["baseline"].append(total_reward)

        # Agent
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        done, total_reward = terminated or truncated, 0
        while not done:
            action = agent.act(obs_vec)
            obs_vec, reward, terminated, truncated, info, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results[agent_name].append(total_reward)

        print(f"Episode {ep_id} | {agent_name}: {results[agent_name][-1]:.2f} | Baseline: {results['baseline'][-1]:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(f"compare_{agent_name.lower()}_vs_baseline.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df["episode"], df["baseline"], label="Baseline", linestyle="--")
    plt.plot(df["episode"], df[agent_name], label=agent_name, marker='o')
    plt.title(f"{agent_name} vs Baseline Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"compare_{agent_name.lower()}_vs_baseline.png")
    plt.show()

if __name__ == "__main__":
    evaluate_single(agent_name="TD3")  # 替换为 "TD3" 以评估 TD3
