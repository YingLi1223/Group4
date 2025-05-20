import math
import torch
import numpy as np
import torch.nn as nn, torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from L2RPN.ReplayBuffer.VanillaReplayBuffer import ReplayBuffer
from L2RPN.Agents.DDPG import DDPG, DDPGParams # Score of -0.44 in ~470 episodes
from L2RPN.Agents.TD3 import TD3, TD3HParams
from L2RPN.Agents.SAC import SAC, SACHParams # Not really converging in tests...

from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from lightsim2grid.lightSimBackend import LightSimBackend



AGENT = "TD3" # DDPG #"SAC"
# OBS_DIM = 3
# ACT_DIM = 1
# ACT_LIMIT = 2

USE_PER = False
PER_ALPHA = 0.25
BETA = 0.6 # Parameter for importance sampling
BETA_DECAY = 1e-3
LEARNING_DELAY = 40
ENTROPY_ALPHA = 0.05
LOG_STD_MIN = -20
LOG_STD_MAX = 2
POLICY_DELAY = 1
BATCH_SIZE = 128
BUFFER_LEN = 200000
GAMMA = 0.98

# HIDDEN_PI = [400,400]
# HIDDEN_Q = [400, 400]
# LR_PI = 1e-5
# LR_Q = 1e-4
# TAU = 0.001
# NOISE_SCALE = 0.1
# TARGET_NOISE = 0.2
# NOISE_CLIP = 0.5
TARGET_NOISE = 0.1
NOISE_CLIP = 0.3
HIDDEN_PI = [400, 300]
HIDDEN_Q = [400, 300]
LR_PI = 1e-4
LR_Q = 3e-4
TAU = 0.005

NOISE_SCALE = 0.2           # 初始 exploration 强度

N_GAMES = 3000
START_STEPS = 10000
NOISE_DEC_FACTOR = 0.9997
NOISE_MIN = 0.05
# ENV_NAME = "Pendulum-v1"#"LunarLander-v2" # "MountainCarContinuous-v0"#
ENV_NAME = "LunarLander-v3" # "MountainCarContinuous-v0"#
ENV_KWARGS = {"continuous": True} if "LunarLander" in ENV_NAME else {}

N = 10
M = 2
G = 20
POLICY_LOSS_MODE = "min"
AUTO_ENTROPY = True
N_STEPS = 1
VALIDATE_EVERY = 25





if __name__ == "__main__":
    env = gym.make(ENV_NAME, **ENV_KWARGS)
    
    # 自动设置 obs_dim, act_dim, act_limit
    OBS_DIM = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        ACT_DIM = env.action_space.shape[0]
        ACT_LIMIT = float(env.action_space.high[0])  # 假设动作是对称的
    else:
        raise ValueError("当前环境使用的是离散动作空间，不适用于 TD3/DDPG/SAC 等算法")

    if AGENT == "DDPG":
        agent_params = DDPGParams(
            OBS_DIM, ACT_DIM, HIDDEN_PI, HIDDEN_Q, nn.ReLU, ACT_LIMIT, optim.AdamW, optim.AdamW,
            LR_PI, LR_Q, "cuda", GAMMA, TAU, NOISE_SCALE, 3
        )
        agent = DDPG(agent_params)
    elif AGENT == "TD3":
        agent_params = TD3HParams(
            OBS_DIM, ACT_DIM, ACT_LIMIT, HIDDEN_PI, HIDDEN_Q, nn.ReLU, optim.AdamW, optim.AdamW,
            LR_PI, LR_Q, "cuda", GAMMA, TAU, NOISE_SCALE, TARGET_NOISE, POLICY_DELAY, NOISE_CLIP,
            3
        )
        agent = TD3(agent_params)
        agent.act_noise = NOISE_SCALE  
    elif AGENT == "SAC":
        agent_params = SACHParams(
            GAMMA, TAU, LR_PI, LR_Q, ENTROPY_ALPHA, OBS_DIM, ACT_DIM, ACT_LIMIT,
            HIDDEN_PI, HIDDEN_Q, nn.ReLU, LOG_STD_MIN, LOG_STD_MAX, optim.AdamW, optim.AdamW,
            N_STEPS, AUTO_ENTROPY, "cuda"
        )
        agent = SAC(agent_params)
    else:
        raise ValueError(f"Invalid agent {AGENT}")
    scores, ma100, mem_beta, val_scores, ma_val = [], [], [], [], []
    pi_losses, q_losses = [], []
    max_score = -math.inf
    

    buffer = ReplayBuffer(BUFFER_LEN, OBS_DIM, GAMMA, N_STEPS)
    tot_steps = []
    val_scores = [-math.inf]
    for i in range(N_GAMES):
        score = 0
        steps = 0
        done = False
        obs, _ = env.reset()

        while not done:
            steps += 1
            if sum(tot_steps) < START_STEPS:
                action = env.action_space.sample()
            else:
                # action = agent.act(obs)
                if AGENT == "SAC":
                    action = agent.act(obs)
                else:
                    action = agent.act(obs)
                    agent.act_noise = max(agent.act_noise * NOISE_DEC_FACTOR, NOISE_MIN)



            next_obs, reward, terminated, truncated, info = env.step(action)

            score += reward

            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            if len(buffer) >= START_STEPS:
                batch = buffer.sample(BATCH_SIZE)
                pi_loss, q_loss = agent.update(batch)
    
                pi_losses.append(pi_loss)
                q_losses.append(q_loss)
        
        scores.append(score)
        if score > max_score: 
            max_score = score
        tot_steps.append(steps)
        if not i % VALIDATE_EVERY and len(buffer) >= START_STEPS:
            for _ in range(5):
                obs, _ = env.reset()
                val_score = 0
                done = False
                while not done:
                    action = agent.act(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    val_score += reward
                    obs = next_obs
                    done = terminated or truncated
                val_scores.append(val_score)

        best_val_score = max(val_scores)
        avg_score = np.mean(scores[-50:])
        print(f"""Episode: {i} :: Score: {round(score, 2)} :: Max score: {round(max_score, 2)} :: Average score: {round(avg_score, 2)} :: Best Val score: {round(best_val_score, 2)} :: Latest Val score: {round(val_scores[-1], 2)} :: Ep. Steps: {steps} :: Tot. Steps: {sum(tot_steps)} :: Act noise: {round(agent.act_noise, 3)}""")
        # print(f"""Episode: {i} :: Score: {round(score, 2)} :: Max score: {round(max_score, 2)} :: Average score: {round(avg_score, 2)} :: Best Val score: {round(best_val_score, 2)}::Latest Val score: {round(val_scores[-1], 2)} Ep. Steps: {steps} :: Tot. Steps: {sum(tot_steps)}""")
        # :: Act noise: {round(agent.noise_scale, 3)}""")
        # print(f"Average policy loss: {np.mean(pi_losses[-100:])} :: Average critic loss: {np.mean([q_losses[-100:]])}")

        def smooth_curve(data, window_size=20):
            if len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    episode_rewards = scores
    smoothed_rewards = smooth_curve(episode_rewards)

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label="TD3 episode rewards", alpha=0.4)
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TD3 on LunarLander-v3")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(smooth_curve(pi_losses), label="Policy Loss")
    plt.plot(smooth_curve(q_losses), label="Q Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()
    plt.show()



    


