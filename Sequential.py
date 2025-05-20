"""
Main scropt for running RL Agent Training.
Complete the psuedocode here as part of your project implementation. 
Feel free to write more scripts to help manage your codebase.
"""
import numpy as np
import tqdm
import torch
import os
import pandas as pd
from typing import Literal, Tuple
from lightsim2grid.lightSimBackend import LightSimBackend
from jsonargparse import auto_cli
from L2RPN.Agents import DDPG, DDPGParams, TD3, TD3HParams, SAC, SACHParams
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from L2RPN.ReplayBuffer.VanillaReplayBuffer import ReplayBuffer
import torch.nn as nn
import torch.optim as optim  
from grid2op.Reward import GameplayReward
from grid2op.Reward import L2RPNReward
import pickle
# noise decay parameter (DDPG)
START_STEPS = 000
NOISE_MIN = 0.01
NOISE_DEC_FACTOR = 0.9999
ep_rhos = []
pi_losses = []
q_losses = []


def run(env_name:str="l2rpn_case14_sandbox", agent_name:Literal['DDPG','TD3','SAC']="TD3",
        n_active:int=100000, replay_size:int=200000, rho_threshold:float=0.95, stage:Literal["TRAIN","VALIDATE","TEST"]="TRAIN", 
        batch_size:int=128, seed:int=0, verbose:bool=False) -> Tuple[float]:
    """
    Run one Reinforcement Learning (RL) loop. 
    Args:
        env_name (str, optional): Name of Grid2Op Environment to usee, will be downloaded automatically (if it exists). Defaults to "l2rpn_case14_sandbox".
        agent (str, Literal['DDPG','TD3','SAC'], optional): Name of agent to use. Defaults to "DDPG".
        n_active (int, optional): Number of active steps to run for. Defaults to 5000.
        replay_size (int, optional): How big the replay buffer is. Defaults to 10000.
        rho_threshold (float, optional): Fraction of line thermal limit above which agent will be activated. Defaults to 0.95.
        stage (Literal['TRAIN','VALIDATE','TEST'], optional): Which stage we are in. Defaults to "TRAIN".
        batch_size (int, optional). Number of experiences in each batch. Defaults to 32.
        seed (int, optional): Seed for reproducibility. Defaults to 0.
        verbose (bool, optional): Whether to print out extra information. Defaults to False.

    Returns:
        Tuple[float]: _description_
    """
    reward_gt_100_count = 0
    max_step_reward = float("-inf") 

    # TODO: Figure out how to split into TRAIN, VALIDATE, and TEST Environments
    # We need to do this to prevent data from leaking, the agent should be evaluated on unseen data!
    env = TemplateEnvWrapper(env_name, backend=LightSimBackend(automatically_disconnect=True),env_kwargs={'reward_class':GameplayReward},
                             rho_threshold=rho_threshold, verbose=verbose, stage=stage)
    n_eps, ep_ids = env.get_env_size()
    
    obs = env.tracker.state

    # TODO: Figure out what Obervation / Action size is appropriate
    # Hint: You will need to implement this inside TemplateEnvWrapper
    obs_vec, *_ = env.reset()
    OBS_DIM = len(obs_vec)
    ACT_DIM = env.env.n_gen  # redispatch + curtail
    ACT_LIMIT = 1.0  # 
    
    # TODO: Agent not converging? Maybe try non-default hyperparameters / hyperparameter optimization
    hyperparameters = {}
    agent_lookup = {"DDPG":(DDPG, DDPGParams),
                    "TD3":(TD3,TD3HParams), 
                    "SAC":(SAC,SACHParams)}
    agent_class, agent_hparams = agent_lookup[agent_name]

    if agent_name == "DDPG":
        agent_params = agent_hparams(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            pi_hidden_sizes=[128, 128],
            q_hidden_sizes=[128, 128],
            activation_function=nn.ReLU,
            act_limit=ACT_LIMIT,
            optim_pi=optim.Adam,
            optim_q=optim.Adam,
            lr_pi=1e-4,
            lr_q=2e-4,
            device="cuda",
            gamma=0.99,
            tau=0.004,
            noise_scale=1.0,
            n_steps=1
        )
    elif agent_name == "TD3":
        agent_params = agent_hparams(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            act_limit=ACT_LIMIT,
            pi_hidden=[128, 128],
            q_hidden=[128, 128],
            activation=nn.ReLU,
            optim_pi=optim.Adam,
            optim_q=optim.Adam,
            lr_pi=1e-4,
            lr_q=2e-4,
            device="cuda",
            gamma=0.99,
            tau=0.003,
            act_noise=0.7,
            target_noise=0.1,
            policy_delay=2,
            noise_clip=0.3,
            n_steps=1
        )
    elif agent_name == "SAC": # not tested yet
        agent_params = agent_hparams(
            gamma=0.9,
            tau=0.005,
            lr_pi=3e-4,
            lr_q=3e-4,
            alpha=1.0, 
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            act_limit=ACT_LIMIT,
            hidden_pi = [256, 256],
            hidden_q = [256, 256],
            activation=nn.ReLU,
            log_std_min=-20,
            log_std_max=2,
            optim_pi=optim.AdamW,
            optim_q=optim.AdamW,
            n_steps=1,
            auto_entropy=True,  
            device="cuda"
        )


    agent = agent_class(agent_params)
    latest_ckpt_path = f"checkpoints/{agent_name.lower()}_latest.pt"
    if os.path.exists(latest_ckpt_path):
        print(f" Loading previous checkpoint from {latest_ckpt_path}")
        agent.ac.load_state_dict(torch.load(latest_ckpt_path))

    
    # >> Replay Buffer <<
    buffer_path = f"checkpoints/{agent_name.lower()}_buffer.pkl"
    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            buffer = pickle.load(f)
        print(f"✅ Loaded replay buffer from {buffer_path}")
    else:
        buffer = ReplayBuffer(max_size=replay_size, obs_dim=OBS_DIM, gamma=agent.gamma, N_steps=agent.n_steps)

    
    ep_no = 0
    total_steps = 0
    ep_rewards = []
    window = 50
    threshold = 200000

    while(total_steps < n_active):
        ep_pos = ep_no % n_eps
        ep_id = ep_ids[ep_pos]
        
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        done = terminated or truncated

        ep_steps, ep_reward = 0, info["reward"]
        rho_history = []  # rho records for each episode

        while not done:
            if total_steps < START_STEPS:
                action = np.random.uniform(low=-1.0, high=1.0, size=ACT_DIM).astype(np.float32)
            else:
                action = agent.act(obs_vec)
                if isinstance(agent, DDPG):
                    old_noise = agent.noise_scale
                    agent.noise_scale = max(agent.noise_scale * NOISE_DEC_FACTOR, NOISE_MIN)
                    if verbose:
                        print(f"Noise decay: {old_noise:.4f} → {agent.noise_scale:.4f}")
                

                elif isinstance(agent, TD3):
                    old_noise = agent.act_noise
                    agent.act_noise = max(agent.act_noise * NOISE_DEC_FACTOR, NOISE_MIN)
                    if verbose:
                        print(f"TD3 Noise decay: {old_noise:.4f} → {agent.act_noise:.4f}")


            next_obs_vec, reward, terminated, truncated, info, action_processed = env.step(action)

            if reward > 1000:
                reward_gt_100_count += 1
            max_step_reward = max(max_step_reward, reward) # added


            rho_history.append(env.tracker.state.rho.copy())  # Record the current moment of rho
            done = terminated or truncated
            
            if stage.upper() == "TRAIN":
                # buffer.add(obs_vec, action, reward, next_obs_vec, terminated) # Only want done in the buffer if episode ended prematurely
                buffer.add(obs_vec, action_processed, reward, next_obs_vec, terminated)
                
                # Backpropagate
                if  buffer.size > batch_size:
                    # Get a batch of experiences from the replay buffer
                    batch = buffer.sample(batch_size)
                    # Update the parameters (weights & biases) of the agent
                    # based on the batch
                    # agent.update(batch)
                    pi_loss, q_loss = agent.update(batch)
                    if pi_loss is not None:  # Some agents (e.g. TD3) have policies that are not updated every step.
                        pi_losses.append(pi_loss)
                    q_losses.append(q_loss)

                    # TODO: Decay epsilon for DDPG? 
               
            ep_reward += reward
            ep_steps += 1
            if done: break
            obs_vec = next_obs_vec
        # TODO: Logging per-episode? What is your agent doing?
        ep_rewards.append(ep_reward)
        # 🆕 Replace with max/average rho for the entire episode.
        if rho_history:
            all_rho = np.stack(rho_history)
            rho_max_ep = np.max(all_rho)
            rho_mean_ep = np.mean(all_rho)
        else:
            rho_max_ep = 0.0
            rho_mean_ep = 0.0

        # Determining whether a termination is unlawful
        grid_failed = info.get("is_illegal", False) or bool(info.get("exception"))
        success_flag = 0 if grid_failed else 1

        # Print the reason for termination
        if grid_failed:
            print(f" Episode {ep_no} ended due to grid failure.")
        else:
            print(f" Episode {ep_no} ended successfully.")

        ep_rhos.append((ep_no, ep_steps, ep_reward, rho_max_ep, rho_mean_ep, success_flag))


        os.makedirs("checkpoints", exist_ok=True)

        if ep_no % 100 == 0:
            torch.save(agent.ac.state_dict(), f"checkpoints/{agent_name.lower()}_ep{ep_no}.pt")

        torch.save(agent.ac.state_dict(), f"checkpoints/{agent_name.lower()}_latest.pt")
        with open(buffer_path, "wb") as f:
            pickle.dump(buffer, f)


        # TODO: Early-Stopping (if agent has converged, we can stop early)

        if len(ep_rewards) >= window:
            avg_recent = np.mean(ep_rewards[-window:])
            std_recent = np.std(ep_rewards[-window:])
            if avg_recent > threshold and std_recent < 100000:
                print(f"🟢 Early stopping triggered! Avg: {avg_recent:.2f}, Std: {std_recent:.2f}")
                break

        # for DDPG
        # print(f"Episode {ep_no} | Steps: {ep_steps} | Reward: {ep_reward:.2f} | noise: {agent.noise_scale:.4f}")
        # for TD3
        print(f"Episode {ep_no} | Steps: {ep_steps} | Reward: {ep_reward:.2f} | noise: {agent.act_noise:.4f}")

        total_steps += ep_steps
        ep_no += 1

    print("=== Training finished ===")

    print(f" Step reward > 1000 count: {reward_gt_100_count}")
    print(f" Max single-step reward: {max_step_reward:.2f}")

    logfile = f"metrics_log_{agent_name.lower()}.csv"
    columns = ["episode", "steps", "reward", "rho_max", "rho_mean", "success"]


    # If the old log exists, splice it
    if os.path.exists(logfile):
        prev = pd.read_csv(logfile)
        df = pd.concat([prev, pd.DataFrame(ep_rhos, columns=columns)], ignore_index=True)
    else:
        df = pd.DataFrame(ep_rhos, columns=columns)

    df.to_csv(logfile, index=False)
    print(f" Saved metrics log to {logfile}")

    # Save loss to CSV (with stitching of old logs)
    loss_logfile = f"loss_log_{agent_name.lower()}.csv"

    loss_new = pd.DataFrame({
        "step": list(range(len(q_losses))),
        "q_loss": q_losses,
        "pi_loss": [pi_losses[i] if i < len(pi_losses) else np.nan for i in range(len(q_losses))]
    })

    if os.path.exists(loss_logfile):
        prev_loss = pd.read_csv(loss_logfile)
        loss_df = pd.concat([prev_loss, loss_new], ignore_index=True)
    else:
        loss_df = loss_new

    loss_df.to_csv(loss_logfile, index=False)
    print(f" Saved loss log to {loss_logfile}")


    return np.mean(ep_rewards)

        
if __name__ == "__main__":
    auto_cli(run)