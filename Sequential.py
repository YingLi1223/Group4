"""
Main scropt for running RL Agent Training.
Complete the psuedocode here as part of your project implementation. 
Feel free to write more scripts to help manage your codebase.
"""
import numpy as np
import tqdm
from typing import Literal, Tuple
from lightsim2grid.lightSimBackend import LightSimBackend
from jsonargparse import auto_cli
from L2RPN.Agents import DDPG, DDPGParams, TD3, TD3HParams, SAC, SACHParams
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from L2RPN.ReplayBuffer.VanillaReplayBuffer import ReplayBuffer

def run(env_name:str="l2rpn_case14_sandbox", agent:Literal['DDPG','TD3','SAC']="DDPG",
        n_active:int=5000, replay_size:int=10000, rho_threshold:float=0.95, stage:Literal["TRAIN","VALIDATE","TEST"]="TRAIN", 
        batch_size:int=32, seed:int=0, verbose:bool=False) -> Tuple[float]:
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
    # TODO: Figure out how to split into TRAIN, VALIDATE, and TEST Environments
    # We need to do this to prevent data from leaking, the agent should be evaluated on unseen data!
    env = TemplateEnvWrapper(env_name, backend=LightSimBackend(automatically_disconnect=True),
                             rho_threshold=rho_threshold, verbose=verbose)
    n_eps, ep_ids = env.get_env_size()
    # TODO: Figure out what Obervation / Action size is appropriate
    # Hint: You will need to implement this inside TemplateEnvWrapper
    OBS_DIM = None
    ACT_DIM = None
    ACT_LIMIT = None
    # TODO: Agent not converging? Maybe try non-default hyperparameters / hyperparameter optimization
    hyperparameters = {}
    agent_lookup = {"DDPG":(DDPG, DDPGParams),
                    "TD3":(TD3,TD3HParams), 
                    "SAC":(SAC,SACHParams)}
    agent_class, agent_hparams = agent_lookup[agent]
    agent:DDPG|TD3|SAC = agent_class(agent_hparams(obs_dim=OBS_DIM, act_dim=ACT_DIM, act_limit=ACT_LIMIT, 
                                                   **hyperparameters))
    
    # >> Replay Buffer <<
    buffer = ReplayBuffer(max_size=replay_size, obs_dim=OBS_DIM,
                          gamma=agent.gamma, N_steps=agent.n_steps)
    
    ep_no = 0
    total_steps = 0
    ep_rewards = np.zeros()
    while(total_steps < n_active):
        ep_pos = ep_no % n_eps
        ep_id = ep_ids[ep_pos]
        
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_id})
        done = terminated or truncated

        ep_steps, ep_reward = 0, info["reward"]
        while not done:
            action = agent.act(obs_vec)
            next_obs_vec, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if stage.upper() == "TRAIN":
                buffer.add(obs_vec, action, reward, next_obs_vec, terminated) # Only want done in the buffer if episode ended prematurely
                
                # Backpropagate
                if buffer.size > batch_size:
                    # Get a batch of experiences from the replay buffer
                    batch = buffer.sample(batch_size)
                    # Update the parameters (weights & biases) of the agent
                    # based on the batch
                    agent.update(batch)
                    # TODO: Decay epsilon for DDPG?
               
            ep_reward += reward
            ep_steps += 1
            if done: break
            obs_vec = next_obs_vec
        # TODO: Logging per-episode? What is your agent doing?
        ep_rewards[ep_no] = ep_reward
        # TODO: Early-Stopping (if agent has converged, we can stop early)
        
        total_steps += ep_steps
        ep_no += 1
    return np.mean(ep_rewards)

if __name__ == "__main__":
    auto_cli(run)
