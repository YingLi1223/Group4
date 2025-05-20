import sys
import numpy as np
import torch
from lightsim2grid.lightSimBackend import LightSimBackend
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
from L2RPN.Agents import SAC, SACHParams

# Redirect all prints to a log file
log_file = open('test_sac_debug.log', 'w')
sys.stdout = log_file

def test_sac_steps(env_name: str = "l2rpn_case14_sandbox", n_steps: int = 100, device: str = "cpu"):
    """
    Run SAC for a fixed number of steps, writing debug info to stdout (redirected to log):
      - raw action vector
      - clipped redispatch and curtailment controls
      - computed MW values for redispatch and curtailment
      - resulting state (rho, prod_p, load_p, load_q)
    """
    # Initialize environment wrapper
    env = TemplateEnvWrapper(
        env_name,
        backend=LightSimBackend(automatically_disconnect=True),
        env_kwargs={},
        rho_threshold=0.95,
        verbose=False
    )
    # Warm reset to fetch dims
    obs_vec, info, terminated, truncated = env.reset()
    OBS_DIM = len(obs_vec)
    ACT_DIM = env.env.n_gen  # now equal to number of generators

    # Configure SAC hyperparameters (match those in Sequential.py)
    params = SACHParams(
        gamma=0.99,
        tau=0.01,
        lr_pi=1e-4,
        lr_q=5e-5,
        alpha=0.2,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        act_limit=1.0,
        hidden_pi=[128, 128],
        hidden_q=[128, 128],
        activation=torch.nn.ReLU,
        log_std_min=-20,
        log_std_max=2,
        optim_pi=torch.optim.AdamW,
        optim_q=torch.optim.AdamW,
        n_steps=3,
        auto_entropy=True,
        device=device
    )
    agent = SAC(params)

    # Begin debug loop
    obs_vec, info, terminated, truncated = env.reset()
    for step in range(n_steps):
        action = agent.act(obs_vec)
        n_gen = env.env.n_gen

        # 1) Clip raw into two masks
        red_ratio = np.clip(action, -1, 1)
        ct_ratio = np.clip(action, 0, 1)

        # 2) Determine renewable units
        is_renew = np.array([str(t).lower() in ("solar", "wind") for t in env.gen_type])
        red_ratio[is_renew] = 0.0
        ct_ratio[~is_renew] = 0.0

        # 3) Compute MW values
        rd_mw = np.zeros(n_gen, dtype=np.float32)
        ct_mw = np.zeros(n_gen, dtype=np.float32)
        # conventional units: redispatch
        rd_mw[~is_renew] = red_ratio[~is_renew] * np.where(
            red_ratio[~is_renew] >= 0,
            env.gen_max_ramp_up[~is_renew],
            env.gen_max_ramp_down[~is_renew]
        )
        # renewables: curtailment proportion directly
        ct_mw[is_renew] = ct_ratio[is_renew]

        # Write debug info
        print(f"\n--- Step {step} ---")
        print("Raw action vector:", action)
        print("Redispatch ratio:", red_ratio)
        print("Curtailment ratio:", ct_ratio)
        print("Redispatch MW:", rd_mw)
        print("Curtailment MW:", ct_mw)

        # Step environment
        obs_vec, reward, terminated, truncated, info = env.step(action)
        st = env.tracker.state

        print("rho:", st.rho)
        print("prod_p:", st.prod_p)
        print("load_p:", st.load_p)
        print("load_q:", st.load_q)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

if __name__ == "__main__":
    test_sac_steps(n_steps=100)
    log_file.close()
