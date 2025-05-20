from lightsim2grid.lightSimBackend import LightSimBackend
from L2RPN.EnvWrappers.TemplateWrapper import TemplateEnvWrapper
import numpy as np

def test_wrapper():
    print("🟡 Starting test_wrapper()")

    env_kwargs = {}
    env = TemplateEnvWrapper(
        env_name="l2rpn_case14_sandbox",
        backend=LightSimBackend(),
        env_kwargs=env_kwargs,
        rho_threshold=0.95,
        verbose=True
    )

    


    n_eps, ep_ids = env.get_env_size()
    print(f"🟢 Environment has {n_eps} episodes")

    try:
        obs_vec, info, terminated, truncated = env.reset(options={"time serie id": ep_ids[0]})
        
        obs = env.tracker.state
        print("🔍 Generator voltage sample (prod_v):", obs.prod_v[:3])
        print("🔍 Line voltage OR sample (v_or):", obs.v_or[:3])

    except Exception as e:
        print("Exception during reset:", e)
        return

    
    n_gen = env.env.n_gen
    action_dim = 2 * n_gen  # redispatch + curtailment
    random_action = np.random.uniform(-1, 1, action_dim)



    try:
        action = env.process_agent_action(random_action)
        next_obs, reward, terminated, truncated, info = env.step(random_action)
        print("✅ Step successful!")
        print(f"Next obs shape: {next_obs.shape}, Reward: {reward}")
    except Exception as e:
        print("❌ Exception during step:", e)

if __name__ == "__main__":
    test_wrapper()
