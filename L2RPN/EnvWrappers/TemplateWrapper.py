import json, time
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from io import StringIO

import torch
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import RESET_OPTIONS_TYPING

from L2RPN.Abstract import BaseEnvWrapper
from .Utilities import VanillaTracker


def load_df(df_json):
    return pd.read_json(StringIO(df_json["_object"]), orient=df_json["orient"])


class TemplateEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env_name: str,
        backend: Backend,
        env_kwargs: dict,
        *args,
        rho_threshold: float = 0.95,
        verbose: bool = False,
        stage:str = "TRAIN",
        **kwargs
    ):
        if not isinstance(backend, Backend):
            try:
                backend = backend()
            except TypeError:
                print("A backend instance was not provided and initialization failed. Please provide a valid backend or backend class")
                return
        super().__init__(env_name=env_name, backend=backend, grid2op_params=env_kwargs)
        self.rho_threshold = rho_threshold
        self.tracker: VanillaTracker = VanillaTracker()
        self.verbose = verbose
        self.stage = stage.upper()

        # Reading generator characteristics
        prod_charac_path = Path("C:/Users/10452/data_grid2op/l2rpn_case14_sandbox/prods_charac.csv")
        prod_df = pd.read_csv(prod_charac_path)

        self.gen_max_prod = prod_df["Pmax"].to_numpy(dtype=np.float32)

        self.gen_min_prod = prod_df["Pmin"].to_numpy(dtype=np.float32)
        self.gen_max_ramp_up = prod_df["max_ramp_up"].to_numpy(dtype=np.float32)
        self.gen_max_ramp_down = prod_df["max_ramp_down"].to_numpy(dtype=np.float32)
        self.gen_type = prod_df["type"].to_numpy() 
        self.gen_voltage_levels = prod_df["V"].to_numpy(dtype=np.float32)
 

        grid_json_path = Path("C:/Users/10452/data_grid2op/l2rpn_case14_sandbox/grid.json")

        with open(grid_json_path, "r") as f:
            grid_data = json.load(f)

        # Read the bus data and sort it to make sure the index corresponds to the bus id.
        bus_json = grid_data["_object"]["bus"]
        bus_raw = json.loads(bus_json["_object"])  # This is the actual content of the DataFrame: columns, data, index

        bus_df = pd.DataFrame(data=bus_raw["data"], columns=bus_raw["columns"], index=bus_raw["index"])
        bus_df = bus_df.sort_index()  # Ensure alignment with bus_id

        self.bus_vn_kv = bus_df["vn_kv"].to_numpy(dtype=np.float32)


        self.load_p_max = np.load("load_p_max.npy")
        self.load_q_max = np.load("load_q_max.npy")

       

    def get_env_size(self) -> tuple[int, list[str]]:
        ids = [x.stem for x in Path(self.env.chronics_handler.path).glob("*") if x.is_dir()]
        return len(ids), ids


    def process_agent_action(self, action) -> BaseAction:
       
        debug_logs = []
        n_gen = self.env.n_gen
        raw_action = np.asarray(action, dtype=np.float32)

        redispatch_mw = np.zeros(n_gen, dtype=np.float32)
        curtailment_final = np.full(n_gen, np.nan, dtype=np.float32)

        current_prod = self.tracker.state.prod_p
        is_renew = np.array([str(t).lower() in ("solar", "wind") for t in self.gen_type])

        for i in range(n_gen):
            if is_renew[i]:
                # Allowed curtailment when WTGs are available.
                if current_prod[i] > 1e-3:
                    curtailment_final[i] = np.clip(raw_action[i], 0.0, 1.0)
                else:
                    curtailment_final[i] = 0.0
            else:
                # Current output and episode Starting output
                initial = self.initial_prod_p[i]
                target_dispatch = self.tracker.state.target_dispatch[i]  # Current dispatched value (cumulative)

                red_ratio = np.clip(raw_action[i], -1.0, 1.0)
                ramp = self.gen_max_ramp_up[i] if red_ratio >= 0 else self.gen_max_ramp_down[i]
                delta = red_ratio * ramp

                # Calculating Expectations for the Next Outpouring of Effort
                next_prod = initial + target_dispatch + delta

                # clip Final target output value
                next_prod = np.clip(next_prod, self.gen_min_prod[i], self.gen_max_prod[i])

                # recalculate delta = next_prod − current − target_dispatch
                delta = next_prod - initial - target_dispatch

                # clip delta（Consider ramp limits）
                delta = np.clip(delta, -self.gen_max_ramp_down[i], self.gen_max_ramp_up[i])

                redispatch_mw[i] = delta

                setpoint = float(target_dispatch+initial)

                debug_logs.append(
                    f"[Gen {i}] red_ratio={red_ratio:.2f}, delta={delta:.2f}, "
                    f"setpoint={setpoint:.2f}, "
                    f"current={current_prod[i]:.2f},"
                    f"[Pmax={self.gen_max_prod[i]:.2f}]"
                )

        redispatch_list = [
            (i, redispatch_mw[i])
            for i in range(n_gen)
            if not is_renew[i] and abs(redispatch_mw[i]) > 1e-3
        ]

        curtail_list = [
            (i, curtailment_final[i])
            for i in range(n_gen)
            if is_renew[i] and not np.isnan(curtailment_final[i]) 
        ]

        act_dict = {}
        if redispatch_list:
            act_dict["redispatch"] = redispatch_list
        if curtail_list:
            act_dict["curtail"] = curtail_list

       
        return self.env.action_space(act_dict), debug_logs, redispatch_mw


    # def convert_observation(self, observation: BaseObservation) -> np.ndarray:
    def convert_observation(self, observation: BaseObservation, redispatch_mw: np.ndarray) -> np.ndarray:
        obs = observation
        eps = 1e-6

        # Standardised power
        prod_p = obs.prod_p / (self.gen_max_prod + eps)
        load_p = obs.load_p / (self.load_p_max + eps)
        load_q = obs.load_q / (self.load_q_max + eps)
        
        prod_v = obs.prod_v / self.gen_voltage_levels
        
        voltage_levels = np.array([138.0, 20.0, 14.0], dtype=np.float32)

        # For each line starting voltage, match the nearest rated voltage
        nearest_vn = np.array([
            voltage_levels[np.argmin(np.abs(voltage_levels - v))]
            for v in obs.v_or
        ], dtype=np.float32)

        v_or_norm = obs.v_or / (nearest_vn)
        nearest_vn_2 = np.array([voltage_levels[np.argmin(np.abs(voltage_levels - v))] for v in obs.v_ex], dtype=np.float32)
        v_ex_norm = obs.v_ex / nearest_vn_2
        
        # State Class Characteristics
        line_status = obs.line_status.astype(np.float32)
        # print("line_status:",line_status)
        topo_vect = obs.topo_vect.astype(np.float32)

        dispatchable_idx = [i for i, t in enumerate(self.gen_type) if str(t).lower() not in ("solar", "wind")]
        n_dispatchable = len(dispatchable_idx)

        #  setpoint = target_dispatch + initial_prod_p
        setpoint = np.zeros(len(self.gen_type), dtype=np.float32)
        setpoint[dispatchable_idx] = obs.target_dispatch[dispatchable_idx] + self.initial_prod_p[dispatchable_idx]
        setpoint_norm = setpoint[dispatchable_idx] / (self.gen_max_prod[dispatchable_idx] + eps)

        # Calculate the upper and lower adjustment margins
        delta_up = np.zeros(n_dispatchable, dtype=np.float32)
        delta_down = np.zeros(n_dispatchable, dtype=np.float32)

        for j, i in enumerate(dispatchable_idx):
            up_margin = min(self.gen_max_prod[i] - setpoint[i], self.gen_max_ramp_up[i])
            down_margin = min(setpoint[i] - self.gen_min_prod[i], self.gen_max_ramp_down[i])
            delta_up[j] = max(up_margin, 0.0)
            delta_down[j] = max(down_margin, 0.0)

        delta_up_ratio = np.clip(delta_up / (self.gen_max_ramp_up[dispatchable_idx] + eps), 0.0, 1.0)
        delta_down_ratio = np.clip(delta_down / (self.gen_max_ramp_down[dispatchable_idx] + eps), 0.0, 1.0)


    


        features = np.concatenate([
            obs.rho,            # line load factor
            prod_p, 
            prod_v,     # Power generation, voltage
            load_p, load_q,     # load
            delta_up_ratio,
            delta_down_ratio
        ])
        # print(features)
        return features.astype(np.float32)


    def step(self, agent_action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.tracker.reset_step()
        action, debug_logs, redispatch_mw = self.process_agent_action(agent_action)
        
        obs, reward, done, info = self.env.step(action)
        self.tracker.step(obs, reward, done, info)


        if reward in (-0.5, 0.5):
            print(" reward =", reward, "| Info:", info)
            print(" Raw agent_action:", np.round(agent_action, 3))
            print(" Final redispatch:", np.round(action.redispatch, 3))
            print(" Final curtailment:", np.round(action.curtail, 3))
            for log in debug_logs:
                print(log)


        self._step_while_safe()

        self.tracker.info.update({"time": time.perf_counter() - self.tracker.start})
        obs_vec = self.convert_observation(self.tracker.state, redispatch_mw)
        terminated, truncated = self._get_terminated_truncated()

        n_gen = len(self.gen_type)
        action_processed = np.zeros(n_gen, dtype=np.float32)
        for i in range(n_gen):
            if str(self.gen_type[i]).lower() in ("solar", "wind"):
                action_processed[i] = np.clip(action.curtail[i], 0.0, 1.0)
            else:
                delta = redispatch_mw[i]
                ramp = self.gen_max_ramp_up[i] if delta >= 0 else self.gen_max_ramp_down[i]
                action_processed[i] = np.clip(delta / (ramp + 1e-6), -1.0, 1.0)


        return obs_vec, self.tracker.tot_reward, terminated, truncated, self.tracker.info, action_processed

    def _step_while_safe(self):
        while not self.tracker.done and not np.any(self.tracker.state.rho >= self.rho_threshold):
            action_ = self.env.action_space({})
            obs, reward, done, info = self.env.step(action_)
            self.tracker.step(obs, reward, done, info)

    def reset(self, seed: int | None = None, options: RESET_OPTIONS_TYPING = {}) -> Tuple[np.ndarray, dict, bool, bool]:
        if "time serie id" in options:
            ep_id = options["time serie id"]
        else:
            ep_id = self.env.chronics_handler.get_name()
            options["time serie id"] = ep_id

        obs = self.env.reset(options=options)
        self.initial_prod_p = obs.prod_p.copy()  # Record initial output
        self.tracker.reset_episode(obs)

        n_gen = len(self.gen_type)
        empty_redispatch = np.zeros(n_gen, dtype=np.float32)
        obs_vec = self.convert_observation(self.tracker.state, empty_redispatch)

        terminated, truncated = self._get_terminated_truncated()
        return obs_vec, dict(reward=0), terminated, truncated

    def _get_terminated_truncated(self) -> Tuple[bool, bool]:
        done = self.tracker.done
        step = self.env.nb_time_step
        env_max_step = self.env.max_episode_duration()
        terminated = done and not (step == env_max_step)
        truncated = done and (step == env_max_step)
        return terminated, truncated

    def set_id(self, chronic_id: int | str):
        self.env.set_id(chronic_id)

    def seed(self, seed: int) -> None:
        self.env.seed(seed=seed)

    def max_episode_duration(self) -> int:
        return self.env.max_episode_duration()

