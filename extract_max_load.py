import pandas as pd
import numpy as np
from pathlib import Path
import json

def extract_max_load_bz2():
    chronics_path = Path("C:/Users/10452/data_grid2op/l2rpn_case14_sandbox/chronics")
    config_path = Path("C:/Users/10452/data_grid2op/l2rpn_case14_sandbox/config.py")

    p_max_list = []
    q_max_list = []

    print(f"🔍 Scanning chronics in: {chronics_path.resolve()}")

    for chronic_dir in chronics_path.iterdir():
        if chronic_dir.is_dir():
            load_p_file = chronic_dir / "load_p.csv.bz2"
            load_q_file = chronic_dir / "load_q.csv.bz2"

            if load_p_file.exists() and load_q_file.exists():
                try:
                    df_p = pd.read_csv(load_p_file, compression="bz2", sep=";")
                    df_q = pd.read_csv(load_q_file, compression="bz2", sep=";")

                    p_max = df_p.max().to_numpy()
                    q_max = df_q.max().to_numpy()

                    p_max_list.append(p_max)
                    q_max_list.append(q_max)

                except Exception as e:
                    print(f"⚠️ Failed reading {chronic_dir.name}: {e}")

    if not p_max_list:
        raise ValueError("❌ No valid load_p/q.csv.bz2 files found.")

    load_p_max = np.max(np.vstack(p_max_list), axis=0).astype(np.float32)
    load_q_max = np.max(np.vstack(q_max_list), axis=0).astype(np.float32)

    # ✅ 从 config.py 中读取 thermal_limits
    print(f"\n📂 Reading thermal_limits from: {config_path.resolve()}")
    config_data = {}
    with open(config_path, "r") as f:
        exec(f.read(), config_data)

    thermal_limits = config_data["config"].get("thermal_limits", None)
    if thermal_limits is None:
        raise ValueError("❌ 'thermal_limits' not found in config.py!")

    thermal_limits = np.array(thermal_limits, dtype=np.float32)
    np.save("line_p_or_max.npy", thermal_limits)  # 用于 p_or 归一化

    # ✅ 保存负荷最大值
    np.save("load_p_max.npy", load_p_max)
    np.save("load_q_max.npy", load_q_max)

    print("\n🎉 Extraction complete.")
    print("➡️  Global load_p_max:", load_p_max)
    print("➡️  Global load_q_max:", load_q_max)
    print("➡️  Saved line_p_or_max (from thermal_limits):", thermal_limits)
    print("💾 Saved to: load_p_max.npy, load_q_max.npy, line_p_or_max.npy")

if __name__ == "__main__":
    extract_max_load_bz2()
