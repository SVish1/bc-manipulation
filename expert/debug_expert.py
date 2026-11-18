import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs"), str(ROOT / "expert")]
from panda_env import PandaPickEnv
from scripted_expert import ScriptedExpert

env, expert = PandaPickEnv(), ScriptedExpert()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
wins = 0
for seed in range(n):
    obs = env.reset(seed=seed); expert.reset()
    for t in range(env.max_steps):
        obs, success, timeout = env.step(expert.act(obs))
        if success or timeout: break
    wins += success
    print(f"seed={seed:2d} success={success!s:5} steps={t+1:3d} phase={expert.phase:8s} "
          f"hand_z={obs[10]:.3f} cube=({obs[11]:.2f},{obs[12]:.2f},{obs[13]:.3f})")
print(f"{wins}/{n} succeeded")
