import sys, argparse, numpy as np, h5py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs"), str(ROOT / "expert")]
from panda_env import PandaPickEnv
from scripted_expert import ScriptedExpert

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=300)
p.add_argument("--noise", type=float, default=0.1)
p.add_argument("--out", default=str(ROOT / "data/demos.h5"))
args = p.parse_args()

env, expert = PandaPickEnv(), ScriptedExpert()
noise_rng = np.random.default_rng(123)
kept = 0
with h5py.File(args.out, "w") as f:
    for i in range(args.episodes):
        obs = env.reset(seed=i); expert.reset()
        O, A = [], []
        for _ in range(env.max_steps):
            a = expert.act(obs)
            a_exec = a.copy()
            a_exec[:3] += noise_rng.normal(0, args.noise, 3)
            O.append(obs); A.append(a)
            obs, success, timeout = env.step(a_exec)
            if success or timeout: break
        if success:
            g = f.create_group(f"ep_{kept:04d}")
            g["obs"], g["actions"] = np.array(O), np.array(A)
            kept += 1
    f.attrs["n_episodes"] = kept
print(f"kept {kept}/{args.episodes} successful episodes -> {args.out}")
