import os
os.environ.setdefault("MUJOCO_GL", "egl")      # if egl fails: MUJOCO_GL=osmesa python envs/render_compare.py
import sys, csv, numpy as np, torch, torch.nn as nn, mujoco, imageio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs")]
from panda_env import PandaPickEnv

def load_policy(path):
    ck = torch.load(path, weights_only=False)
    mean, std = ck["mean"], ck["std"]
    m = nn.Sequential(nn.Linear(len(mean), 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4), nn.Tanh())
    m.load_state_dict(ck["model"]); m.eval()
    def pol(obs):
        with torch.no_grad():
            return m(torch.tensor(((obs - mean) / std).astype(np.float32))).numpy()
    return pol

def rows(name):
    return {int(r["seed"]): r for r in csv.DictReader(open(ROOT / f"eval/results_{name}.csv"))}
clean, noisy = rows("bc_clean_300"), rows("bc_noisy_300")
seeds = [s for s in clean if clean[s]["category"] == "cube_knocked" and noisy[s]["success"] == "1"]
print(len(seeds), "seeds where the clean policy knocked the cube but the noisy policy succeeded")
seed = seeds[0]; print("using seed", seed)

env = PandaPickEnv()
renderer = mujoco.Renderer(env.model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.5, 0.0, 0.2]; cam.distance = 1.4; cam.azimuth = 135; cam.elevation = -25

def run(policy, path):
    obs = env.reset(seed=seed); frames = []
    for t in range(env.max_steps):
        obs, success, timeout = env.step(policy(obs))
        if t % 2 == 0:
            renderer.update_scene(env.data, camera=cam); frames.append(renderer.render())
        if success or timeout: break
    imageio.mimsave(path, frames, fps=25)
    print(path, "success" if success else "failed")

os.makedirs(ROOT / "videos", exist_ok=True)
run(load_policy(ROOT / "models/bc_clean.pt"), str(ROOT / f"videos/clean_seed{seed}.mp4"))
run(load_policy(ROOT / "models/bc_noisy.pt"), str(ROOT / f"videos/noisy_seed{seed}.mp4"))