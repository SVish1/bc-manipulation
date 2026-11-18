import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs")]
from panda_env import PandaPickEnv
SEED = 1007

def load_policy(path):
    ck = torch.load(path, weights_only=False); mean, std = ck["mean"], ck["std"]
    m = nn.Sequential(nn.Linear(len(mean), 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4), nn.Tanh())
    m.load_state_dict(ck["model"]); m.eval()
    return lambda o: m(torch.tensor(((o - mean) / std).astype(np.float32))).detach().numpy()

for name in ["clean", "noisy"]:
    pol = load_policy(ROOT / f"models/bc_{name}.pt")
    env = PandaPickEnv(); obs = env.reset(seed=SEED); c0 = obs[11:13].copy()
    print(f"\n{name}: step | hand z | xy dist | finger | cube z | cube moved | grip cmd")
    for t in range(100):
        a = pol(obs)
        obs, success, timeout = env.step(a)
        if 40 <= t <= 99 and t % 3 == 0:
            print(f"{t:4d} | {obs[10]:.3f} | {np.linalg.norm(obs[8:10]-obs[11:13]):.3f} | "
                  f"{obs[7]:.3f} | {obs[13]:.3f} | {np.linalg.norm(obs[11:13]-c0):.3f} | {a[3]:+.2f}")
        if success or timeout: break