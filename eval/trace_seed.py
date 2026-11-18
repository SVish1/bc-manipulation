
import sys, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs")]
from panda_env import PandaPickEnv
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1007

def load_policy(path):
    ck = torch.load(path, weights_only=False); mean, std = ck["mean"], ck["std"]
    m = nn.Sequential(nn.Linear(len(mean), 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4), nn.Tanh())
    m.load_state_dict(ck["model"]); m.eval()
    return lambda o: m(torch.tensor(((o - mean) / std).astype(np.float32))).detach().numpy()

def trace(pol):
    env = PandaPickEnv(); obs = env.reset(seed=SEED); c0 = obs[11:13].copy(); T = []
    for t in range(env.max_steps):
        a = pol(obs); obs, success, timeout = env.step(a)
        T.append([obs[10], np.linalg.norm(obs[8:10] - obs[11:13]), np.linalg.norm(obs[11:13] - c0), obs[7]])
        if success or timeout: break
    return np.array(T)

fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
for name in ["clean", "noisy"]:
    T = trace(load_policy(ROOT / f"models/bc_{name}.pt"))
    hit = np.where(T[:, 2] > 0.01)[0]
    if len(hit):
        i = hit[0]
        print(f"{name}: cube first moved >1 cm at step {i}: hand z={T[i,0]:.3f}, "
              f"hand-cube xy dist={T[i,1]:.3f}, finger opening={T[i,3]:.3f}")
    else:
        print(f"{name}: cube never moved more than 1 cm before the end (steps={len(T)})")
    for k, lab in enumerate(["hand z (m)", "hand-cube xy dist (m)", "cube displacement (m)"]):
        ax[k].plot(T[:, k], label=name); ax[k].set_ylabel(lab)
ax[0].legend(); ax[2].set_xlabel("step"); ax[0].set_title(f"seed {SEED}")
plt.tight_layout(); plt.savefig("eval/trace_seed.png", dpi=150); print("saved eval/trace_seed.png")
