import sys, argparse, csv, collections, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs"), str(ROOT / "expert")]
from panda_env import PandaPickEnv
from scripted_expert import ScriptedExpert

p = argparse.ArgumentParser()
p.add_argument("--model")
p.add_argument("--expert", action="store_true", help="evaluate the scripted expert instead")
p.add_argument("--n", type=int, default=100)
p.add_argument("--name", required=True)
args = p.parse_args()

env = PandaPickEnv()
if args.expert:
    expert = ScriptedExpert()
else:
    ck = torch.load(args.model, weights_only=False)     # checkpoint holds numpy stats
    mean, std = ck["mean"], ck["std"]
    d = len(mean)
    model = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4), nn.Tanh())
    model.load_state_dict(ck["model"]); model.eval()
    def policy(obs):
        x = torch.tensor(((obs - mean) / std).astype(np.float32))
        with torch.no_grad():
            return model(x).numpy()

rows = []
for i in range(args.n):
    seed = 1000 + i
    obs = env.reset(seed=seed)
    if args.expert: expert.reset()
    start = obs[11:14].copy()
    min_xy, min_hz, closed = 9.0, 9.0, 0
    for t in range(env.max_steps):
        a = expert.act(obs) if args.expert else policy(obs)
        min_xy = min(min_xy, np.linalg.norm(obs[8:10] - obs[11:13]))
        min_hz = min(min_hz, obs[10])
        closed += int(a[3] < 0)
        obs, success, timeout = env.step(a)
        if success or timeout: break
    moved = np.linalg.norm(obs[11:13] - start[:2])
    if success:              cat = "success"
    elif min_xy > 0.02:      cat = "never_reached_cube"
    elif min_hz > 0.14:      cat = "did_not_descend"
    elif closed == 0:        cat = "never_closed_gripper"
    elif moved > 0.03:       cat = "cube_knocked"
    elif obs[13] < 0.05:     cat = "grasp_failed"
    else:                    cat = "other"
    rows.append([seed, start[0], start[1], int(success), cat, t + 1, obs[13]])

out = ROOT / f"eval/results_{args.name}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "cube_x", "cube_y", "success", "category", "steps", "final_cube_z"])
    w.writerows(rows)

ok = np.array([r[3] for r in rows]); x = np.array([r[1] for r in rows])
print(f"\n{args.name}: {ok.sum()}/{len(rows)} success ({100*ok.mean():.0f}%)")
print("cube x < 0.46 :", f"{ok[x < 0.46].sum()}/{(x < 0.46).sum()}")
print("cube x >= 0.46:", f"{ok[x >= 0.46].sum()}/{(x >= 0.46).sum()}")
for k, v in collections.Counter(r[4] for r in rows).most_common():
    print(f"  {k:22s} {v}")
print("saved", out)
