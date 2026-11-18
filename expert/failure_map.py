import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs"), str(ROOT / "expert")]
from panda_env import PandaPickEnv
from scripted_expert import ScriptedExpert

env, expert = PandaPickEnv(), ScriptedExpert()
fails, wins = [], []
for seed in range(100):
    obs = env.reset(seed=seed); expert.reset()
    start = obs[11:13].copy()
    for t in range(env.max_steps):
        obs, success, timeout = env.step(expert.act(obs))
        if success or timeout: break
    (wins if success else fails).append(start)
fails, wins = np.array(fails), np.array(wins)
print(f"{len(wins)}/100 succeeded")
print("fail x range:", fails[:,0].min().round(2), fails[:,0].max().round(2),
      "| success x range:", wins[:,0].min().round(2), wins[:,0].max().round(2))
print("fail y range:", fails[:,1].min().round(2), fails[:,1].max().round(2),
      "| success y range:", wins[:,1].min().round(2), wins[:,1].max().round(2))
print("mean |y| fail:", np.abs(fails[:,1]).mean().round(3), "success:", np.abs(wins[:,1]).mean().round(3))
