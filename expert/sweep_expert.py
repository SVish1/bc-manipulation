import sys, itertools, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "envs"), str(ROOT / "expert")]
from panda_env import PandaPickEnv
from scripted_expert import ScriptedExpert

env = PandaPickEnv()
n = 30
for gz, cs in itertools.product([0.115, 0.120, 0.125, 0.130, 0.135, 0.140], [20]):
    ScriptedExpert.GRASP_Z, ScriptedExpert.CLOSE_STEPS = gz, cs
    expert, wins = ScriptedExpert(), 0
    for seed in range(n):
        obs = env.reset(seed=seed); expert.reset()
        for t in range(env.max_steps):
            obs, success, timeout = env.step(expert.act(obs))
            if success or timeout: break
        wins += success
    print(f"GRASP_Z={gz:.3f} CLOSE_STEPS={cs:2d} -> {wins}/{n}")
