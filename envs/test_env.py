import numpy as np, mujoco.viewer
from panda_env import PandaPickEnv
env = PandaPickEnv(); env.reset(seed=0)
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    for _ in range(200):
        env.step(np.array([0, 0, -0.3, 1])); v.sync()