import numpy as np
from panda_env import PandaPickEnv
env = PandaPickEnv()
obs = env.reset(seed=0)
print("start hand z:", obs[10])
for _ in range(50):
    obs, ok, done = env.step(np.array([0, 0, -0.3, 1]))
print("end hand z:  ", obs[10])
print("cube:", obs[11:14])
