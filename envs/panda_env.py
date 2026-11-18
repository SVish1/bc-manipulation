import numpy as np, mujoco
from pathlib import Path

XML = Path(__file__).resolve().parents[1] / "mujoco_menagerie/franka_emika_panda/pick_scene.xml"
HOME = np.array([0, -0.785, 0, -2.356, 0, 1.571, -0.785])

class PandaPickEnv:
    def __init__(self, max_steps=300):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.hand = self.model.body("hand").id
        self.cube_adr = self.model.joint("cube_joint").qposadr[0]
        self.max_steps, self.n_sub = max_steps, 10
        self.rng = np.random.default_rng()

    def reset(self, seed=None, cube_xy=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = HOME
        self.data.ctrl[:7] = HOME
        self.q_des = HOME.copy()
        self.data.ctrl[7] = 255                      # gripper open
        if cube_xy is None:
            cube_xy = self.rng.uniform([0.4, -0.2], [0.6, 0.2])
        a = self.cube_adr
        self.data.qpos[a:a+3] = [cube_xy[0], cube_xy[1], 0.02]
        self.data.qpos[a+3:a+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):                           # let things settle
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        return self._obs()

    def _obs(self):
        d = self.data
        cube = d.qpos[self.cube_adr:self.cube_adr+3]
        return np.concatenate([d.qpos[:7], d.qpos[7:8], d.xpos[self.hand], cube]).astype(np.float32)

    def step(self, action):
        """action = [dx, dy, dz, grip]; dx in [-1,1] -> up to 2 cm per step; grip>0 = open."""
        dx = np.clip(action[:3], -1, 1) * 0.02
        jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jp, jr, self.hand)
        J = np.vstack([jp, jr])[:, :7]
        twist = np.concatenate([dx, np.zeros(3)])     # hold orientation
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)
        q = self.data.qpos[:7]
        self.q_des = np.clip(self.q_des + dq, q - 0.3, q + 0.3)   # persistent target, limited windup
        self.data.ctrl[:7] = self.q_des
        self.data.ctrl[7] = 255 if action[3] > 0 else 0
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        obs = self._obs()
        success = obs[-1] > 0.10                      # cube lifted 8 cm
        return obs, success, self.t >= self.max_steps