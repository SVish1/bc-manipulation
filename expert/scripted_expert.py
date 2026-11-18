import numpy as np

class ScriptedExpert:
    ABOVE_Z = 0.25     # hand height while approaching
    GRASP_Z = 0.115    # hand z at grasp (fingertips ~10 cm below hand origin) - TUNE
    LIFT_Z = 0.30
    CLOSE_STEPS = 20

    def reset(self):
        self.phase, self.close_count = "approach", 0

    def _move(self, hand, target, grip):
        delta = (np.asarray(target) - hand) / 0.02
        a = np.array([*np.clip(delta, -1, 1), grip], dtype=np.float32)
        return a, np.linalg.norm(np.asarray(target) - hand)

    def act(self, obs):
        hand, cube = obs[8:11], obs[11:14]
        if self.phase == "approach":
            a, d = self._move(hand, [cube[0], cube[1], self.ABOVE_Z], 1.0)
            if d < 0.01: self.phase = "descend"
        elif self.phase == "descend":
            a, d = self._move(hand, [cube[0], cube[1], self.GRASP_Z], 1.0)
            if d < 0.008: self.phase = "grasp"
        elif self.phase == "grasp":
            a = np.array([0, 0, 0, -1.0], dtype=np.float32)
            self.close_count += 1
            if self.close_count >= self.CLOSE_STEPS: self.phase = "lift"
        else:
            a, _ = self._move(hand, [hand[0], hand[1], self.LIFT_Z], -1.0)
        return a
