# From https://github.com/LeonidMurashov/advantage-actor-critic-gae/tree/master/SpiderEnv

import mujoco_py as mp
import os
import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env


class SpiderEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.env_timer = 0
        xml_path = os.path.join(os.path.dirname(__file__), "spider_designed.xml")
        self.model = mp.load_model_from_path(xml_path)
        self.simulation = mp.MjSim(self.model)
        self.viewer = None
        self.direction = np.array([1, 0])
        self.prevCoord = self.simulation.data.body_xpos[1][:2]
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def _move(self, angles):
        for i in range(len(angles)):
            self.simulation.data.ctrl[i] = angles[i]

    def step(self, act):
        act += np.array([-(i > 5) * 45 for i in range(18)])

        self._move(act)
        self.simulation.step()

        obs = np.concatenate(
            [
                self.simulation.data.ctrl,
                self.direction,
                self.simulation.data.body_xquat.flatten(),
                self.simulation.data.qvel.flatten(),
                self.simulation.data.body_xvelp.flatten(),
                self.simulation.data.body_xvelr.flatten(),
                self.simulation.data.cvel.flatten(),
                self.simulation.data.actuator_force.flatten(),
                self.simulation.data.actuator_velocity.flatten(),
                self.simulation.data.actuator_moment.flatten(),
            ]
        )

        current_coord = self.simulation.data.body_xpos[1][:2]
        rew = ((current_coord - self.prevCoord) * self.direction).sum() * 1000 + 0.5
        done = self.simulation.data.body_xpos[1][2] < 0.7
        self.prevCoord = current_coord.copy()

        self.env_timer += 1
        if self.env_timer >= 3000:
            done = True

        return obs, rew, done, 0

    def render(self):
        if self.viewer == None:
            self.viewer = mp.MjViewer(self.simulation)
        self.viewer.render()

    def action_sample(self):
        return [random.randint(-90, 90) for i in range(18)]

    def reset(self):
        self.env_timer = 0
        self.simulation.reset()
        self.simulation.step()
        self._move([-(i > 5) * 45 for i in range(18)])
        for i in range(50):
            self.simulation.step()

        obs = np.concatenate(
            [
                self.simulation.data.ctrl,
                self.direction,
                self.simulation.data.body_xquat.flatten(),
                self.simulation.data.qvel.flatten(),
                self.simulation.data.body_xvelp.flatten(),
                self.simulation.data.body_xvelr.flatten(),
                self.simulation.data.cvel.flatten(),
                self.simulation.data.actuator_force.flatten(),
                self.simulation.data.actuator_velocity.flatten(),
                self.simulation.data.actuator_moment.flatten(),
            ]
        )
        self.prevCoord = self.simulation.data.body_xpos[1][:2].copy()
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}
