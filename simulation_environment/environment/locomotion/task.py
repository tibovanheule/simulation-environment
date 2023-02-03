from __future__ import annotations

from typing import Dict, Callable

import numpy as np
from dm_control import composer
from dm_control import mjcf, viewer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import Floor
from dm_control.mujoco.math import euler2quat
from dm_env import TimeStep

from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.phenome import MJCMorphology
from simulation_environment.brittle_star import examples
from simulation_environment.brittle_star.brittle_star import MJCBrittleStarMorphology


class LocomotionTask(composer.Task):
    def __init__(self, config: LocomotionEnvironmentConfig, morphology: MJCMorphology) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._agent = self._attach_agent(agent=morphology)
        self._task_observables = self._configure_observables()

        self._previous_distance_from_origin = 0.0

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def task_observables(self) -> Dict:
        return self._task_observables

    def _build_arena(self) -> composer.Arena:
        arena = Floor()
        return arena

    def _attach_agent(self, agent: MJCMorphology) -> MJCMorphology:
        self._arena.add_free_entity(entity=morphology)
        agent.after_attachment()
        return agent

    def _configure_agent_observables(self) -> None:
        self._agent.observables.enable_all()

    def _get_agent_position(self, physics: mjcf.Physics) -> np.ndarray:
        position, quaternion = self._agent.get_pose(physics=physics)
        return position

    def _configure_task_observables(self) -> Dict[str, observable.Observable]:
        task_observables = dict()
        task_observables["robot_position"] = observable.Generic(self._get_agent_position)

        for obs in task_observables.values():
            obs.enabled = True

        return task_observables

    def _configure_observables(self) -> Dict[str, observable.Observable]:
        self._configure_agent_observables()
        task_observables = self._configure_task_observables()
        return task_observables

    def _calculate_agent_distance_from_origin(self, physics: mjcf.Physics) -> float:
        position = self._get_agent_position(physics=physics)
        position_in_xy_plane = position[:2]
        distance_from_origin = np.linalg.norm(position_in_xy_plane)
        return distance_from_origin

    def get_reward(self, physics: mjcf.Physics) -> None:
        # Locomotion reward -> additional distance travelled from origin since previous timestep
        #   Positive if we travelled further, negative if we went back closer to origin
        current_distance_from_origin = self._calculate_agent_distance_from_origin(physics=physics)
        reward = current_distance_from_origin - self._previous_distance_from_origin
        self._previous_distance_from_origin = current_distance_from_origin

        return reward

    def _initialize_agent_pose(self, physics: mjcf.Physics) -> None:
        initial_position = np.array([0.0, 0.0, 0.5])
        initial_quaternion = euler2quat(0, 0, 0)

        self._agent.set_pose(physics=physics,
                             position=initial_position,
                             quaternion=initial_quaternion)

    def initialize_episode(self, physics: mjcf.Physics, random_state: np.random.RandomState) -> None:
        self._initialize_agent_pose(physics=physics)

        self._previous_distance_from_origin = self._calculate_agent_distance_from_origin(physics=physics)


class LocomotionEnvironmentConfig(MJCEnvironmentConfig):
    @property
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task]:
        return LocomotionTask

    @property
    def simulation_time(self) -> float:
        return 10.0

    @property
    def num_substeps(self) -> int:
        return 20

    @property
    def time_scale(self) -> float:
        return 1.0

    @property
    def physics_time_delta(self) -> float:
        return 0.002


if __name__ == '__main__':
    env_config = LocomotionEnvironmentConfig(42, np.random.RandomState(seed=42))
    specification = examples.default_brittle_star()
    morphology = MJCBrittleStarMorphology(specification=specification)

    task = env_config.task(env_config, morphology)
    dm_env = composer.Environment(task=task)

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()

    num_actions = action_spec.shape[0]


    def random_policy_fn(timestep: TimeStep) -> np.ndarray:
        global action_spec, num_actions

        # Generate random actions
        random_values = np.random.rand(num_actions)
        # renomalize to bounds
        random_actions = random_values * (action_spec.maximum - action_spec.minimum) + action_spec.minimum
        return random_actions


    viewer.launch(dm_env, policy=random_policy_fn)

    t = 0
    period = 2000


    def oscillatory_policy_fn(timestep: TimeStep) -> np.ndarray:
        global action_spec, num_actions, t

        actions = np.ones(num_actions)

        actions[::2] = 1
        actions[1::2] = -1
        if t > period // 2:
            actions *= -1

        t = (t + 1) % period
        return actions


    viewer.launch(dm_env, policy=oscillatory_policy_fn)
