import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from erpy import random_state
from erpy.framework.specification import RobotSpecification
from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.brittle_star.specification.default import default_brittle_star_morphology_specification
from simulation_environment.environment.locomotion.task import LocomotionEnvironmentConfig


def random_policy_fn(timestep: TimeStep) -> np.ndarray:
    global action_spec, num_actions

    # Generate random actions
    random_values = random_state.rand(num_actions)
    # renomalize to bounds
    random_actions = random_values * (action_spec.maximum - action_spec.minimum) + action_spec.minimum
    return random_actions


def oscillatory_policy_fn(timestep: TimeStep) -> np.ndarray:
    global num_actions

    time = timestep.observation["task/time"][0]
    actions = np.zeros(num_actions)

    # phase1 = 0
    # phase2 = np.pi
    # frequency = 1
    #
    # actions[0::4] = np.sin(frequency * time + phase1)
    # actions[1::4] = np.sin(frequency * time + phase1)
    # actions[2::4] = -np.sin(frequency * time + phase1)
    # actions[3::4] = -np.sin(frequency * time + phase2)

    up = np.array([0, 1, 0, 1])
    down = np.array([0, -1, 0, -1])
    left = np.array([1, 0, 1, 0])
    right = np.array([-1, 0, -1, 0])
    if time % 4 < 1:
        actions = up + left
    elif time % 4 < 2:
        actions = left + down
    elif time % 4 < 3:
        actions = down + right
    else:
        actions = right + up

    return actions


if __name__ == '__main__':
    env_config = LocomotionEnvironmentConfig(with_target=True)
    morphology_specification = default_brittle_star_morphology_specification(num_arms=2, num_segments_per_arm=5,
                                                                             use_cartesian_actuation=True)
    robot_specification = RobotSpecification(morphology_specification=morphology_specification,
                                             controller_specification=True)
    morphology = MJCBrittleStarMorphology(specification=robot_specification)
    dm_env = env_config.environment(morphology=morphology,
                                    wrap2gym=False)

    observation_spec = env_config.observation_specification
    action_spec = env_config.action_specification

    num_actions = action_spec.shape[0]

    viewer.launch(dm_env, policy=oscillatory_policy_fn)
