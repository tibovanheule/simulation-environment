import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.brittle_star.specification.default import default_brittle_star_morphology_specification
from simulation_environment.environment.locomotion.task import LocomotionEnvironmentConfig


def random_policy_fn(timestep: TimeStep) -> np.ndarray:
    global action_spec, num_actions

    # Generate random actions
    random_values = np.random.rand(num_actions)
    # renomalize to bounds
    random_actions = random_values * (action_spec.maximum - action_spec.minimum) + action_spec.minimum
    return random_actions


def oscillatory_policy_fn(timestep: TimeStep) -> np.ndarray:
    global num_actions

    time = timestep.observation["time"][0]

    actions = np.ones(num_actions)

    phase1 = 0
    phase2 = np.pi
    frequency = 5

    # actions[::2] = np.sin(frequency * time + phase1)
    # actions[1::2] = np.sin(frequency * time + phase2)
    actions[::4] = np.sin(frequency * time + phase1)
    actions[1::4] = np.sin(frequency * time + phase1)
    actions[2::4] = np.sin(frequency * time + phase2)
    actions[3::4] = np.sin(frequency * time + phase2)

    return actions


if __name__ == '__main__':
    env_config = LocomotionEnvironmentConfig(42, np.random.RandomState(seed=42))
    specification = default_brittle_star_morphology_specification()
    morphology = MJCBrittleStarMorphology(specification=specification)

    dm_env = env_config.environment(morphology=morphology,
                                    wrap2gym=False)

    observation_spec = env_config.observation_specification
    action_spec = env_config.action_specification

    num_actions = action_spec.shape[0]

    # viewer.launch(dm_env, policy=random_policy_fn)

    viewer.launch(dm_env, policy=oscillatory_policy_fn)
