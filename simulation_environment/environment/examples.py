import numpy as np
from dm_control import composer, viewer
from dm_env import TimeStep

from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.brittle_star.specification import examples
from simulation_environment.environment.locomotion.task import LocomotionEnvironmentConfig

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
