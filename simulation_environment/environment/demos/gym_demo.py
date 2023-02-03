import numpy as np

from erpy.utils.video import create_video
from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.brittle_star.specification import examples
from simulation_environment.environment.locomotion.task import LocomotionEnvironmentConfig

if __name__ == '__main__':
    env_config = LocomotionEnvironmentConfig(42, np.random.RandomState(seed=42))
    specification = examples.default_brittle_star()
    morphology = MJCBrittleStarMorphology(specification=specification)

    gym_env = env_config.environment(morphology=morphology,
                                     wrap2gym=True)

    observation_spec = gym_env.observation_space
    action_spec = gym_env.action_space

    done = False
    observations = gym_env.reset()
    step = 0
    frames = []
    while not done:
        print(f"Step: {step}")
        actions = gym_env.action_space.sample()
        observations, reward, done, info = gym_env.step(actions)
        step += 1

        frame = gym_env.render()
        frames.append(frame)

    create_video(frames=frames, framerate=len(frames) / env_config.simulation_time, out_path="./env_demo.mp4")
