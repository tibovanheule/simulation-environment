import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from erpy.framework.specification import RobotSpecification
from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.brittle_star.specification.default import default_brittle_star_morphology_specification
from simulation_environment.environment.locomotion.task import LocomotionEnvironmentConfig


def policy_fn(timestep: TimeStep) -> np.ndarray:
    observations = timestep.observation

    arm_up = [0, 1]
    arm_down = [0, -1]
    arm_left = [1, 0]
    arm_right = [-1, 0]

    return arm_left + arm_right


if __name__ == '__main__':
    env_config = LocomotionEnvironmentConfig(with_target=True)
    morphology_specification = default_brittle_star_morphology_specification(num_arms=2, num_segments_per_arm=5,
                                                                             use_cartesian_actuation=False)
    robot_specification = RobotSpecification(morphology_specification=morphology_specification,
                                             controller_specification=None)
    morphology = MJCBrittleStarMorphology(specification=robot_specification)
    morphology.export_to_xml_with_assets("tendon")
    dm_env = env_config.environment(morphology=morphology,
                                    wrap2gym=False)

    observation_spec = env_config.observation_specification
    action_spec = env_config.action_specification

    num_actions = action_spec.shape[0]

    viewer.launch(dm_env, policy=policy_fn)
