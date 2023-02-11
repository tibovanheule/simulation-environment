import numpy as np

from simulation_environment.brittle_star.specification.specification import BrittleStarMorphologySpecification, \
    BrittleStarDiscSpecification, BrittleStarJointSpecification, BrittleStarArmSegmentSpecification, \
    BrittleStarArmSpecification, BrittleStarTendonSpecification, BrittleStarActuationSpecification


def default_brittle_star_morphology_specification(num_arms: int = 5,
                                                  num_segments_per_arm: int = 5,
                                                  use_cartesian_actuation: bool = False) -> BrittleStarMorphologySpecification:
    disc_specification = BrittleStarDiscSpecification(radius=0.25, height=0.2)

    arm_specifications = list()
    for arm_index in range(num_arms):
        segment_specifications = list()
        for segment_index in range(num_segments_per_arm):
            in_plane_joint_spec = BrittleStarJointSpecification(
                range=np.array([-10, 10]) / 180 * np.pi,
                stiffness=21.674,
                damping_factor=0.1)
            out_of_plane_joint_spec = BrittleStarJointSpecification(
                range=np.array([-10, 10]) / 180 * np.pi,
                stiffness=21.674,
                damping_factor=0.1)
            segment_specification = BrittleStarArmSegmentSpecification(
                radius=0.05, length=0.1, tendon_offset=0.025,
                in_plane_joint_specification=in_plane_joint_spec,
                out_of_plane_joint_specification=out_of_plane_joint_spec)

            segment_specifications.append(segment_specification)

        arm_specification = BrittleStarArmSpecification(
            segment_specifications=segment_specifications
        )
        arm_specifications.append(arm_specification)

    tendon_specification = BrittleStarTendonSpecification(contraction_factor=0.5, stretch_factor=2.0)
    actuation_specification = BrittleStarActuationSpecification(use_tendons=not use_cartesian_actuation,
                                                                use_cartesian=use_cartesian_actuation)
    specification = BrittleStarMorphologySpecification(disc_specification=disc_specification,
                                                       arm_specifications=arm_specifications,
                                                       tendon_specification=tendon_specification,
                                                       actuation_specification=actuation_specification)

    return specification
