import numpy as np

from simulation_environment.brittle_star.specification import BrittleStarMorphologySpecification, \
    BrittleStarDiscSpecification, BrittleStarJointSpecification, BrittleStarArmSegmentSpecification, \
    BrittleStarArmSpecification, BrittleStarTendonSpecification


def default_brittle_star() -> BrittleStarMorphologySpecification:
    disc_specification = BrittleStarDiscSpecification(radius=0.25, height=0.1)

    joint_spec = BrittleStarJointSpecification(range=np.array([-20, 20]) / 180 * np.pi,
                                               stiffness=0.1,
                                               damping=0.01)
    arm_segment_specification = [BrittleStarArmSegmentSpecification(radius=0.05, length=0.1, tendon_offset=0.025,
                                                                    in_plane_joint_specification=joint_spec,
                                                                    out_of_plane_joint_specification=joint_spec) for _
                                 in
                                 range(5)]
    arm_specifications = [BrittleStarArmSpecification(segment_specifications=arm_segment_specification) for _ in
                          range(5)]
    tendon_specification = BrittleStarTendonSpecification(contraction_factor=0.5, stretch_factor=2.0)
    specification = BrittleStarMorphologySpecification(name="default",
                                                       disc_specification=disc_specification,
                                                       arm_specifications=arm_specifications,
                                                       tendon_specification=tendon_specification)

    return specification
