from typing import List, Tuple

from erpy.framework.parameters import FixedParameter
from erpy.framework.specification import MorphologySpecification, Specification


class BrittleStarJointSpecification(Specification):
    def __init__(self, range: Tuple[float, float], stiffness: float, damping: float) -> None:
        self.range = FixedParameter(value=range)
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)


class BrittleStarArmSegmentSpecification(Specification):
    def __init__(self, radius: float, length: float,
                 tendon_offset: float,
                 in_plane_joint_specification: BrittleStarJointSpecification,
                 out_of_plane_joint_specification: BrittleStarJointSpecification) -> None:
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.tendon_offset = FixedParameter(tendon_offset)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class BrittleStarArmSpecification(Specification):
    def __init__(self, segment_specifications: List[BrittleStarArmSegmentSpecification]) -> None:
        self.segment_specifications = segment_specifications

    @property
    def number_of_segments(self) -> int:
        return len(self.segment_specifications)


class BrittleStarDiscSpecification(Specification):
    def __init__(self, radius: float, height: float) -> None:
        self.radius = FixedParameter(radius)
        self.height = FixedParameter(height)


class BrittleStarTendonSpecification(Specification):
    def __init__(self, contraction_factor: float, stretch_factor: float) -> None:
        self.contraction_factor = FixedParameter(contraction_factor)
        self.stretch_factor = FixedParameter(stretch_factor)


class BrittleStarMorphologySpecification(MorphologySpecification):
    def __init__(self, name: str,
                 disc_specification: BrittleStarDiscSpecification,
                 arm_specifications: List[BrittleStarArmSpecification],
                 tendon_specification: BrittleStarTendonSpecification) -> None:
        super(BrittleStarMorphologySpecification, self).__init__(name)
        self.disc_specification = disc_specification
        self.arm_specifications = arm_specifications
        self.tendon_specification = tendon_specification

    @property
    def number_of_arms(self) -> int:
        return len(self.arm_specifications)
