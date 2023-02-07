from collections import defaultdict
from typing import Union, Dict, List

import numpy as np
from dm_control import mjcf

from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology
from erpy.utils import colors
from simulation_environment.brittle_star.specification.specification import BrittleStarMorphologySpecification, \
    BrittleStarJointSpecification


class MJCBrittleStarArmSegment(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> BrittleStarMorphologySpecification:
        return super().specification

    def _build(self, arm_index: int, segment_index: int) -> None:
        self._arm_index = arm_index
        self._segment_index = segment_index

        self._arm_specification = self.specification.arm_specifications[self._arm_index]
        self._segment_specification = self._arm_specification.segment_specifications[self._segment_index]

        self._tendon_attachment_points = defaultdict(list)

        self._build_capsule()
        self._configure_joints()
        self._build_tendon_plates()

    def _build_capsule(self) -> None:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        self.mjcf_body.add("geom",
                           name=f"{self.base_name}_capsule",
                           type="capsule",
                           pos=self.center_of_capsule,
                           euler=[0, np.pi / 2, 0],
                           size=[radius, length / 2],
                           rgba=colors.rgba_green)

    @property
    def center_of_capsule(self) -> np.ndarray:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value
        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])

    def _configure_joint(self, name: str, axis: np.ndarray,
                         joint_specification: BrittleStarJointSpecification) -> mjcf.Element:
        joint = self.mjcf_body.add("joint",
                                   name=name,
                                   type="hinge",
                                   limited=True,
                                   range=joint_specification.range.value,
                                   axis=axis,
                                   pos=np.zeros(3),
                                   stiffness=joint_specification.stiffness.value,
                                   damping=joint_specification.damping)
        return joint

    def _configure_joints(self) -> None:
        self._in_plane_joint = self._configure_joint(name=f"{self.base_name}_in_plane_joint",
                                                     axis=[0, 0, 1],
                                                     joint_specification=self._segment_specification.in_plane_joint_specification)
        self._out_of_plane_joint = self._configure_joint(name=f"{self.base_name}_out_of_plane_joint",
                                                         axis=[0, 1, 0],
                                                         joint_specification=self._segment_specification.out_of_plane_joint_specification)

    @property
    def tendon_plate_radius(self) -> float:
        capsule_radius = self._segment_specification.radius.value
        tendon_offset = self._segment_specification.tendon_offset.value

        tendon_plate_radius = capsule_radius + tendon_offset
        return tendon_plate_radius

    def _build_tendon_plate(self, side: str, position: np.ndarray) -> None:
        capsule_radius = self._segment_specification.radius.value
        tendon_offset = self._segment_specification.tendon_offset.value

        tendon_plate_radius = capsule_radius + tendon_offset
        self.mjcf_body.add("geom",
                           name=f"{self.base_name}_tendon_plate_{side}",
                           type="cylinder",
                           size=[tendon_plate_radius, 0.01],
                           pos=position,
                           euler=[0, np.pi / 2, 0],
                           rgba=colors.rgba_green)

        self._configure_tendon_attachment_points(side=side, x_offset=position[0])

    def _build_tendon_plates(self) -> None:
        self._build_tendon_plate(side="proximal",
                                 position=0.5 * self.center_of_capsule)
        self._build_tendon_plate(side="distal",
                                 position=1.5 * self.center_of_capsule)

    @property
    def tendon_attachment_points(self) -> Dict[str, List[mjcf.Element]]:
        return self._tendon_attachment_points

    def _configure_tendon_attachment_points(self, side: str, x_offset: float) -> None:
        # 4 equally spaced tendon attachment points
        angles = [-np.pi / 4, - 3 * np.pi / 4, np.pi / 4, 3 * np.pi / 4]

        tendon_plate_radius = self.tendon_plate_radius
        for tendon_index, angle in enumerate(angles):
            position = tendon_plate_radius * np.array([0, np.cos(angle), np.sin(angle)])
            position[0] = x_offset
            name = f"{self.base_name}_muscle_attachment_point_{side}_{tendon_index}"

            attachment_point = self.mjcf_body.add("site",
                                                  type="sphere",
                                                  name=name,
                                                  pos=position,
                                                  size=[0.01],
                                                  rgba=colors.rgba_red)

            self.tendon_attachment_points[side].append(attachment_point)
