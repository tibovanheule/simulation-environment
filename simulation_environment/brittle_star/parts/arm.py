from typing import Union, List, Tuple

import numpy as np
from dm_control import mjcf

from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology
from erpy.utils import colors
from simulation_environment.brittle_star.parts.arm_segment import MJCBrittleStarArmSegment
from simulation_environment.brittle_star.specification import BrittleStarMorphologySpecification
from simulation_environment.brittle_star.utils.tendon import calculate_relaxed_tendon_length


class MJCBrittleStarArm(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> BrittleStarMorphologySpecification:
        return super().specification

    def _build(self, arm_index: int) -> None:
        self._arm_index = arm_index
        self._arm_specification = self.specification.arm_specifications[self._arm_index]

        self._build_segments()
        self._configure_tendon_attachment_points()
        self._build_tendons()
        self._configure_actuators()
        self._configure_sensors()

    def _build_segments(self) -> None:
        self._segments = []

        number_of_segments = self._arm_specification.number_of_segments

        for segment_index in range(number_of_segments):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
            except IndexError:
                position = np.zeros(3)
                parent = self

            segment = MJCBrittleStarArmSegment(parent=parent,
                                               name=f"{self.base_name}_segment_{segment_index}",
                                               pos=position,
                                               euler=np.zeros(3),
                                               arm_index=self._arm_index,
                                               segment_index=segment_index)
            self._segments.append(segment)

    def _configure_tendon_attachment_points(self) -> None:
        first_segment = self._segments[0]

        attachment_points = first_segment.tendon_attachment_points["proximal"]
        attachment_positions = [attachment_point.pos for attachment_point in attachment_points]

        self._tendon_attachment_points = []
        for tendon_index, attachment_position in enumerate(attachment_positions):
            attachment_position[0] = 0.0
            attachment_point = self.mjcf_body.add("site",
                                                  type="sphere",
                                                  name=f"{self.base_name}_muscle_attachment_point_{tendon_index}",
                                                  pos=attachment_position,
                                                  size=[0.01],
                                                  rgba=colors.rgba_red)
            self._tendon_attachment_points.append(attachment_point)

    def _get_tendon_morphology_parts_and_attachment_points(self, tendon_index: int) -> Tuple[
        List[MJCMorphologyPart], List[mjcf.Element]]:
        attachment_points = [self._tendon_attachment_points[tendon_index]]
        morphology_parts = [self]

        for segment in self._segments:
            for side in ["proximal", "distal"]:
                attachment_point = segment.tendon_attachment_points[side][tendon_index]
                attachment_points.append(attachment_point)
                morphology_parts.append(segment)

        return morphology_parts, attachment_points

    def _build_tendons(self) -> None:
        num_tendons = len(self._tendon_attachment_points)
        self._tendons = []
        for tendon_index in range(num_tendons):
            morphology_parts, attachment_points = self._get_tendon_morphology_parts_and_attachment_points(
                tendon_index=tendon_index)
            relaxed_length = calculate_relaxed_tendon_length(morphology_parts=morphology_parts,
                                                             attachment_points=attachment_points)

            contraction_factor = self.specification.tendon_specification.contraction_factor.value
            stretch_factor = self.specification.tendon_specification.stretch_factor.value
            min_tendon_length = (1 - contraction_factor) * relaxed_length
            max_tendon_length = (1 + stretch_factor) * relaxed_length

            tendon = self.mjcf_model.tendon.add("spatial",
                                                name=f"{self.base_name}_tendon_{tendon_index}",
                                                width=0.01,
                                                rgba=colors.rgba_tendon_relaxed,
                                                limited=True,
                                                range=(min_tendon_length, max_tendon_length))
            for attachment_point in attachment_points:
                tendon.add('site', site=attachment_point)

            self._tendons.append(tendon)

    def _configure_actuators(self) -> None:
        for tendon in self._tendons:
            self.mjcf_model.actuator.add('motor',
                                         name=f"{tendon.name}_motor",
                                         tendon=tendon,
                                         gear=[200],
                                         forcelimited=True,
                                         forcerange=[-200, 0],
                                         ctrllimited=True,
                                         ctrlrange=[-1, 1])

    def _configure_muscle_length_sensors(self) -> None:
        for tendon in self._tendons:
            self.mjcf_model.sensor.add("tendonpos",
                                       name=f"{tendon.name}_tendonpos",
                                       tendon=tendon)

    def _configure_sensors(self) -> None:
        self._configure_muscle_length_sensors()
