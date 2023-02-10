import numpy as np
from dm_control import mjcf

from erpy.framework.specification import RobotSpecification
from erpy.interfaces.mujoco.phenome import MJCMorphology
from erpy.utils import colors
from simulation_environment.brittle_star.morphology.observables import BrittleStarObservables
from simulation_environment.brittle_star.morphology.parts.arm import MJCBrittleStarArm
from simulation_environment.brittle_star.morphology.parts.disc import MJCBrittleStarDisc
from simulation_environment.brittle_star.specification.default import default_brittle_star_morphology_specification
from simulation_environment.brittle_star.specification.specification import BrittleStarMorphologySpecification


class MJCBrittleStarMorphology(MJCMorphology):
    def __init__(self, specification: RobotSpecification) -> None:
        super().__init__(specification)

    @property
    def morphology_specification(self) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(self, *args, **kwargs):
        self._configure_compiler()
        self._build_disc()
        self._build_arms()

        self._prepare_tendon_coloring()

    def _build_observables(self) -> None:
        return BrittleStarObservables(self)

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _build_disc(self) -> None:
        self._disc = MJCBrittleStarDisc(parent=self,
                                        name="central_disc",
                                        pos=np.zeros(3),
                                        euler=np.zeros(3))

    def _build_arms(self) -> None:
        # Equally spaced over the disc
        self._arms = []

        disc_radius = self.morphology_specification.disc_specification.radius.value

        number_of_arms = self.morphology_specification.number_of_arms
        for arm_index, arm_specification in enumerate(self.morphology_specification.arm_specifications):
            normalized_index = arm_index / number_of_arms
            angle = normalized_index * 2 * np.pi
            position = disc_radius * np.array([np.cos(angle), np.sin(angle), 0])

            arm = MJCBrittleStarArm(parent=self._disc,
                                    name=f"arm_{arm_index}",
                                    pos=position,
                                    euler=[0, 0, angle],
                                    arm_index=arm_index)
            self._arms.append(arm)

    def _prepare_tendon_coloring(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons:
            self._tendon_actuators = list(filter(lambda actuator: actuator.tendon is not None, self.actuators))
            self._tendons = [actuator.tendon for actuator in self._tendon_actuators]

            self._contracted_rgbas = np.ones((len(self._tendons), 4))
            self._contracted_rgbas[:] = colors.rgba_tendon_contracted

            self._color_changes = np.ones((len(self._tendons), 4))
            self._color_changes[:] = colors.rgba_tendon_relaxed - colors.rgba_tendon_contracted
            self._color_changes = self._color_changes.T

    def _color_muscles(self, physics: mjcf.Physics) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            # Called often -> need high performance -> cache stuff as much as possible -> _prepare_tendon_coloring()!

            tendon_control = np.array(physics.bind(self._tendon_actuators).ctrl)
            # [-1, 1] to [0, 1] (relaxation because 0 -> fully contracted and 1 means relaxed
            tendon_relaxation = tendon_control + 1 / 2

            physics.bind(self._tendons).rgba = self._contracted_rgbas + (tendon_relaxation * self._color_changes).T

    def after_step(self, physics, random_state) -> None:
        self._color_muscles(physics=physics)


if __name__ == '__main__':
    specification = default_brittle_star_morphology_specification()
    morphology = MJCBrittleStarMorphology(specification)
    morphology.export_to_xml_with_assets()
