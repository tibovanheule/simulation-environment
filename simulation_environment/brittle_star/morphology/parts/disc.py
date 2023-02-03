from typing import Union

import numpy as np

from erpy.interfaces.mujoco.phenome import MJCMorphology, MJCMorphologyPart
from erpy.utils import colors
from simulation_environment.brittle_star.specification.specification import BrittleStarMorphologySpecification


class MJCBrittleStarDisc(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> BrittleStarMorphologySpecification:
        return super().specification

    def _build(self) -> None:
        self._disc_specification = self.specification.disc_specification

        self._build_cylinder()

    def _build_cylinder(self) -> None:
        radius = self.specification.disc_specification.radius.value
        height = self.specification.disc_specification.height.value

        self._disc = self.mjcf_body.add("geom",
                                        name=f"{self.base_name}_disc",
                                        type=f"cylinder",
                                        pos=np.zeros(3),
                                        euler=np.zeros(3),
                                        size=[radius, height / 2],
                                        rgba=colors.rgba_green)
