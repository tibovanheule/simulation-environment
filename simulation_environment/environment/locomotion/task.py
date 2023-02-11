from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Union, Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import Floor
from dm_control.mujoco.math import euler2quat
from transforms3d.euler import quat2euler

from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.phenome import MJCMorphology
from simulation_environment.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from simulation_environment.environment.locomotion.entities.target import Target


class LocomotionTask(composer.Task):
    def __init__(self, config: LocomotionEnvironmentConfig, morphology: MJCBrittleStarMorphology) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._morphology = self._attach_morphology(morphology=morphology)
        self._target = self._attach_target()
        self._task_observables = self._configure_observables()

        self._configure_contacts()
        self._previous_distance = 0.0

    @property
    def root_entity(self) -> Union[composer.Entity, Floor]:
        return self._arena

    @property
    def task_observables(self) -> Dict:
        return self._task_observables

    def _build_arena(self) -> composer.Arena:
        arena = Floor(size=(20, 20))
        return arena

    def _configure_contacts(self) -> None:
        geom_default = self.root_entity.mjcf_model.default.geom

        # Disable all collisions between geoms by default
        geom_default.contype = 1
        geom_default.conaffinity = 0

        # Set default friction lower to make the sliding-based locomotion a bit easier
        geom_default.friction = [0.3, 0.1, 0.1]

        # Need to add conaffinity 1 to the floor -> we still want contacts there
        self.root_entity._ground_geom.conaffinity = 1

    def _attach_morphology(self, morphology: MJCBrittleStarMorphology) -> MJCBrittleStarMorphology:
        self._arena.add_free_entity(entity=morphology)
        morphology.after_attachment()
        return morphology

    def _attach_target(self) -> Optional[composer.Entity]:
        if self.config.with_target:
            target = Target()
            self._arena.attach(target)
            return target

    def _configure_morphology_observables(self) -> None:
        self._morphology.observables.enable_all()

    def _get_morphology_position(self, physics: mjcf.Physics) -> np.ndarray:
        position, _ = self._morphology.get_pose(physics=physics)
        return position

    def _get_target_position(self, physics: mjcf.Physics) -> np.ndarray:
        position, _ = self._target.get_pose(physics=physics)
        return position

    def _get_time(self, physics: mjcf.Physics) -> np.ndarray:
        return physics.time()

    def _get_z_angle_to_target(self, physics: mjcf.Physics) -> float:
        morphology_position, morphology_quaternion = self._morphology.get_pose(physics)
        _, _, z_angle = quat2euler(morphology_quaternion)
        # Negative Y-axis is the main axis of locomotion
        z_angle = z_angle - np.pi / 2
        morphology_direction = -1 * np.array([np.cos(z_angle), np.sin(z_angle)])
        morphology_xy = morphology_position[:2]

        target_position, target_quaternion = self._target.get_pose(physics)
        target_xy = target_position[:2]
        direction_to_target = target_xy - morphology_xy
        try:
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        except RuntimeWarning:
            # position will be 0 before init -> ignore this
            pass

        angle_to_target = np.arccos(np.clip(np.dot(morphology_direction, direction_to_target), -1.0, 1.0))
        return angle_to_target

    def _configure_task_observables(self) -> Dict[str, observable.Observable]:
        task_observables = dict()

        task_observables["task/time"] = observable.Generic(self._get_time)

        if self.config.with_target:
            task_observables["task/z_angle_to_target"] = observable.Generic(self._get_z_angle_to_target)

        for obs in task_observables.values():
            obs.enabled = True

        return task_observables

    def _configure_observables(self) -> Dict[str, observable.Observable]:
        self._configure_morphology_observables()
        task_observables = self._configure_task_observables()
        return task_observables

    def get_reward(self, physics: mjcf.Physics) -> float:
        distance_delta = self._calculate_distance_delta(physics)
        if self.config.with_target:
            distance_delta *= -1  # we want to get closer to the target
        return distance_delta

    def _initialize_morphology_pose(self, physics: mjcf.Physics) -> None:
        disc_height = self._morphology.morphology_specification.disc_specification.height.value
        initial_position = np.array([0.0, 0.0, disc_height / 2])
        initial_quaternion = euler2quat(0, 0, 0)

        self._morphology.set_pose(physics=physics,
                                  position=initial_position,
                                  quaternion=initial_quaternion)

    def _initialize_target_pose(self, physics: mjcf.Physics) -> None:
        # Random position at distance 5
        target_distance_from_origin = 5
        angle = np.random.uniform(-np.pi, np.pi)

        initial_position = target_distance_from_origin * np.array([np.cos(angle), np.sin(angle), 0.0])
        initial_quaternion = euler2quat(0, 0, 0)

        self._target.set_pose(physics=physics,
                              position=initial_position,
                              quaternion=initial_quaternion)

    def initialize_episode(self, physics: mjcf.Physics, random_state: np.random.RandomState) -> None:
        self._initialize_morphology_pose(physics=physics)
        self._initialize_target_pose(physics=physics)
        self._previous_distance = self._calculate_distance(physics)

    def _calculate_distance_delta(self, physics: mjcf.Physics) -> float:
        current_distance = self._calculate_distance(physics=physics)
        distance_delta = current_distance - self._previous_distance
        self._previous_distance = current_distance
        return distance_delta

    def _calculate_xy_distance_from_origin(self, physics: mjcf.Physics) -> float:
        position = self._get_morphology_position(physics=physics)
        position_in_xy_plane = position[:2]
        distance = np.linalg.norm(position_in_xy_plane)
        return distance

    def _calculate_distance_to_target(self, physics: mjcf.Physics) -> float:
        morphology_position = self._get_morphology_position(physics)[:2]
        target_position = self._get_target_position(physics)[:2]
        distance = np.linalg.norm(target_position - morphology_position)
        return distance

    def _calculate_distance(self, physics: mjcf.Physics) -> float:
        if self.config.with_target:
            return self._calculate_distance_to_target(physics=physics)
        else:
            return self._calculate_xy_distance_from_origin(physics=physics)


@dataclass
class LocomotionEnvironmentConfig(MJCEnvironmentConfig):
    with_target: bool = False

    @property
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task]:
        return LocomotionTask

    @property
    def simulation_time(self) -> float:
        return 10.0

    @property
    def num_substeps(self) -> int:
        return 20

    @property
    def time_scale(self) -> float:
        return 1.0

    @property
    def physics_time_delta(self) -> float:
        return 0.002
