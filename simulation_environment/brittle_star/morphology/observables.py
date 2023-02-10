import numpy as np
from dm_control import composer
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf.physics import SynchronizingArrayWrapper
from transforms3d.euler import quat2euler

from erpy.interfaces.mujoco.observables import ConfinedMJCFFeature


def normalizer_factory(original_range: np.ndarray, target_range: np.ndarray = np.array([-1, 1])):
    def normalizer(observations: SynchronizingArrayWrapper, *args, **kwargs) -> np.ndarray:
        data = np.array(observations)

        delta1 = original_range[:, 1] - original_range[:, 0]
        delta2 = target_range[1] - target_range[0]

        return (delta2 * (data - original_range[:, 0]) / delta1) + target_range[0]

    return normalizer


class BrittleStarObservables(composer.Observables):
    @composer.observable
    def end_effector_YZ_direction(self) -> MJCFFeature:
        sensors = self._entity.mjcf_model.find_all('sensor')
        framepos_sensor = list(filter(lambda sensor: sensor.tag == "framepos", sensors))
        end_effector_sensors = list(filter(lambda sensor: "end_effector" in sensor.name, framepos_sensor))

        def normalize_directions(observations: SynchronizingArrayWrapper, *args, **kwargs):
            directions = np.array(observations).reshape(-1, 3)
            for i in range(len(directions)):
                directions[i] = directions[i] / np.linalg.norm(directions[i])

            directions = directions[:, 1:].flatten()
            return directions

        return ConfinedMJCFFeature(low=-1,
                                   high=1,
                                   num_obs_per_element=2,
                                   kind='sensordata',
                                   mjcf_element=end_effector_sensors,
                                   corruptor=normalize_directions)

    @composer.observable
    def orientation(self) -> MJCFFeature:
        sensors = self._entity.mjcf_model.find_all('sensor')
        framequat_sensor = list(filter(lambda sensor: sensor.tag == "framequat", sensors))

        def quaternion_to_normalized_eulers(observations: SynchronizingArrayWrapper, *args, **kwargs) -> np.ndarray:
            data = np.array(observations)
            eulers = np.array(quat2euler(quaternion=data))

            normalized_eulers = eulers / np.pi
            return normalized_eulers

        return ConfinedMJCFFeature(low=-1,
                                   high=1,
                                   num_obs_per_element=3,
                                   kind='sensordata',
                                   mjcf_element=framequat_sensor,
                                   corruptor=quaternion_to_normalized_eulers)

    @composer.observable
    def cartesian_actuator_ctrl(self) -> MJCFFeature:
        position_actuators = self._entity.mjcf_model.find_all('actuator')

        return ConfinedMJCFFeature(low=-np.inf,
                                   high=np.inf,
                                   num_obs_per_element=1,
                                   kind='ctrl',
                                   mjcf_element=position_actuators,
                                   )
