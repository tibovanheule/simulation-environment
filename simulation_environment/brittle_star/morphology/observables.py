import numpy as np
from dm_control import composer
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf.physics import SynchronizingArrayWrapper
from scipy.spatial.transform import Rotation

from erpy.interfaces.mujoco.observables import ConfinedMJCFFeature


def normalizer_factory(original_range: np.ndarray, target_range: np.ndarray = np.array([-1, 1])):
    def normalizer(observations: SynchronizingArrayWrapper, *args, **kwargs) -> np.ndarray:
        data = np.array(observations)

        delta1 = original_range[:, 1] - original_range[:, 0]
        delta2 = target_range[1] - target_range[0]

        return (delta2 * (data - original_range[:, 0]) / delta1) + target_range[0]

    return normalizer


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion).as_euler("xyz", degrees=False)


class BrittleStarObservables(composer.Observables):
    @composer.observable
    def position(self) -> MJCFFeature:
        sensors = self._entity.mjcf_model.find_all('sensor')
        framepos_sensor = list(filter(lambda sensor: sensor.tag == "framepos", sensors))

        return ConfinedMJCFFeature(low=-np.inf,
                                   high=np.inf,
                                   num_obs_per_element=3,
                                   kind='sensordata',
                                   mjcf_element=framepos_sensor)

    @composer.observable
    def orientation(self) -> MJCFFeature:
        sensors = self._entity.mjcf_model.find_all('sensor')
        framequat_sensor = list(filter(lambda sensor: sensor.tag == "framequat", sensors))

        def quaternion_to_eulers(observations: SynchronizingArrayWrapper, *args, **kwargs) -> np.ndarray:
            data = np.array(observations)
            eulers = quaternion_to_euler(data)
            return eulers

        return ConfinedMJCFFeature(low=-np.inf,
                                   high=np.inf,
                                   num_obs_per_element=3,
                                   kind='sensordata',
                                   mjcf_element=framequat_sensor,
                                   corruptor=quaternion_to_eulers)

    @composer.observable
    def normalized_tendon_lengths(self) -> MJCFFeature:
        sensors = self._entity.mjcf_model.find_all('sensor')
        tendon_length_sensors = list(filter(lambda sensor: sensor.tag == "tendonpos", sensors))

        original_range = np.array([sensor.tendon.range for sensor in tendon_length_sensors])
        normalizer = normalizer_factory(original_range=original_range)

        return ConfinedMJCFFeature(low=-1,
                                   high=1,
                                   num_obs_per_element=1,
                                   kind='sensordata',
                                   mjcf_element=tendon_length_sensors,
                                   corruptor=normalizer)
