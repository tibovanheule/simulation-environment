from dm_control import composer, mjcf

from erpy.utils import colors


class Target(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""

    def _build(self):
        self._mjcf_model = mjcf.RootElement()
        self._geom = self._mjcf_model.worldbody.add(
            'geom', type='sphere',
            size=[0.2],
            rgba=colors.rgba_red)

    @property
    def mjcf_model(self):
        return self._mjcf_model
