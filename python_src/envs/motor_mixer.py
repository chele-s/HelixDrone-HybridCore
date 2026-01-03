import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class MixerConfig(IntEnum):
    X_CONFIG = 0
    PLUS_CONFIG = 1


@dataclass
class MotorMixerConfig:
    config_type: MixerConfig = MixerConfig.X_CONFIG
    thrust_scale: float = 1.0
    roll_scale: float = 0.5
    pitch_scale: float = 0.5
    yaw_scale: float = 0.3


class MotorMixer:
    MIXER_MATRICES = {
        MixerConfig.X_CONFIG: np.array([
            [1.0, -1.0, -1.0, -1.0],
            [1.0,  1.0, -1.0,  1.0],
            [1.0,  1.0,  1.0, -1.0],
            [1.0, -1.0,  1.0,  1.0],
        ]),
        MixerConfig.PLUS_CONFIG: np.array([
            [1.0,  0.0, -1.0, -1.0],
            [1.0,  1.0,  0.0,  1.0],
            [1.0,  0.0,  1.0, -1.0],
            [1.0, -1.0,  0.0,  1.0],
        ]),
    }

    def __init__(self, config: MotorMixerConfig = None):
        self.config = config or MotorMixerConfig()
        self._mixer_matrix = self.MIXER_MATRICES[self.config.config_type].copy()
        self._apply_scaling()

    def _apply_scaling(self):
        self._mixer_matrix[:, 0] *= self.config.thrust_scale
        self._mixer_matrix[:, 1] *= self.config.roll_scale
        self._mixer_matrix[:, 2] *= self.config.pitch_scale
        self._mixer_matrix[:, 3] *= self.config.yaw_scale

    def mix(self, commands: np.ndarray) -> np.ndarray:
        motor_outputs = self._mixer_matrix @ commands
        return np.clip(motor_outputs, -1.0, 1.0)

    def mix_components(self, thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return self.mix(np.array([thrust, roll, pitch, yaw]))

    @staticmethod
    def inverse_mix(motor_outputs: np.ndarray, config_type: MixerConfig = MixerConfig.X_CONFIG) -> np.ndarray:
        mixer = MotorMixer.MIXER_MATRICES[config_type]
        return np.linalg.lstsq(mixer, motor_outputs, rcond=None)[0]
