import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import csv
from datetime import datetime


@dataclass
class TelemetryFrame:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    motor_rpms: np.ndarray
    reward: float = 0.0
    action: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'timestamp': self.timestamp,
            'pos_x': self.position[0],
            'pos_y': self.position[1],
            'pos_z': self.position[2],
            'vel_x': self.velocity[0],
            'vel_y': self.velocity[1],
            'vel_z': self.velocity[2],
            'quat_w': self.orientation[0],
            'quat_x': self.orientation[1],
            'quat_y': self.orientation[2],
            'quat_z': self.orientation[3],
            'ang_vel_x': self.angular_velocity[0],
            'ang_vel_y': self.angular_velocity[1],
            'ang_vel_z': self.angular_velocity[2],
            'rpm_0': self.motor_rpms[0],
            'rpm_1': self.motor_rpms[1],
            'rpm_2': self.motor_rpms[2],
            'rpm_3': self.motor_rpms[3],
            'reward': self.reward
        }
        
        if self.action is not None:
            for i, a in enumerate(self.action):
                d[f'action_{i}'] = a
        
        if self.target is not None:
            d['target_x'] = self.target[0]
            d['target_y'] = self.target[1]
            d['target_z'] = self.target[2]
        
        return d
    
    def to_unity_dict(self, scale: float = 1.0) -> Dict[str, Any]:
        pos_unity = np.array([
            -self.position[1] * scale,
            self.position[2] * scale,
            self.position[0] * scale
        ])
        
        quat_unity = np.array([
            self.orientation[0],
            -self.orientation[2],
            self.orientation[3],
            self.orientation[1]
        ])
        
        d = {
            'timestamp': self.timestamp,
            'pos_x': pos_unity[0],
            'pos_y': pos_unity[1],
            'pos_z': pos_unity[2],
            'rot_w': quat_unity[0],
            'rot_x': quat_unity[1],
            'rot_y': quat_unity[2],
            'rot_z': quat_unity[3],
            'rpm_0': self.motor_rpms[0],
            'rpm_1': self.motor_rpms[1],
            'rpm_2': self.motor_rpms[2],
            'rpm_3': self.motor_rpms[3]
        }
        
        if self.target is not None:
            target_unity = np.array([
                -self.target[1] * scale,
                self.target[2] * scale,
                self.target[0] * scale
            ])
            d['target_x'] = target_unity[0]
            d['target_y'] = target_unity[1]
            d['target_z'] = target_unity[2]
        
        return d


@dataclass
class CSVLoggerConfig:
    output_dir: str = 'logs'
    prefix: str = 'replay'
    unity_format: bool = True
    unity_scale: float = 1.0
    include_actions: bool = True
    include_rewards: bool = True
    float_precision: int = 6


class CSVLogger:
    def __init__(self, config: Optional[CSVLoggerConfig] = None):
        self.config = config or CSVLoggerConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._frames: List[TelemetryFrame] = []
        self._current_file: Optional[str] = None
        self._episode_count = 0
    
    def start_episode(self, episode_id: Optional[int] = None):
        self._frames = []
        if episode_id is not None:
            self._episode_count = episode_id
        else:
            self._episode_count += 1
    
    def log_step(
        self,
        state,
        action: Optional[np.ndarray] = None,
        reward: float = 0.0,
        timestamp: Optional[float] = None,
        target: Optional[np.ndarray] = None
    ):
        if timestamp is None:
            timestamp = len(self._frames) * 0.02
        
        if hasattr(state, 'position'):
            position = np.array([state.position.x, state.position.y, state.position.z])
            velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
            orientation = np.array([
                state.orientation.w, state.orientation.x,
                state.orientation.y, state.orientation.z
            ])
            angular_velocity = np.array([
                state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z
            ])
            motor_rpms = np.array(list(state.motor_rpm) if hasattr(state, 'motor_rpm') else [0, 0, 0, 0])
        else:
            position = np.zeros(3)
            velocity = np.zeros(3)
            orientation = np.array([1, 0, 0, 0])
            angular_velocity = np.zeros(3)
            motor_rpms = np.zeros(4)
        
        frame = TelemetryFrame(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            motor_rpms=motor_rpms,
            reward=reward,
            action=action,
            target=target
        )
        
        self._frames.append(frame)
    
    def save(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.config.prefix}_{timestamp}_ep{self._episode_count}.csv'
        
        filepath = self.output_dir / filename
        
        if not self._frames:
            return str(filepath)
        
        if self.config.unity_format:
            rows = [f.to_unity_dict(self.config.unity_scale) for f in self._frames]
        else:
            rows = [f.to_dict() for f in self._frames]
        
        fieldnames = list(rows[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                formatted_row = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        formatted_row[k] = round(v, self.config.float_precision)
                    else:
                        formatted_row[k] = v
                writer.writerow(formatted_row)
        
        self._current_file = str(filepath)
        return self._current_file
    
    def get_frames(self) -> List[TelemetryFrame]:
        return self._frames.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self._frames:
            return {}
        
        positions = np.array([f.position for f in self._frames])
        velocities = np.array([f.velocity for f in self._frames])
        rewards = np.array([f.reward for f in self._frames])
        
        return {
            'num_frames': len(self._frames),
            'duration': self._frames[-1].timestamp - self._frames[0].timestamp,
            'total_reward': np.sum(rewards),
            'mean_reward': np.mean(rewards),
            'position_range': {
                'x': (positions[:, 0].min(), positions[:, 0].max()),
                'y': (positions[:, 1].min(), positions[:, 1].max()),
                'z': (positions[:, 2].min(), positions[:, 2].max())
            },
            'max_speed': np.max(np.linalg.norm(velocities, axis=1)),
            'mean_speed': np.mean(np.linalg.norm(velocities, axis=1))
        }


class BinaryLogger:
    HEADER_MAGIC = b'HXDR'
    VERSION = 1
    
    def __init__(self, output_dir: str = 'logs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: List[TelemetryFrame] = []
    
    def start_episode(self):
        self._frames = []
    
    def log_step(
        self,
        state,
        action: Optional[np.ndarray] = None,
        reward: float = 0.0,
        timestamp: Optional[float] = None
    ):
        if timestamp is None:
            timestamp = len(self._frames) * 0.02
        
        position = np.array([state.position.x, state.position.y, state.position.z])
        velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        orientation = np.array([
            state.orientation.w, state.orientation.x,
            state.orientation.y, state.orientation.z
        ])
        angular_velocity = np.array([
            state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z
        ])
        motor_rpms = np.array(list(state.motor_rpm) if hasattr(state, 'motor_rpm') else [0, 0, 0, 0])
        
        frame = TelemetryFrame(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            motor_rpms=motor_rpms,
            reward=reward,
            action=action
        )
        
        self._frames.append(frame)
    
    def save(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'replay_{timestamp}.hxdr'
        
        filepath = self.output_dir / filename
        
        timestamps = np.array([f.timestamp for f in self._frames], dtype=np.float32)
        positions = np.array([f.position for f in self._frames], dtype=np.float32)
        velocities = np.array([f.velocity for f in self._frames], dtype=np.float32)
        orientations = np.array([f.orientation for f in self._frames], dtype=np.float32)
        angular_vels = np.array([f.angular_velocity for f in self._frames], dtype=np.float32)
        motor_rpms = np.array([f.motor_rpms for f in self._frames], dtype=np.float32)
        rewards = np.array([f.reward for f in self._frames], dtype=np.float32)
        
        with open(filepath, 'wb') as f:
            f.write(self.HEADER_MAGIC)
            f.write(np.array([self.VERSION], dtype=np.uint32).tobytes())
            f.write(np.array([len(self._frames)], dtype=np.uint32).tobytes())
            
            f.write(timestamps.tobytes())
            f.write(positions.tobytes())
            f.write(velocities.tobytes())
            f.write(orientations.tobytes())
            f.write(angular_vels.tobytes())
            f.write(motor_rpms.tobytes())
            f.write(rewards.tobytes())
        
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> List[TelemetryFrame]:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != cls.HEADER_MAGIC:
                raise ValueError("Invalid file format")
            
            version = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            num_frames = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            timestamps = np.frombuffer(f.read(num_frames * 4), dtype=np.float32)
            positions = np.frombuffer(f.read(num_frames * 12), dtype=np.float32).reshape(-1, 3)
            velocities = np.frombuffer(f.read(num_frames * 12), dtype=np.float32).reshape(-1, 3)
            orientations = np.frombuffer(f.read(num_frames * 16), dtype=np.float32).reshape(-1, 4)
            angular_vels = np.frombuffer(f.read(num_frames * 12), dtype=np.float32).reshape(-1, 3)
            motor_rpms = np.frombuffer(f.read(num_frames * 16), dtype=np.float32).reshape(-1, 4)
            rewards = np.frombuffer(f.read(num_frames * 4), dtype=np.float32)
        
        frames = []
        for i in range(num_frames):
            frames.append(TelemetryFrame(
                timestamp=timestamps[i],
                position=positions[i],
                velocity=velocities[i],
                orientation=orientations[i],
                angular_velocity=angular_vels[i],
                motor_rpms=motor_rpms[i],
                reward=rewards[i]
            ))
        
        return frames
