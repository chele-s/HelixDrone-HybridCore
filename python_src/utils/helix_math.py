import numpy as np
from typing import Tuple, Union, Optional, List
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Bounds:
    low: np.ndarray
    high: np.ndarray
    
    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        self.range = self.high - self.low
        self.center = (self.high + self.low) / 2


class Quaternion:
    __slots__ = ('w', 'x', 'y', 'z')
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        half = angle * 0.5
        s = np.sin(half)
        return cls(np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float32)
    
    def to_euler(self) -> Tuple[float, float, float]:
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def to_rotation_matrix(self) -> np.ndarray:
        xx, yy, zz = self.x * self.x, self.y * self.y, self.z * self.z
        xy, xz, yz = self.x * self.y, self.x * self.z, self.y * self.z
        wx, wy, wz = self.w * self.x, self.w * self.y, self.w * self.z
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ], dtype=np.float32)
    
    def normalize(self) -> 'Quaternion':
        n = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-12:
            return Quaternion()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        n2 = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n2 < 1e-12:
            return Quaternion()
        return Quaternion(self.w/n2, -self.x/n2, -self.y/n2, -self.z/n2)
    
    def rotate(self, v: np.ndarray) -> np.ndarray:
        qv = np.array([self.x, self.y, self.z])
        uv = np.cross(qv, v)
        uuv = np.cross(qv, uv)
        return v + 2.0 * (self.w * uv + uuv)
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
    
    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        if dot > 0.9995:
            return Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            ).normalize()
        
        theta0 = np.arccos(dot)
        theta = theta0 * t
        
        sin_theta = np.sin(theta)
        sin_theta0 = np.sin(theta0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta0
        s1 = sin_theta / sin_theta0
        
        return Quaternion(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z
        )


class Normalizer:
    def __init__(self, bounds: Bounds, target_range: Tuple[float, float] = (-1.0, 1.0)):
        self.bounds = bounds
        self.target_low = target_range[0]
        self.target_high = target_range[1]
        self.target_range = target_range[1] - target_range[0]
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.bounds.low) / (self.bounds.range + 1e-8)
        return normalized * self.target_range + self.target_low
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.target_low) / self.target_range
        return normalized * self.bounds.range + self.bounds.low


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -clip, clip
        ).astype(np.float32)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def euclidean_distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a)[:2] - np.asarray(b)[:2]
    return float(np.linalg.norm(diff))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-12)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def wrap_angle(angle: float) -> float:
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def angular_distance(a1: float, a2: float) -> float:
    diff = wrap_angle(a2 - a1)
    return abs(diff)


def rotation_matrix_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotation_matrix_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)


def rotation_matrix_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rotation_matrix_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    
    return float(roll), float(pitch), float(yaw)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float32)


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_array(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def exponential_decay(initial: float, decay_rate: float, step: int) -> float:
    return initial * (decay_rate ** step)


def linear_schedule(initial: float, final: float, progress: float) -> float:
    return initial + (final - initial) * clamp(progress, 0.0, 1.0)


class CoordinateTransform:
    @staticmethod
    def zup_to_yup(pos: np.ndarray) -> np.ndarray:
        return np.array([pos[0], pos[2], pos[1]], dtype=np.float32)
    
    @staticmethod
    def yup_to_zup(pos: np.ndarray) -> np.ndarray:
        return np.array([pos[0], pos[2], pos[1]], dtype=np.float32)
    
    @staticmethod
    def euler_zup_to_yup(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
        return roll, yaw, pitch
    
    @staticmethod
    def euler_yup_to_zup(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
        return roll, yaw, pitch
    
    @staticmethod
    def quaternion_zup_to_yup(q: Quaternion) -> Quaternion:
        return Quaternion(q.w, q.x, q.z, q.y)
    
    @staticmethod
    def scale_to_unity(pos: np.ndarray, scale: float = 1.0) -> np.ndarray:
        unity_pos = CoordinateTransform.zup_to_yup(pos)
        return unity_pos * scale
    
    @staticmethod
    def rpm_to_normalized(rpm: np.ndarray, min_rpm: float = 0.0, max_rpm: float = 22000.0) -> np.ndarray:
        return (rpm - min_rpm) / (max_rpm - min_rpm)


class TrajectoryUtils:
    @staticmethod
    def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
        if len(positions) < 2:
            return np.zeros_like(positions)
        velocities = np.zeros_like(positions)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        return velocities
    
    @staticmethod
    def compute_acceleration(velocities: np.ndarray, dt: float) -> np.ndarray:
        return TrajectoryUtils.compute_velocity(velocities, dt)
    
    @staticmethod
    def compute_curvature(positions: np.ndarray) -> np.ndarray:
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        curvatures = np.zeros(len(positions))
        for i in range(1, len(positions) - 1):
            p0, p1, p2 = positions[i-1], positions[i], positions[i+1]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                curvatures[i] = norm_cross / (norm_v1 * norm_v2 * (norm_v1 + norm_v2) / 2)
        
        return curvatures
    
    @staticmethod
    def compute_arc_length(positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 0.0
        diffs = np.diff(positions, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    @staticmethod
    def resample_trajectory(positions: np.ndarray, num_points: int) -> np.ndarray:
        arc_lengths = np.zeros(len(positions))
        for i in range(1, len(positions)):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        
        total_length = arc_lengths[-1]
        if total_length < 1e-8:
            return np.tile(positions[0], (num_points, 1))
        
        target_lengths = np.linspace(0, total_length, num_points)
        resampled = np.zeros((num_points, positions.shape[1]))
        
        for i, target in enumerate(target_lengths):
            idx = np.searchsorted(arc_lengths, target)
            if idx == 0:
                resampled[i] = positions[0]
            elif idx >= len(positions):
                resampled[i] = positions[-1]
            else:
                t = (target - arc_lengths[idx-1]) / (arc_lengths[idx] - arc_lengths[idx-1] + 1e-12)
                resampled[i] = lerp_array(positions[idx-1], positions[idx], t)
        
        return resampled.astype(np.float32)


class PIDController:
    def __init__(
        self, 
        kp: float, 
        ki: float, 
        kd: float,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
        integral_limits: Optional[Tuple[float, float]] = None
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits or (-10.0, 10.0)
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def compute(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = clamp(self._integral, *self.integral_limits)
        
        derivative = (error - self._prev_error) / (dt + 1e-12)
        
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = clamp(output, *self.output_limits)
        
        self._prev_error = error
        
        return output


class LowPassFilter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._value = None
    
    def reset(self, value: Optional[np.ndarray] = None):
        self._value = value
    
    def filter(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._value is None:
            self._value = x.copy()
        else:
            self._value = self.alpha * x + (1 - self.alpha) * self._value
        return self._value


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self._x_prev = None
        self._dx_prev = None
    
    def reset(self):
        self._x_prev = None
        self._dx_prev = None
    
    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def filter(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x
        
        dx = (x - self._x_prev) / dt
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._smoothing_factor(dt, cutoff.mean())
        x_hat = alpha * x + (1 - alpha) * self._x_prev
        
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        
        return x_hat
