import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, List
import gymnasium as gym


class FrameStack:
    def __init__(self, env, num_stack: int = 3):
        self._env = env
        self.num_stack = num_stack
        
        self._is_vectorized = getattr(env, 'is_vectorized', False) or hasattr(env, 'num_envs')
        self._num_envs = getattr(env, 'num_envs', 1) if self._is_vectorized else 1
        
        base_shape = env.observation_space.shape[0]
        self._base_shape = base_shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(base_shape * num_stack,), 
            dtype=np.float32
        )
        self.action_space = env.action_space
        
        if self._is_vectorized:
            self._frames_buffer: List[deque] = [deque(maxlen=num_stack) for _ in range(self._num_envs)]
        else:
            self._frames_buffer = deque(maxlen=num_stack)
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self._env.reset(**kwargs)
        
        if self._is_vectorized:
            for i in range(self._num_envs):
                self._frames_buffer[i].clear()
                for _ in range(self.num_stack):
                    self._frames_buffer[i].append(obs[i].copy())
        else:
            self._frames_buffer.clear()
            for _ in range(self.num_stack):
                self._frames_buffer.append(obs.copy())
        
        return self._get_observation(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._is_vectorized:
            for i in range(self._num_envs):
                if isinstance(terminated, np.ndarray) and (terminated[i] or truncated[i]):
                    self._frames_buffer[i].clear()
                    for _ in range(self.num_stack):
                        self._frames_buffer[i].append(obs[i].copy())
                else:
                    self._frames_buffer[i].append(obs[i].copy())
        else:
            self._frames_buffer.append(obs.copy())
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        if self._is_vectorized:
            stacked_obs = []
            for i in range(self._num_envs):
                frames_list = list(self._frames_buffer[i])
                stacked = np.concatenate(frames_list, axis=0)
                stacked_obs.append(stacked)
            return np.array(stacked_obs, dtype=np.float32)
        else:
            stacked = np.concatenate(list(self._frames_buffer), axis=0)
            return stacked.astype(np.float32)


class DeltaObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, include_velocity: bool = True):
        super().__init__(env)
        self.include_velocity = include_velocity
        self.prev_obs = None
        
        base_dim = self.observation_space.shape[0]
        new_dim = base_dim * 2 if include_velocity else base_dim + base_dim
        
        low = np.full(new_dim, -np.inf, dtype=np.float32)
        high = np.full(new_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs.copy()
        return self._augment_obs(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented = self._augment_obs(obs)
        self.prev_obs = obs.copy()
        return augmented, reward, terminated, truncated, info
    
    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        delta = obs - self.prev_obs if self.prev_obs is not None else np.zeros_like(obs)
        return np.concatenate([obs, delta], axis=0).astype(np.float32)
