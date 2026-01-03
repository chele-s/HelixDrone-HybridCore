import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import io

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class TrajectoryData:
    positions: np.ndarray
    orientations: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    motor_rpms: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None


@dataclass
class VisualizationConfig:
    figsize: Tuple[int, int] = (12, 10)
    dpi: int = 100
    trajectory_color: str = '#2196F3'
    trajectory_alpha: float = 0.8
    trajectory_linewidth: float = 2.0
    target_color: str = '#4CAF50'
    target_size: float = 100
    drone_color: str = '#FF5722'
    drone_size: float = 0.3
    axis_limits: Optional[Tuple[float, float]] = None
    show_grid: bool = True
    show_axes_labels: bool = True
    dark_mode: bool = False
    fps: int = 30
    trail_length: int = 50


class DroneVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
        
        self.config = config or VisualizationConfig()
        self._setup_style()
    
    def _setup_style(self):
        if self.config.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def plot_trajectory_3d(
        self,
        data: TrajectoryData,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        positions = data.positions
        
        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color=self.config.trajectory_color,
            alpha=self.config.trajectory_alpha,
            linewidth=self.config.trajectory_linewidth
        )
        
        ax.scatter(
            positions[0, 0], positions[0, 1], positions[0, 2],
            color='green', s=100, marker='o', label='Start'
        )
        ax.scatter(
            positions[-1, 0], positions[-1, 1], positions[-1, 2],
            color='red', s=100, marker='s', label='End'
        )
        
        if data.target is not None:
            target = np.asarray(data.target)
            if target.ndim == 2:
                ax.plot(
                    target[:, 0], target[:, 1], target[:, 2],
                    color=self.config.target_color,
                    alpha=0.5,
                    linewidth=1.5,
                    linestyle='--',
                    label='Target Trajectory'
                )
            else:
                ax.scatter(
                    target[0], target[1], target[2],
                    color=self.config.target_color,
                    s=self.config.target_size,
                    marker='*',
                    label='Target'
                )
        
        self._set_axis_properties(ax, positions)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_trajectory_2d(
        self,
        data: TrajectoryData,
        projection: str = 'xy',
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.config.dpi)
        
        projections = [('xy', 0, 1, 'X', 'Y'), ('xz', 0, 2, 'X', 'Z'), ('yz', 1, 2, 'Y', 'Z')]
        positions = data.positions
        
        for ax, (name, i, j, xlabel, ylabel) in zip(axes, projections):
            ax.plot(
                positions[:, i], positions[:, j],
                color=self.config.trajectory_color,
                alpha=self.config.trajectory_alpha,
                linewidth=self.config.trajectory_linewidth
            )
            
            ax.scatter(positions[0, i], positions[0, j], color='green', s=50, marker='o')
            ax.scatter(positions[-1, i], positions[-1, j], color='red', s=50, marker='s')
            
            if data.target is not None:
                target = np.asarray(data.target)
                if target.ndim == 2:
                    ax.plot(
                        target[:, i], target[:, j],
                        color=self.config.target_color,
                        alpha=0.5,
                        linestyle='--'
                    )
                else:
                    ax.scatter(
                        target[i], target[j],
                        color=self.config.target_color,
                        s=100, marker='*'
                    )
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{name.upper()} Projection')
            ax.grid(self.config.show_grid, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_state_history(
        self,
        data: TrajectoryData,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        num_plots = 2 + (1 if data.velocities is not None else 0) + \
                    (1 if data.motor_rpms is not None else 0) + \
                    (1 if data.rewards is not None else 0)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3*num_plots), dpi=self.config.dpi)
        if num_plots == 1:
            axes = [axes]
        
        t = data.timestamps if data.timestamps is not None else np.arange(len(data.positions))
        idx = 0
        
        ax = axes[idx]
        ax.plot(t, data.positions[:, 0], label='X', color='#F44336')
        ax.plot(t, data.positions[:, 1], label='Y', color='#4CAF50')
        ax.plot(t, data.positions[:, 2], label='Z', color='#2196F3')
        if data.target is not None:
            target = np.asarray(data.target)
            if target.ndim == 2:
                ax.plot(t[:len(target)], target[:, 2], color='#2196F3', linestyle='--', alpha=0.5, label='Target Z')
            else:
                ax.axhline(y=target[2], color='#2196F3', linestyle='--', alpha=0.5)
        ax.set_ylabel('Position (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        idx += 1
        
        if data.orientations is not None:
            ax = axes[idx]
            ax.plot(t, np.rad2deg(data.orientations[:, 0]), label='Roll', color='#F44336')
            ax.plot(t, np.rad2deg(data.orientations[:, 1]), label='Pitch', color='#4CAF50')
            ax.plot(t, np.rad2deg(data.orientations[:, 2]), label='Yaw', color='#2196F3')
            ax.set_ylabel('Orientation (deg)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            idx += 1
        
        if data.velocities is not None:
            ax = axes[idx]
            ax.plot(t, data.velocities[:, 0], label='Vx', color='#F44336')
            ax.plot(t, data.velocities[:, 1], label='Vy', color='#4CAF50')
            ax.plot(t, data.velocities[:, 2], label='Vz', color='#2196F3')
            ax.set_ylabel('Velocity (m/s)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            idx += 1
        
        if data.motor_rpms is not None:
            ax = axes[idx]
            for i in range(4):
                ax.plot(t, data.motor_rpms[:, i], label=f'M{i+1}')
            ax.set_ylabel('Motor RPM')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            idx += 1
        
        if data.rewards is not None:
            ax = axes[idx]
            ax.plot(t, data.rewards, color='#9C27B0')
            ax.fill_between(t, data.rewards, alpha=0.3, color='#9C27B0')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            idx += 1
        
        axes[-1].set_xlabel('Time (s)' if data.timestamps is not None else 'Step')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_animation(
        self,
        data: TrajectoryData,
        save_path: str,
        format: str = 'gif'
    ):
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        positions = data.positions
        self._set_axis_properties(ax, positions)
        
        line, = ax.plot([], [], [], color=self.config.trajectory_color, linewidth=2)
        point, = ax.plot([], [], [], 'o', color=self.config.drone_color, markersize=10)
        
        if data.target is not None:
            target = np.asarray(data.target)
            if target.ndim == 2:
                ax.plot(
                    target[:, 0], target[:, 1], target[:, 2],
                    color=self.config.target_color, linewidth=1.5, linestyle='--', alpha=0.5
                )
            else:
                ax.scatter(
                    target[0], target[1], target[2],
                    color=self.config.target_color, s=200, marker='*'
                )
        
        trail_length = min(self.config.trail_length, len(positions))
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def animate(i):
            start = max(0, i - trail_length)
            line.set_data(positions[start:i+1, 0], positions[start:i+1, 1])
            line.set_3d_properties(positions[start:i+1, 2])
            point.set_data([positions[i, 0]], [positions[i, 1]])
            point.set_3d_properties([positions[i, 2]])
            return line, point
        
        frames = len(positions)
        interval = 1000 // self.config.fps
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=frames, interval=interval, blit=True
        )
        
        if format == 'gif':
            if HAS_PIL:
                anim.save(save_path, writer='pillow', fps=self.config.fps)
            else:
                anim.save(save_path, writer='imagemagick', fps=self.config.fps)
        elif format == 'mp4':
            anim.save(save_path, writer='ffmpeg', fps=self.config.fps)
        
        plt.close(fig)
    
    def _set_axis_properties(self, ax, positions: np.ndarray):
        if self.config.axis_limits:
            limit = self.config.axis_limits[1]
        else:
            max_range = np.max(np.ptp(positions, axis=0)) * 0.6
            limit = max(max_range, 1.0)
            center = np.mean(positions, axis=0)
        
        if self.config.axis_limits:
            ax.set_xlim(self.config.axis_limits)
            ax.set_ylim(self.config.axis_limits)
            ax.set_zlim(self.config.axis_limits)
        else:
            ax.set_xlim(center[0] - limit, center[0] + limit)
            ax.set_ylim(center[1] - limit, center[1] + limit)
            ax.set_zlim(max(0, center[2] - limit), center[2] + limit)
        
        if self.config.show_axes_labels:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')


class TrainingVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
        
        self.config = config or VisualizationConfig()
    
    def plot_training_curves(
        self,
        rewards: List[float],
        losses: Optional[Dict[str, List[float]]] = None,
        eval_rewards: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        num_plots = 1 + (1 if losses else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots), dpi=self.config.dpi)
        
        if num_plots == 1:
            axes = [axes]
        
        ax = axes[0]
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, color='#2196F3')
        
        window = min(100, len(rewards) // 10 + 1)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(rewards)), smoothed, color='#2196F3', linewidth=2)
        
        if eval_rewards:
            eval_x = np.linspace(0, len(rewards)-1, len(eval_rewards))
            ax.scatter(eval_x, eval_rewards, color='#4CAF50', s=50, zorder=5)
            ax.plot(eval_x, eval_rewards, color='#4CAF50', linestyle='--', alpha=0.7, label='Eval')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        if losses:
            ax = axes[1]
            for name, loss_values in losses.items():
                if loss_values:
                    steps = np.arange(len(loss_values))
                    ax.plot(steps, loss_values, label=name, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_reward_distribution(
        self,
        rewards: List[float],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)
        
        ax.hist(rewards, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(rewards), color='#F44336', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax.axvline(np.median(rewards), color='#4CAF50', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
