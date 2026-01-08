import sys
sys.path.insert(0, 'build/Release')

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig
import numpy as np

print("=== Prueba SOTAActuator ===")

config = EnvConfig(use_sota_actuator=True)
env = QuadrotorEnv(config=config)

obs, info = env.reset()
print(f"Estado inicial: pos={obs[:3]*5}, vel={obs[3:6]*5}")

action = np.array([0.1, 0.0, 0.0, 0.0])

for i in range(10):
    obs, reward, done, trunc, info = env.step(action)
    pos_z = obs[2] * 5
    print(f"Step {i+1}: reward={reward:.2f}, z={pos_z:.2f}m, done={done}")
    if done:
        break

print("\nSOTAActuator integrado correctamente!")
