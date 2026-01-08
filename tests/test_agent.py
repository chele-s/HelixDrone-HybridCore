import sys
import os
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))

import numpy as np
import torch


class TestNetworkShapes(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 4
        self.batch_size = 32
        self.device = torch.device('cpu')
    
    def test_actor_output_shape(self):
        from python_src.agents import Actor
        
        actor = Actor(self.state_dim, self.action_dim)
        state = torch.randn(self.batch_size, self.state_dim)
        action = actor(state)
        
        self.assertEqual(action.shape, (self.batch_size, self.action_dim))
    
    def test_actor_single_input(self):
        from python_src.agents import Actor
        
        actor = Actor(self.state_dim, self.action_dim)
        state = torch.randn(1, self.state_dim)
        action = actor(state)
        
        self.assertEqual(action.shape, (1, self.action_dim))
    
    def test_critic_output_shape(self):
        from python_src.agents import Critic
        
        critic = Critic(self.state_dim, self.action_dim)
        state = torch.randn(self.batch_size, self.state_dim)
        action = torch.randn(self.batch_size, self.action_dim)
        q1, q2 = critic(state, action)
        
        self.assertEqual(q1.shape, (self.batch_size, 1))
        self.assertEqual(q2.shape, (self.batch_size, 1))
    
    def test_critic_q1_forward(self):
        from python_src.agents import Critic
        
        critic = Critic(self.state_dim, self.action_dim)
        state = torch.randn(self.batch_size, self.state_dim)
        action = torch.randn(self.batch_size, self.action_dim)
        q1 = critic.q1_forward(state, action)
        
        self.assertEqual(q1.shape, (self.batch_size, 1))
    
    def test_deep_actor_shape(self):
        from python_src.agents import DeepActor
        
        actor = DeepActor(self.state_dim, self.action_dim, num_residual_blocks=2)
        state = torch.randn(self.batch_size, self.state_dim)
        action = actor(state)
        
        self.assertEqual(action.shape, (self.batch_size, self.action_dim))


class TestActionBounds(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 4
        self.device = torch.device('cpu')
    
    def test_actor_output_bounds(self):
        from python_src.agents import Actor
        
        actor = Actor(self.state_dim, self.action_dim, max_action=1.0)
        
        for _ in range(100):
            state = torch.randn(64, self.state_dim)
            action = actor(state)
            
            self.assertTrue(torch.all(action >= -1.0).item())
            self.assertTrue(torch.all(action <= 1.0).item())
    
    def test_actor_custom_max_action(self):
        from python_src.agents import Actor
        
        max_action = 0.5
        actor = Actor(self.state_dim, self.action_dim, max_action=max_action)
        
        state = torch.randn(32, self.state_dim)
        action = actor(state)
        
        self.assertTrue(torch.all(action >= -max_action).item())
        self.assertTrue(torch.all(action <= max_action).item())
    
    def test_td3_agent_action_bounds(self):
        from python_src.agents import TD3Agent
        
        agent = TD3Agent(self.state_dim, self.action_dim, self.device, max_action=1.0)
        
        for _ in range(50):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = agent.get_action(state, add_noise=True)
            
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))
    
    def test_ddpg_agent_action_bounds(self):
        from python_src.agents import DDPGAgent
        
        agent = DDPGAgent(self.state_dim, self.action_dim, self.device, max_action=1.0)
        
        for _ in range(50):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = agent.get_action(state, add_noise=True, noise_scale=0.5)
            
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 4
        self.capacity = 1000
        self.device = torch.device('cpu')
    
    def test_replay_buffer_push_sample(self):
        from python_src.agents import ReplayBuffer
        
        buffer = ReplayBuffer(self.capacity, self.device, self.state_dim, self.action_dim)
        
        for i in range(100):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = float(np.random.randn())
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = i % 10 == 0
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 100)
        
        states, actions, rewards, next_states, dones = buffer.sample(32)
        
        self.assertEqual(states.shape, (32, self.state_dim))
        self.assertEqual(actions.shape, (32, self.action_dim))
        self.assertEqual(rewards.shape, (32, 1))
        self.assertEqual(next_states.shape, (32, self.state_dim))
        self.assertEqual(dones.shape, (32, 1))
    
    def test_prioritized_replay_buffer(self):
        from python_src.agents import PrioritizedReplayBuffer
        
        buffer = PrioritizedReplayBuffer(
            self.capacity, self.device,
            self.state_dim, self.action_dim
        )
        
        for i in range(200):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = float(np.random.randn())
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = False
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 200)
        
        states, actions, rewards, next_states, dones, weights, indices = buffer.sample(32)
        
        self.assertEqual(states.shape, (32, self.state_dim))
        self.assertEqual(weights.shape, (32, 1))
        self.assertEqual(len(indices), 32)
        
        td_errors = np.random.rand(32)
        buffer.update_priorities(indices, td_errors)
    
    def test_replay_buffer_overflow(self):
        from python_src.agents import ReplayBuffer
        
        small_capacity = 50
        buffer = ReplayBuffer(small_capacity, self.device, self.state_dim, self.action_dim)
        
        for i in range(100):
            state = np.ones(self.state_dim).astype(np.float32) * i
            action = np.ones(self.action_dim).astype(np.float32)
            buffer.push(state, action, 0.0, state, False)
        
        self.assertEqual(len(buffer), small_capacity)
    
    def test_nstep_replay_buffer(self):
        from python_src.agents import NStepReplayBuffer
        
        buffer = NStepReplayBuffer(
            self.capacity, self.device,
            n_step=3, gamma=0.99,
            state_dim=self.state_dim, action_dim=self.action_dim
        )
        
        for i in range(50):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = 1.0
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = i == 49
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertGreater(len(buffer), 0)


class TestAgentUpdate(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 4
        self.device = torch.device('cpu')
    
    def test_td3_update(self):
        from python_src.agents import TD3Agent, ReplayBuffer
        
        agent = TD3Agent(self.state_dim, self.action_dim, self.device)
        buffer = ReplayBuffer(1000, self.device, self.state_dim, self.action_dim)
        
        for _ in range(100):
            s = np.random.randn(self.state_dim).astype(np.float32)
            a = np.random.randn(self.action_dim).astype(np.float32)
            r = float(np.random.randn())
            ns = np.random.randn(self.state_dim).astype(np.float32)
            buffer.push(s, a, r, ns, False)
        
        for i in range(10):
            metrics = agent.update(buffer, batch_size=32)
            self.assertIn('critic_loss', metrics)
            self.assertIn('q_value', metrics)
    
    def test_ddpg_update(self):
        from python_src.agents import DDPGAgent, ReplayBuffer
        
        agent = DDPGAgent(self.state_dim, self.action_dim, self.device)
        buffer = ReplayBuffer(1000, self.device, self.state_dim, self.action_dim)
        
        for _ in range(100):
            s = np.random.randn(self.state_dim).astype(np.float32)
            a = np.random.randn(self.action_dim).astype(np.float32)
            r = float(np.random.randn())
            ns = np.random.randn(self.state_dim).astype(np.float32)
            buffer.push(s, a, r, ns, False)
        
        metrics = agent.update(buffer, batch_size=32)
        
        self.assertIn('actor_loss', metrics)
        self.assertIn('critic_loss', metrics)
    
    def test_update_with_per(self):
        from python_src.agents import TD3Agent, PrioritizedReplayBuffer
        
        agent = TD3Agent(self.state_dim, self.action_dim, self.device)
        buffer = PrioritizedReplayBuffer(
            1000, self.device, self.state_dim, self.action_dim
        )
        
        for _ in range(100):
            s = np.random.randn(self.state_dim).astype(np.float32)
            a = np.random.randn(self.action_dim).astype(np.float32)
            r = float(np.random.randn())
            ns = np.random.randn(self.state_dim).astype(np.float32)
            buffer.push(s, a, r, ns, False)
        
        for _ in range(5):
            metrics = agent.update(buffer, batch_size=32)
            self.assertIn('critic_loss', metrics)


class TestGPU(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 4
        self.cuda_available = torch.cuda.is_available()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_actor_to_cuda(self):
        from python_src.agents import Actor
        
        device = torch.device('cuda')
        actor = Actor(self.state_dim, self.action_dim).to(device)
        
        state = torch.randn(32, self.state_dim, device=device)
        action = actor(state)
        
        self.assertEqual(action.device.type, 'cuda')
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_critic_to_cuda(self):
        from python_src.agents import Critic
        
        device = torch.device('cuda')
        critic = Critic(self.state_dim, self.action_dim).to(device)
        
        state = torch.randn(32, self.state_dim, device=device)
        action = torch.randn(32, self.action_dim, device=device)
        q1, q2 = critic(state, action)
        
        self.assertEqual(q1.device.type, 'cuda')
        self.assertEqual(q2.device.type, 'cuda')
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_agent_cuda(self):
        from python_src.agents import TD3Agent, ReplayBuffer
        
        device = torch.device('cuda')
        agent = TD3Agent(self.state_dim, self.action_dim, device)
        buffer = ReplayBuffer(100, device, self.state_dim, self.action_dim)
        
        for _ in range(50):
            s = np.random.randn(self.state_dim).astype(np.float32)
            a = np.random.randn(self.action_dim).astype(np.float32)
            buffer.push(s, a, 0.0, s, False)
        
        metrics = agent.update(buffer, batch_size=16)
        self.assertIsNotNone(metrics)


class TestNoise(unittest.TestCase):
    def setUp(self):
        self.action_dim = 4
    
    def test_ou_noise_shape(self):
        from python_src.agents import OUNoise
        
        noise = OUNoise(self.action_dim)
        
        for _ in range(10):
            n = noise.sample()
            self.assertEqual(n.shape, (self.action_dim,))
    
    def test_ou_noise_reset(self):
        from python_src.agents import OUNoise
        
        noise = OUNoise(self.action_dim, mu=0.0)
        
        for _ in range(100):
            noise.sample()
        
        noise.reset()
        self.assertTrue(np.allclose(noise.state, 0.0))
    
    def test_gaussian_noise(self):
        from python_src.agents import GaussianNoise
        
        noise = GaussianNoise(self.action_dim, sigma=0.1)
        
        samples = [noise.sample() for _ in range(1000)]
        samples = np.array(samples)
        
        self.assertEqual(samples.shape, (1000, self.action_dim))
        self.assertLess(abs(samples.mean()), 0.1)


class TestEnvironment(unittest.TestCase):
    def test_env_reset(self):
        from python_src.envs import QuadrotorEnv, EnvConfig
        
        env = QuadrotorEnv(config=EnvConfig())
        obs, info = env.reset()
        
        self.assertEqual(obs.shape, (52,))
        self.assertIn('position', info)
        self.assertIn('target', info)
    
    def test_env_step(self):
        from python_src.envs import QuadrotorEnv, EnvConfig
        
        env = QuadrotorEnv(config=EnvConfig())
        env.reset()
        
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (52,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
    
    def test_env_spaces(self):
        from python_src.envs import QuadrotorEnv, EnvConfig
        
        env = QuadrotorEnv(config=EnvConfig())
        
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(env.observation_space.shape, (52,))
        
        self.assertEqual(env.action_space.low.tolist(), [-1.0] * 4)
        self.assertEqual(env.action_space.high.tolist(), [1.0] * 4)


class TestMathUtils(unittest.TestCase):
    def test_quaternion_euler_conversion(self):
        from python_src.utils.helix_math import Quaternion
        
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = Quaternion.from_euler(roll, pitch, yaw)
        r2, p2, y2 = q.to_euler()
        
        self.assertAlmostEqual(roll, r2, places=5)
        self.assertAlmostEqual(pitch, p2, places=5)
        self.assertAlmostEqual(yaw, y2, places=5)
    
    def test_normalizer(self):
        from python_src.utils.helix_math import Normalizer, Bounds
        
        bounds = Bounds(low=np.array([0, 0]), high=np.array([10, 20]))
        normalizer = Normalizer(bounds, target_range=(-1, 1))
        
        x = np.array([5, 10])
        normalized = normalizer.normalize(x)
        
        self.assertAlmostEqual(normalized[0], 0.0, places=5)
        self.assertAlmostEqual(normalized[1], 0.0, places=5)
        
        denormalized = normalizer.denormalize(normalized)
        np.testing.assert_array_almost_equal(x, denormalized)
    
    def test_euclidean_distance(self):
        from python_src.utils.helix_math import euclidean_distance
        
        a = np.array([0, 0, 0])
        b = np.array([3, 4, 0])
        
        self.assertAlmostEqual(euclidean_distance(a, b), 5.0, places=5)
    
    def test_coordinate_transform(self):
        from python_src.utils.helix_math import CoordinateTransform
        
        pos_zup = np.array([1, 2, 3])
        pos_yup = CoordinateTransform.zup_to_yup(pos_zup)
        
        self.assertEqual(pos_yup[0], 1)
        self.assertEqual(pos_yup[1], 3)
        self.assertEqual(pos_yup[2], 2)
        
        pos_back = CoordinateTransform.yup_to_zup(pos_yup)
        np.testing.assert_array_equal(pos_zup, pos_back)


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkShapes))
    suite.addTests(loader.loadTestsFromTestCase(TestActionBounds))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentUpdate))
    suite.addTests(loader.loadTestsFromTestCase(TestGPU))
    suite.addTests(loader.loadTestsFromTestCase(TestNoise))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestMathUtils))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
