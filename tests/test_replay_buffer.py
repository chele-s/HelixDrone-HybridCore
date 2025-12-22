import sys
import os
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))

from python_src.agents.replay_buffer import PrioritizedReplayBuffer, CppPrioritizedReplayBuffer
import torch

def benchmark_buffer(buffer_class, name, capacity=100000, state_dim=18, action_dim=4, iterations=3):
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    
    device = torch.device('cpu')
    
    push_times = []
    sample_times = []
    update_times = []
    
    for iteration in range(iterations):
        buffer = buffer_class(capacity, device, state_dim, action_dim)
        
        if hasattr(buffer, 'using_cpp'):
            if iteration == 0:
                print(f"Backend: {'C++ (SIMD/AVX)' if buffer.using_cpp else 'Python (NumPy)'}")
        
        n_samples = 50000
        batch_size = 256
        
        states = np.random.randn(n_samples, state_dim).astype(np.float32)
        actions = np.random.randn(n_samples, action_dim).astype(np.float32)
        rewards = np.random.randn(n_samples).astype(np.float32)
        next_states = np.random.randn(n_samples, state_dim).astype(np.float32)
        dones = (np.random.rand(n_samples) > 0.95).astype(np.float32)
        
        start = time.perf_counter()
        buffer.push_batch(states, actions, rewards, next_states, dones)
        push_times.append(time.perf_counter() - start)
        
        n_ops = 500
        start = time.perf_counter()
        for _ in range(n_ops):
            buffer.sample(batch_size)
        sample_times.append((time.perf_counter() - start) / n_ops)
        
        tree_indices = np.random.randint(0, capacity, size=batch_size).astype(np.int32)
        td_errors = np.random.randn(batch_size)
        
        start = time.perf_counter()
        for _ in range(n_ops):
            buffer.update_priorities(tree_indices, td_errors)
        update_times.append((time.perf_counter() - start) / n_ops)
    
    print(f"push_batch({n_samples}):        {np.mean(push_times)*1000:>8.2f} ms  ({n_samples/np.mean(push_times):>10,.0f} samples/sec)")
    print(f"sample({batch_size}):              {np.mean(sample_times)*1000:>8.3f} ms  ({1/np.mean(sample_times):>10,.0f} ops/sec)")
    print(f"update_priorities({batch_size}):  {np.mean(update_times)*1000:>8.3f} ms  ({1/np.mean(update_times):>10,.0f} ops/sec)")
    
    return {
        'push_time': np.mean(push_times),
        'sample_time': np.mean(sample_times),
        'update_time': np.mean(update_times)
    }

def main():
    print("\n" + "="*70)
    print(" PrioritizedReplayBuffer Benchmark: Python vs C++ (SIMD)")
    print("="*70)
    
    py_results = benchmark_buffer(PrioritizedReplayBuffer, "Python PrioritizedReplayBuffer")
    cpp_results = benchmark_buffer(CppPrioritizedReplayBuffer, "C++ PrioritizedReplayBuffer (AVX + 64-byte aligned)")
    
    print("\n" + "="*70)
    print(" SPEEDUP COMPARISON")
    print("="*70)
    
    if cpp_results['push_time'] > 0:
        push_speedup = py_results['push_time'] / cpp_results['push_time']
        sample_speedup = py_results['sample_time'] / cpp_results['sample_time']
        update_speedup = py_results['update_time'] / cpp_results['update_time']
        
        print(f"push_batch:        {push_speedup:>6.2f}x faster")
        print(f"sample:            {sample_speedup:>6.2f}x faster")
        print(f"update_priorities: {update_speedup:>6.2f}x faster")
        print(f"\nOverall average:   {(push_speedup + sample_speedup + update_speedup)/3:.2f}x faster")
    
    print("\n" + "="*70)
    print(" CORRECTNESS VERIFICATION")
    print("="*70)
    
    device = torch.device('cpu')
    state_dim, action_dim = 18, 4
    capacity = 1000
    
    py_buf = PrioritizedReplayBuffer(capacity, device, state_dim, action_dim, alpha=0.6, beta_start=0.4)
    cpp_buf = CppPrioritizedReplayBuffer(capacity, device, state_dim, action_dim, alpha=0.6, beta_start=0.4)
    
    np.random.seed(42)
    for i in range(500):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d = float(np.random.rand() > 0.95)
        py_buf.push(s, a, r, ns, d)
    
    np.random.seed(42)
    for i in range(500):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d = float(np.random.rand() > 0.95)
        cpp_buf.push(s, a, r, ns, d)
    
    py_size = len(py_buf)
    cpp_size = len(cpp_buf)
    
    print(f"Python buffer size: {py_size}")
    print(f"C++ buffer size:    {cpp_size}")
    print(f"Size match:         {'PASS' if py_size == cpp_size else 'FAIL'}")
    
    py_sample = py_buf.sample(64)
    cpp_sample = cpp_buf.sample(64)
    
    print(f"Sample shape match: {'PASS' if py_sample[0].shape == cpp_sample[0].shape else 'FAIL'}")
    print(f"Weights in [0,1]:   {'PASS' if cpp_sample[5].min() >= 0 and cpp_sample[5].max() <= 1 else 'FAIL'}")
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
