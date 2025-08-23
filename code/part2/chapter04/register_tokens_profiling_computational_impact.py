"""
Profiling computational impact of register tokens

Extracted from: part2/chapter04/register_tokens.tex
Block: 7
Lines: 67
"""

import time
import torch.profiler

def profile_register_token_impact():
    """Profile computational overhead of register tokens."""
    
    # Models with different register token configurations
    model_configs = [
        {'num_register_tokens': 0, 'name': 'baseline'},
        {'num_register_tokens': 2, 'name': 'reg_2'},
        {'num_register_tokens': 4, 'name': 'reg_4'},
        {'num_register_tokens': 8, 'name': 'reg_8'},
    ]
    
    results = {}
    
    for config in model_configs:
        model = ViTWithRegisterTokens(**config)
        model.eval()
        
        # Warm-up
        dummy_input = torch.randn(32, 3, 224, 224)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Profile
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True
        ) as prof:
            with torch.no_grad():
                for _ in range(100):
                    _ = model(dummy_input)
        
        # Extract timing information
        total_time = sum([event.cpu_time_total for event in prof.events()])
        
        results[config['name']] = {
            'total_time_ms': total_time / 1000,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
    
    return results

def benchmark_inference_speed():
    """Benchmark inference speed with different register configurations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_sizes = [1, 8, 16, 32]
    register_configs = [0, 2, 4, 8]
    
    results = {}
    
    for num_registers in register_configs:
        results[f'reg_{num_registers}'] = {}
        
        model = ViTWithRegisterTokens(num_register_tokens=num_registers).to(device)
        model.eval()
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Warm-up
            for _ in range(20):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(100):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / 100
            throughput = batch_size * 100 / (end_time - start_time)
            
            results[f'reg_{num_registers}'][f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time_ms,
                'throughput_samples_per_sec': throughput
            }
    
    return results