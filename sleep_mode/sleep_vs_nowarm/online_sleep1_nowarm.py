import sys
import subprocess
import os
import time
import requests
import signal
import sys
import atexit
from openai import OpenAI
from typing import Optional, Dict

# Global list to track all child processes
child_processes = []
# Global timing dictionary
timing_data = {}

def terminate_process(process):
    """Safely terminate a process and remove it from tracking"""
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as e:
        print(f"Error terminating process: {e}")
    finally:
        if process in child_processes:
            child_processes.remove(process)

def cleanup_processes():
    """Kill all child processes"""
    for process in child_processes:
        terminate_process(process)

def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_processes()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_processes)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def time_event(event_name: str):
    """Decorator to time events"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            timing_data[event_name] = duration
            print(f"{event_name}: {duration:.2f} seconds")
            return result
        return wrapper
    return decorator

def run_vllm_server(
    model='HuggingFaceTB/SmolLM3-3B',
    count=1,
    port=8001,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    max_num_seqs=2,
    enable_sleep_mode=True,
    vllm_server_dev_mode='1',
    quantization: Optional[str] = None 
):
    env = os.environ.copy()
    env['VLLM_SERVER_DEV_MODE'] = vllm_server_dev_mode

    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'

    safe_cwd = "/tmp"

    
    cmd = [
        # 'nsys', 'profile', '-o', f'report{count}.nsys-rep','--trace-fork-before-exec=true','--cuda-graph-trace=node','--delay', '10','--duration','300', '--force-overwrite', 'true',
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model,
        '--trust-remote-code',
        '--tensor-parallel-size', str(tensor_parallel_size),
        '--max_model_len', str(max_model_len),
        '--max_num_seqs', str(max_num_seqs),
        '--gpu_memory_utilization', str(gpu_memory_utilization),
        '--port', str(port),
        '--no-enable-prefix-caching',
        # '--enforce-eager'
        # '--compilation-config', '{"use_inductor": true, "backend": "inductor", "cudagraph_mode": "NONE", "cudagraph_num_of_warmups": 0}',
        '--compilation-config', '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
    ]

    if quantization:
        cmd.extend(['--quantization', quantization])

    if enable_sleep_mode:
        cmd.append('--enable-sleep-mode')

    debug_import_check(env, safe_cwd)
    process = subprocess.Popen(cmd, env=env, cwd=safe_cwd)
    child_processes.append(process)
    return process


def debug_import_check(env, cwd):
    check = [
        sys.executable, "-c",
        "import vllm,inspect,sys; "
        "print('vllm_version=', vllm.__version__); "
        "print('vllm_file=', inspect.getfile(vllm)); "
        "print('sys_path0=', sys.path[0])"
    ]
    out = subprocess.check_output(check, env=env, cwd=cwd).decode()
    print('=== IMPORT CHECK ===\n' + out + '=====================')

def make_request(method: str, url: str, path: str = "", json_data: Optional[Dict] = None, timeout: int = 30, silent: bool = True):
    full_url = f"{url.rstrip('/')}/{path.lstrip('/')}" if path else url
    try:
        response = requests.request(method.upper(), url=full_url, json=json_data, timeout=timeout)
        return response
    except Exception as e:
        if not silent:
            print(f"Request failed: {e}")
        return None

def wait_for_server_ready(base_url: str, max_wait_time: int = 3000):
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        response = make_request('GET', base_url, '/health', timeout=5, silent=True)
        if response and response.status_code == 200:
            return True
        time.sleep(5)
    print("Server failed to start")
    return False

def prompt_model(base_url: str, model: str, content: str,
                 *, max_tokens: int = 100,
                 min_tokens: int | None = None,
                 temperature=0.0, top_p=1.0, top_k=0,
                 stop: list[str] | None = None,
                 stream: bool = True):
    client = OpenAI(api_key="EMPTY", base_url=base_url + '/v1')
    messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":content}]
    
    extra = {"top_k": top_k, "seed": 0}
    if min_tokens is not None:
        extra["min_tokens"] = min_tokens
    if stop:
        extra["stop"] = stop

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, messages=messages, stream=stream,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        extra_body=extra,
    )

    if stream:
        first = True
        out = []
        for ch in resp:
            delta = ch.choices[0].delta.content or ""
            if first and (delta != ""):
                ttft = time.time() - t0
                print(f"TTFT: {ttft:.3f}s")
                first = False
            out.append(delta)
        reply = "".join(out)
    else:
        reply = resp.choices[0].message.content or ""

    make_request('POST', base_url, '/reset_prefix_cache')
    print(f"Model replied: {reply}")
    return reply


def sleep_model(base_url: str, level:int = 1):
    json_data = {"level": level}
    response = make_request('POST', base_url, '/sleep', json_data=json_data)
    return response and response.status_code == 200

def wake_up_model(base_url: str):
    response = make_request('POST', base_url, '/wake_up')
    return response and response.status_code == 200


def load_model(model_name: str, port: int, enable_sleep_mode: bool, event_prefix: str, count: int, quantization: Optional[str] = None):
    """Unified function to load and test a model"""
    @time_event(f"{event_prefix} load")
    def load_model():
        process = run_vllm_server(model=model_name, port=port, enable_sleep_mode=enable_sleep_mode, count=count, quantization=quantization)
        base_url = f'http://localhost:{port}'
        
        if wait_for_server_ready(base_url):
            return process, base_url
        return None, None
    
    return load_model()

def switch_model_timed(sleep_url: str, wake_url: str, event_name: str):
    @time_event(event_name)
    def switch():
        sleep_model(sleep_url)
        wake_up_model(wake_url)
    return switch()

def sleep_model_timed(sleep_url: str, event_name: str, level: int = 1):
    @time_event(event_name)
    def sleep():
        sleep_model(sleep_url, level)
    return sleep()

def wake_model_timed(wake_url: str, event_name: str):
    @time_event(event_name)
    def wake():
        wake_up_model(wake_url)
    return wake()

def prompt_model_timed(base_url: str, model: str, content: str, event_name: str, **kwargs):
    @time_event(event_name)
    def prompt():
        return prompt_model(base_url, model, content,**kwargs)
    return prompt()

def warmup_model_timed(base_url: str, model: str, event_name: str):
    @time_event(event_name)
    def warm():
        # Use extremely short one-time paths to generate trigger graphs/kernels, etc.
        return prompt_model(base_url, model, "warmup", max_tokens=1, min_tokens=1,
                            temperature=0.0, top_p=1.0, top_k=0, stream=False)
    return warm()

def test_with_sleep_mode(model1, model2, prompt, do_warmup: bool = True):
    prompt1 = "Can you explain why the sky is blue?"
    prompt2 = "Which is bigger? 9.9 or 9.11?"
    prompt3 = "Generate 5 horror stories consisting of 3 words only."
    print("\n=== TESTING WITH SLEEP MODE ===")

    first_model, first_model_url = load_model(model1, 8001, True, 'Sleep mode first model', count=1, quantization=None)

    if first_model and first_model_url:
        if do_warmup:
            # Warm up the first model immediately after loading.
            warmup_model_timed(first_model_url, model1, 'Sleep mode first model warmup')
        else:
            print("Skipping Warm up for first model")

        sleep_model(first_model_url)
        
        second_model, second_model_url = load_model(model2, 8002, True, 'Sleep mode second model', count=2, quantization=None)

        if second_model and second_model_url:
            if do_warmup:
                # Warm up the second model immediately after loading.
                warmup_model_timed(second_model_url, model2, 'Sleep mode first model warmup')
            else:
                print("Skipping Warm up for second model")
            sleep_model(second_model_url)

            wake_model_timed(first_model_url, "Sleep mode first model wake up1")
            prompt_model_timed(first_model_url, model1, prompt1, 'Sleep mode first model prompt1')
            sleep_model_timed(first_model_url, "Sleep mode first model sleep1")
            
            wake_model_timed(second_model_url, "Sleep mode second model wake up1")
            prompt_model_timed(second_model_url, model2, prompt1, 'Sleep mode second model prompt1')
            sleep_model_timed(second_model_url, "Sleep mode second model sleep1")
            
            wake_model_timed(first_model_url, 'Sleep mode first model wake up2')
            prompt_model_timed(first_model_url, model1, prompt2, 'Sleep mode first model prompt2')
            sleep_model_timed(first_model_url, "Sleep mode first model sleep2")
            
            wake_model_timed(second_model_url, 'Sleep mode second model wake up2')
            prompt_model_timed(second_model_url, model2, prompt2, 'Sleep mode second model prompt2')
            sleep_model_timed(second_model_url, "Sleep mode second model sleep2")
            
            wake_model_timed(first_model_url, 'Sleep mode first model wake up3')
            prompt_model_timed(first_model_url, model1, prompt3, 'Sleep mode first model prompt3')
            sleep_model_timed(first_model_url, "Sleep mode first model sleep3")
            
            wake_model_timed(second_model_url, 'Sleep mode second model wake up3')
            prompt_model_timed(second_model_url, model2, prompt3, 'Sleep mode second model prompt3')

            # Cleanup
            terminate_process(second_model)
        
        terminate_process(first_model)
        

# Run in sleep mode, command execution: python online_sleep1_nowarm.py sleep1
# Run in sleep mode nowarmup, command execution: python online_sleep1_nowarm.py nowarmup
def main():
    print("Starting VLLM Server Performance Test")
    model1 = 'Qwen/Qwen3-0.6B'
    # model2 = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    model2 = 'microsoft/phi-3-vision-128k-instruct'
    prompt = "How many R's in Red?"

    test_mode = None
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
    
    try:
        if test_mode == 'sleep1':
            print("=== CONFIG: Running in L1 SLEEP MODE (with Warmup) ===")
            sleep_start = time.time()
            # Call with do_warmup=True
            test_with_sleep_mode(model1, model2, prompt, do_warmup=True)
            sleep_total = time.time() - sleep_start
            timing_data['total_sleep_mode_test'] = sleep_total
        elif test_mode == 'nowarmup':
            print("=== CONFIG: Running in L1 SLEEP MODE (no Warmup) ===")
            sleep_start = time.time()
            # Call with do_warmup=False
            test_with_sleep_mode(model1, model2, prompt, do_warmup=False)
            sleep_total = time.time() - sleep_start
            timing_data['total_sleep_mode_test'] = sleep_total

    
        # Print detailed timing data as JSON
        print(f"\n=== DETAILED TIMING DATA ===")
        for key, value in timing_data.items():
            print(f"{key}\t{value:.2f}")


    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        cleanup_processes()

if __name__ == "__main__":
    main()
