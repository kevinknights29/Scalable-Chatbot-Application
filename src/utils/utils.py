from __future__ import annotations

import functools
import time

import psutil
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        return (result, exec_time)

    return wrapper


def memory_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result, exec_time = func(*args, **kwargs)
            peak_mem = torch.cuda.max_memory_allocated()
            peak_mem_consumption = peak_mem / 1e9
            return peak_mem_consumption, exec_time, result
        else:
            peak_mem_start = psutil.virtual_memory().peak_wset / (1024**3)
            result, exec_time = func(*args, **kwargs)
            peak_mem_end = psutil.virtual_memory().peak_wset / (1024**3)
            peak_mem_consumption = peak_mem_end - peak_mem_start
            return peak_mem_consumption, exec_time, result

    return wrapper


@memory_decorator
@time_decorator
def generate_output(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(input_ids, max_length=500)
    return outputs
