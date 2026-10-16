"""The kv4 store/attention tests need a GPU plus triton and vllm (they run in the serving image).
The CPU job (`pytest sly/tests`) runs the format tests only."""
import importlib.util

import pytest

try:
    import torch
    GPU = torch.cuda.is_available() and importlib.util.find_spec("triton") and importlib.util.find_spec("vllm")
except ImportError:
    GPU = False

collect_ignore = [] if GPU else ["test_kv4_kernels.py"]


def pytest_collection_modifyitems(items):
    if GPU:
        return
    skip = pytest.mark.skip(reason="needs GPU + triton + vllm (serving image)")
    for item in items:
        if item.name.startswith(("test_store_", "test_attn_")):
            item.add_marker(skip)
