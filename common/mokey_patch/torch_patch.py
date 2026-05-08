import io
import torch
import threading
from pathlib import Path
from capture_core.AnnotationIO.aes_cipher import AesCipher


_origin_load = torch.load

lock = threading.Lock()


def compare_torch_version(current_version_str, some_version_str):
    """
    Compare the current torch version with the specified version size relationship.
    If it is greater than or equal to the specified version, return True.
    Otherwise, return False.
    """
    current_version_parts = current_version_str.split('+')[0].split('.')
    some_version_parts = some_version_str.split('+')[0].split('.')
    for (current_version_part, some_version_part) in zip(current_version_parts, some_version_parts):
        if int(current_version_part) < int(some_version_part):
            return False
        elif int(current_version_part) > int(some_version_part):
            return True

    return True


def _modified_load_v1(f, map_location=None):
    """The torch version is less than 1.13.0"""
    if isinstance(f, (str, Path)) and Path(f).exists():
        try:
            with lock:
                model_data = AesCipher.decipher_model(f)
            _BytesIO_data = io.BytesIO(model_data)
            return _origin_load(_BytesIO_data, map_location)
        except Exception:
            return _origin_load(f, map_location)
    elif isinstance(f, io.BytesIO):
        return _origin_load(f, map_location)

    return None


def _modified_load_v2(f, map_location=None, weights_only=False):
    """The torch version is 1.13.0 or higher."""
    if isinstance(f, (str, Path)) and Path(f).exists():
        try:
            with lock:
                model_data = AesCipher.decipher_model(f)
            _BytesIO_data = io.BytesIO(model_data)
            return _origin_load(_BytesIO_data, map_location, weights_only=weights_only)
        except Exception:
            return _origin_load(f, map_location, weights_only=weights_only)
    elif isinstance(f, io.BytesIO):
        return _origin_load(f, map_location, weights_only=weights_only)

    return None


def _modified_load_v3(f, map_location=None, weights_only=False, mmap=None):
    """The torch version is 2.1.0 or higher."""
    if isinstance(f, (str, Path)) and Path(f).exists():
        try:
            with lock:
                model_data = AesCipher.decipher_model(f)
            _BytesIO_data = io.BytesIO(model_data)
            return _origin_load(_BytesIO_data, map_location, weights_only=weights_only, mmap=mmap)
        except Exception:
            return _origin_load(f, map_location, weights_only=weights_only, mmap=mmap)
    elif isinstance(f, io.BytesIO):
        return _origin_load(f, map_location, weights_only=weights_only, mmap=mmap)

    return None


def _apply_monkey_patch4torch():
    current_torch_version = torch.__version__
    if not compare_torch_version(current_torch_version, "1.13.0"):
        torch.load = _modified_load_v1
    elif not compare_torch_version(current_torch_version, "2.1.0"):
        torch.load = _modified_load_v2
    else:
        torch.load = _modified_load_v3
