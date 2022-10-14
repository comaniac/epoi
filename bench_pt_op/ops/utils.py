"""Utilities."""
import importlib


def is_available(lib_name):
    """Check if the given package is available."""
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False
