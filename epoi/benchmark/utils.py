"""Utilities."""
import importlib
import os
import pkg_resources
import subprocess


def is_available(lib_name):
    """Check if the given package is available."""
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False


def get_version_n_commit(lib_name):
    """Get the version or commit hash of the given package."""
    try:
        mod = importlib.import_module(lib_name)
    except ImportError:
        return ("N/A", "N/A")

    if hasattr(mod, "__version__"):
        version = mod.__version__
    else:
        try:
            version = pkg_resources.get_distribution(lib_name).version
        except Exception:
            version = "N/A"

    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
        cmd = ["git", "rev-parse", "HEAD"]
        commit = (
            subprocess.check_output(cmd, cwd=root_dir, stderr=open(os.devnull, "wb"))
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pylint: disable=broad-except
        commit = "N/A"

    return (version, commit)
