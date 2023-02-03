#!/usr/bin/env python3

import distutils.command.clean
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setuptools

this_dir = os.path.dirname(os.path.abspath(__file__))


if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
else:
    version_txt = os.path.join(this_dir, "version.txt")
    with open(version_txt) as f:
        version = f.readline().strip()


def write_version_file():
    version_path = os.path.join(this_dir, "epoi", "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')


def setup():
    write_version_file()
    setuptools.setup(
        name="epoi",
        description="EPOI: Efficient PyTorch Operator Inventory.",
        version=version,
        setup_requires=[],
        install_requires=["tabulate", "triton==2.0.0.dev20221202"],
        packages=setuptools.find_packages(),
        url="https://github.com/comaniac/epoi",
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup()
