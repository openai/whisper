import os
import sys

import pkg_resources
from setuptools import setup, find_packages


def read_version(fname="whisper/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []
if sys.platform.startswith("linux"):
    triton_requirement = "triton>=2.0.0.dev20221202"
    try:
        import re
        import subprocess
        version_line = subprocess.check_output(["nvcc", "--version"]).strip().split(b"\n")[-1]
        major, minor = re.findall(rb"cuda_([\d]+)\.([\d]+)", version_line)[0]
        if (int(major), int(minor)) < (11, 4):
            # the last version supporting CUDA < 11.4
            triton_requirement = "triton==2.0.0.dev20221011"
    except (IndexError, OSError, subprocess.SubprocessError):
        pass
    requirements.append(triton_requirement)

setup(
    name="openai-whisper",
    py_modules=["whisper"],
    version=read_version(),
    description="Robust Speech Recognition via Large-Scale Weak Supervision",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.7",
    author="OpenAI",
    url="https://github.com/openai/whisper",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy"]},
)
