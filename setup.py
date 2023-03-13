import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="whisper/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton==2.0.0")

setup(
    name="openai-whisper",
    py_modules=["whisper"],
    version=read_version(),
    description="Robust Speech Recognition via Large-Scale Weak Supervision",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="OpenAI",
    url="https://github.com/openai/whisper",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
