import platform
import sys
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="whisper/version.py"):
    try:
        exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
        return locals()["__version__"]
    except FileNotFoundError:
        print(f"Error: {fname} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading version: {e}")
        sys.exit(1)


def parse_requirements(filename):
    try:
        with open(filename) as f:
            return [str(r) for r in pkg_resources.parse_requirements(f)]
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing requirements: {e}")
        sys.exit(1)


requirements_file = Path(__file__).with_name("requirements.txt")

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
    install_requires=parse_requirements(requirements_file),
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
