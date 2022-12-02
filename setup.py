import os

import pkg_resources
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.dirname(__file__)
# The text of the README file
with open(os.path.join(HERE, "README.md"), encoding="utf-8", mode="r") as fd:
    README = fd.read()


setup(
    name="whisper.ai",
    py_modules=["whisper"],
    version="1.0.0.1",
    description="Robust Speech Recognition via Large-Scale Weak Supervision",
    long_description=README,
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.7",
    author="OpenAI",
    url="https://github.com/openai/whisper",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points = {
        'console_scripts': ['whisper=whisper.transcribe:cli'],
    },
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
