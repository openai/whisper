#! /bin/bash
set -e
rm -rf build dist
echo "Building Source and Wheel (universal) distribution…"
python setup.py sdist bdist_wheel --universal
echo "Uploading the package to PyPI via Twine…"
twine upload dist/* --verbose