#!/usr/bin/env bash

rm -rf build dist

python setup.py build
python setup.py sdist bdist_wheel

echo "Wheel files:"
ls dist/*.whl
