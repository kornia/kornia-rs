#!/bin/bash -e
set -e -x

cargo clean
rm -rf ./wheelhouse

for PYBIN in /opt/python/cp3[789]*/bin; do
    "${PYBIN}/pip" install maturin -U
    "${PYBIN}/maturin" --version
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

for wheel in ./target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done
