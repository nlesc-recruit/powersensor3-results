#!/bin/bash

set -e

git submodule update --init --recursive
pushd fio || exit 1
./configure
make -j
