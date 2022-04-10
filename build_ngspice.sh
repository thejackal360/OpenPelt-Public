#!/usr/bin/env sh

cd thejackal360-ngspice/ || exit
./autogen.sh
./configure --with-ngshared --prefix=`pwd`
make -j5
make install
cd - || exit
