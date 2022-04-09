#!/usr/bin/env sh

cd thejackal360-ngspice/ || exit;
./autogen.sh;
./configure "--with-ngshared --prefix=$(pwd)";
make;
make install;
cd - || exit
