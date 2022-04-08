#!/usr/bin/env sh

cd thejackal360-ngspice/ ; ./autogen.sh ; ./configure --with-ngshared --prefix=`pwd` ; make ; make install ; cd -
