#!/usr/bin/env bash

if [ "$1" == "tecsim" ]
then
    var="bessel.py"
elif [ "$1" == "characterize" ]
then
    var="system_characterization.py"
else
    echo "No script found!"
    exit
fi

run_command="./src/$var"
echo $run_command

LD_LIBRARY_PATH=$(pwd)/lib $run_command
