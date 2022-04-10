#!/usr/bin/env bash

if [ "$1" == "basic_bang_bang" ]
then
    var="./tests/basic_bang_bang.py"
elif [ "$1" == "op_point_current" ]
then
    var="./tests/op_point_current.py"
elif [ "$1" == "op_point_voltage" ]
then
    var="./tests/op_point_voltage.py"
elif [ "$1" == "pid_cold" ]
then
    var="./tests/pid_cold.py"
elif [ "$1" == "pid_hot" ]
then
    var="./tests/pid_hot.py"
elif [ "$1" == "transient" ]
then
    var="./tests/transient.py"
else
    echo "Running full test suite!"
    var="./tests/run_all.py"
fi

run_command="pytest $var"
echo $run_command

LD_LIBRARY_PATH=$(pwd)/thejackal360-ngspice/lib $run_command
