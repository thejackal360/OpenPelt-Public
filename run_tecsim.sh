#!/usr/bin/env bash

if [ "$1" == "basic_bang_bang" ]
then
    var="basic_bang_bang.py"
elif [ "$1" == "op_point_current" ]
then
    var="op_point_current.py"
elif [ "$1" == "op_point_voltage" ]
then
    var="op_point_voltage.py"
elif [ "$1" == "pid_cold" ]
then
    var="pid_cold.py"
elif [ "$1" == "pid_hot" ]
then
    var="pid_hot.py"
elif [ "$1" == "transient" ]
then
    var="transient.py"
elif [ "$1" == "random_hot" ]
then
    var="random_hot.py"
elif [ "$1" == "fenics_heat_eqn" ]
then
    var="fenics_heat_eqn.py"
else
    echo "No script found!"
    exit
fi

run_command="./tests/$var"
echo $run_command

LD_LIBRARY_PATH=$(pwd)/ngspice $run_command
