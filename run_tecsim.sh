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
else
    echo "Running full test suite!"
    ./run_tecsim.sh basic_bang_bang
    ./run_tecsim.sh op_point_current
    ./run_tecsim.sh op_point_voltage
    ./run_tecsim.sh pid_cold
    ./run_tecsim.sh pid_hot
    ./run_tecsim.sh transient
    exit
fi

run_command="./tests/$var"
echo $run_command

$run_command
