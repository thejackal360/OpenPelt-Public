![](OpenPelt.png)

OpenPelt is a faster-than-real-time temperature control simulation library.
OpenPelt contains utilities for developing and verifying temperature control
algorithms as well as a model of a thermoelectric cooler to act as the plant.
OpenPelt also enables exporting simulation results to [Fenics](https://fenicsproject.org/)
to simulate the control system's impact on a three-dimensional heat diffusion 
model.


##  Dependencies
These are the dependencies one needs to install and use OpenPelt:
  - Numpy >= 1.19.5
  - Matplotlib >= 3.3.4
  - Pytorch 1.9.1+cpu

A **requirements.txt** file is also included.


## Install

We have installed and tested OpenPelt only on Linux for the time being. Thus
we provide installation instructions only for Linux platforms. 

### Linux


## Example usage

Here you can find a simple  source code that uses the OpenPelt to run a
bang-bang controller. 

```
import matplotlib.pyplot as plt
import os
import OpenPelt                     # Import OpenPelt
import numpy

from OpenPelt.controller import bang_bang_controller    # Import the Bang-Bang controller

TEST_NAME = "basic_bang_bang"


if __name__ == "__main__":
    # Define the path where our results will be stored

    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')

    # Instantiate the TEC Plant, the circular buffer and the controller
    pC = OpenPelt.tec_plant("Detector", None, OpenPelt.Signal.VOLTAGE)
    cbs = OpenPelt.circular_buffer_sequencer([50.00, 30.00], pC.get_ncs())
    bbc = bang_bang_controller(cbs)

    # Expose the controller to OpenPelt
    pC.set_controller_f(bbc.controller_f)

    # Run a simulation
    pC.run_sim()

    # Plot the results
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_i_arr()])
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()

```

And you can run the script as
```
$ LD_LIBRARY_PATH=path_to_libngspice python bang_bang.py. 
```

All the tests provided by OpenPelt can run by executing
```
$ ./run_tecsim.sh name_of_script
```
where name of script can be one of the following:

  - **basic_bang_bang** - This is a simple bang-bang controller
  - **op_point_current** - This script characterizes the hot and cold plate temperatures
  at different drive currents.
  - **op_point_voltage** - This script characterizes the hot and cold plate temperatures
  at different drive voltages.
  - **transient** - It characterizes the transient phase of the model.
  - **pid_hot** - A PID controller that controls the temperature of the hot plate.
  - **pid_cold** - Similarly, a PID controller for controlling the temperature of the
  cold plate.
  - **random_hot** - A very naive demo of how to use Pytorch and neural networks with
  OpenPelt.
  - **fenics_heat_eqn** - This script shows how to use Fenics with OpenPelt.
 
## Platforms where OpenPelt has been tested

  - Ubuntu 20.04.4 LTS
    - GCC 9.3.0
    - Python 3.8.10
    - x86_64 64bit

  - Ubuntu 18.04.6 LTS
    - GCC 7.5.0
    - Python 3.6.9
    - x86_64 64bit

## Contributing Guidelines

In case you would like to contribute to OpenPelt, you should use the [Github Pull
Request](https://github.com/thejackal360/OpenPelt/pulls) system. 


## For more help

See the paper, the examples of usage and the source codes in the **tests**
directory.

## Report Bugs

In case you would like to report a bug or you experience any problems with
OpenPelt, you should open an issue using the 
[Githab Issue Tracker](https://github.com/thejackal360/OpenPelt/issues). 


Uses libngspice.so for version 34.
https:// sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/
