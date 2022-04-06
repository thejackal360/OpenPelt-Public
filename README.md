![](OpenPelt.png)

[![DOI](https://zenodo.org/badge/468954069.svg)](https://zenodo.org/badge/latestdoi/468954069)

OpenPelt is a faster-than-real-time temperature control simulation library.
OpenPelt contains utilities for developing and verifying temperature control
algorithms as well as a model of a thermoelectric cooler to act as the plant.
OpenPelt also enables exporting simulation results to [Fenics](https://fenicsproject.org/)
to simulate the control system's impact on a three-dimensional heat diffusion 
model. Furthermore, OpenPelt can be used along with Torch (Pytorch) for developing
neural controllers such as neural networks or reinforcement learning algorithms.
Another advantage of OpenPelt is the integration with Fenics. This means that
the results from an OpenPelt simulation can be used in Fenics in order to 
acquire high-fidelity heat diffusion simulations and study the more realistic
models, such as how a heat transfers from a thermoelectic cooler to neural 
tissue (see the file **fenics_heat_eqn** in the tests directory of the current
repository). 


##  Dependencies
These are the dependencies one needs to install and use OpenPelt:
  - numpy >= 1.19.5
  - matplotlib >= 3.3.4
  - cffi >= 1.15.0
  - pyspice >= 1.5
  - torch 1.9.1+cpu

A **requirements.txt** file is also included.


## Install

We have installed and tested OpenPelt only on Linux for the time being. Thus
we provide installation instructions only for Linux platforms. 

### Linux

In order to install OpenPelt on Linux, first you have to install all the necessary
dependencies:
```
$ pip3 (or pip) install -r requirements.txt
```
Then you can clone OpenPelt repository into a local directory on your machine:
```
$ git clone https://github.com/thejackal360/OpenPelt.git
```
Finally you have to install OpenPelt onto your system
```
$ cd OpenPelt/
$ pip3 (or pip) install .
```
If you'd like to try the provided tests you can just run them by executing the
following command from within the OpenPelt directory
```
$ ./run_tecsim name_of_the_test
```
The names of all the available tests are listed [here](https://github.com/thejackal360/OpenPelt#available-tests). 


## Example usage

Here you can find a simple source code that uses the OpenPelt to simulating a
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


## Available tests

All the tests provided by OpenPelt can run by executing
```
$ ./run_tecsim.sh name_of_script
```
where *name_of_script* can be one of the following:
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
    - GCC 9.3.0 and 9.4.0
    - Python 3.8.10
    - x86_64 64bit

Note: libngspice.so compiled using GCC 9.4.0. thejackal360's ngspice fork is the
used as the source: https://sourceforge.net/u/thejackal360/ngspice/ci/master/tree/.

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
