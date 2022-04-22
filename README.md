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

Fist clone the OpenPelt repository into a local directory on your machine:
```
$ git clone --recursive https://github.com/thejackal360/OpenPelt-Public.git
$ cd OpenPelt/
```
Notice that the --recursive flag is necessary to clone the NgSpice submodule. 

In order to install OpenPelt on Linux, first you have to install all the 
necessary dependencies (including bison and flex):
```
$ pip3 (or pip) install -r requirements.txt
$ sudo apt install bison flex (on Ubuntu)

$ sudo dnf install flex-devel bison-devel (on Fedora)
$ sudo pacman -Syu bison flex (on Arch Linux)
```
The next step is to install the NgSpice. This can be done by using the provided
script **build_ngspice.sh**.
```
./build_ngspice
```
and extend *LD_LIBRARY_PATH* environment variable to include the 
libngspise. This can be done by adding the following line to your *.bashrc*
file (if you use a different shell then please consult this
[link](https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables)
on how to permanently add an environment variable.)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH_TO_OpenPelt-Public/thejackal360-ngspice/lib
```
and source the *.bashrc* by executing the following command
```
$ source ~/.bashrc
```
Finally you have to install OpenPelt and its ngspice submodule onto your system
```
$ pip3 (or pip) install .
```
If you'd like to try the provided tests you can just run them by executing the
following command from within the OpenPelt directory
```
$ ./run_tecsim.sh name_of_the_test
```
The names of all the available tests are listed [here](https://github.com/thejackal360/OpenPelt-Public#available-tests). 


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
$ python (or python3) tests/basic_bang_bang.py. 
```

## Controller Development

To develop new controller algorithms, please take a look at OpenPelt/controller.py.
Simply create a class that inherits the controller abstract class. The user is
responsible for implementing the _controller_f function.

Please look at the source code for tests in the tests/ directory. These tests
demonstrate how to use controllers on the TEC circuit model.

>>>>>>> 36f2866 (Added controller development details to README)
## Available tests

All the tests provided by OpenPelt can run by executing
```
$ pytest tests/run_all.py
```
The tests included in this repository are: 
  - **basic_bang_bang** - This is a simple bang-bang controller
  - **op_point_current** - This script characterizes the hot and cold plate temperatures
  at different drive currents.
  - **op_point_voltage** - This script characterizes the hot and cold plate temperatures
  at different drive voltages.
  - **transient** - It characterizes the transient phase of the model.
  - **pid_hot** - A PID controller that controls the temperature of the hot plate.
  - **pid_cold** - Similarly, a PID controller for controlling the temperature of the
  cold plate.


## Available examples

We provide three distinctive examples-proof of concepts-showing how the user
can combine OpenPelt with other packages such as PyTorch or Fenics. The user
can develop adaptive controllers using neural networks or reinforcement learning
as well as they can simulate heat diffusion equations with Fenics and use them
along with a TEC model provided by OpenPelt.

The demo scripts can be found in the **examples** directory. 
  - **fake_neural_network** - An untrained neural network used as proof of concept
  demonstrating how the user can use neural networks (in this case PyTorch)
  to develop controlling algorithms.
  - **random_hot** - A very naive demo of a random agent used as proof of concept
  for developing reinforcement learning algorithms using OpenPelt as TEC 
  simulated environment.
  - **fenics_heat_eqn** - This script shows how to use Fenics with OpenPelt.
 
## Platforms where OpenPelt (OpenSPICE version) has been tested

  - FreeBSD 13.0-RELEASE
    - Clang 11.0.1
    - Python 3.8.12
    - x86_64 64bit

  - Mac OS Ventura 13.0.1
    - Clang 14.0.0
    - Python 3.10.8
    - x86_64 64bit

  - Ubuntu 20.04.4 LTS
    - GCC 9.3.0 and 9.4.0
    - Python 3.8.10
    - x86_64 64bit

## Contributing Guidelines

In case you would like to contribute to OpenPelt, you should use the [Github Pull
Request](https://github.com/thejackal360/OpenPelt-Public/pulls) system. 


## For more help

See the paper, the examples of usage and the source codes in the **tests**
directory.

## Report Bugs

In case you would like to report a bug or you experience any problems with
OpenPelt, you should open an issue using the 
[Github Issue Tracker](https://github.com/thejackal360/OpenPelt-Public/issues). 
