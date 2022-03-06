![](OpenPelt.png)

OpenPelt is a faster-than-real-time temperature control simulation library.
OpenPelt contains utilities for developing and verifying temperature control
algorithms as well as a model of a thermoelectric cooler to act as the plant.
OpenPelt also enables exporting simulation results to [Fenics](https://fenicsproject.org/)
to simulate the control system's impact on a three-dimensional heat diffusion 
model.


##  Dependencies

  - Numpy
  - Pytorch


## Install

How to install it? 

### Linux


### Mac OS

After you clone the OpenPelt's repository by using your Terminal app (go to 
Finder->Utilities and then open the Terminal app or you can just use the 
Spotlight by pressing command and space and type terminal). Then type the
following command such that you clone the repository:
```
git clone https://github.com/thejackal360/OpenPelt.git
```
Then you will need to install the Ngspice and the libngspice packages using 
Homebrew package manager. If you don't already have installed Homebre you will
need to follow these steps: 


Now you can install the Ngspice and libngspice by typing:

```
# brew install ngspice
# brew install libngspice
```

Then make sure you have installed all the required packages listed in section
**Dependencies**. Finally, you have to install OpenPelt by executing the 
following commands (always in your terminal app)
```
# pip3 (or pip) install .
```
within the OpenPelt directory.


## Example usage


## Platforms where OpenPelt has been tested

  - Ubuntu 20.04.4 LTS
    - GCC 9.3.0
    - Python 3.8.10
    - x86_64 64bit


## Contributing Guidelines

Please first consult the **CONTRIBUTING.md** file for the Code of Conduct.

In case you would like to contribute to GAIM, you should use the [Gitlab Merge
Request](https://github.com/thejackal360/OpenPelt/pulls) system. 


## For more help

See the paper, the examples of usage and the source codes in the **examples**
directory.

## Report Bugs

In case you would like to report a bug or you experience any problems with
OpenPelt, you should open an issue using the 
[Githab Issue Tracker](https://github.com/thejackal360/OpenPelt/issues). 


Uses libngspice.so for version 34.
https:// sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/
