# GPS-Z

[Global Pathway Selection](https://www.sciencedirect.com/science/article/pii/S0010218016000638) (GPS) is an algorithm to effectively generates reduced (skeletal) chemistry mechanisms, which speeds up simulations and can be used as a systematic analytics tool to extract insights from complex reacting system.

This repo ([GPS-Z](https://github.com/llnl/GPS-Z)) is a fork of the original repository to use [Zero-RK](https://github.com/llnl/zero-rk) for accelerated solution of ignition delay problems. GPS-Z also uses python's multiprocessing library to enable parallel evaluation of GPS algorithm tasks.

![](https://github.com/llnl/GPS-Z/blob/master/ui/ui_main.PNG)

## How to install
This repo is developed with Python 3.x and depends on many packages. The recommended way to install most dependencies is with [Anaconda](https://www.anaconda.com/distribution/). 
* You can simply set up the environment by typing `conda env create -f env.yml`.

The user will also need to provide an installation of Zero-RK by following instructions in the [Zero-RK repo](https://github.com/llnl/zero-rk).  After Zero-RK is installed, GPS-Z will try to find the IDT executable by examining environment variable `ZERORK_HOME`, which should point to the Zero-RK installation folder.

## How to use
* Activate the anaconda environment: `conda activate gps`
* Start GPS: `python GPS.py`

For a more detailed tutorial, please see [Tutorial_v1.0.0.pdf](https://github.com/golsun/GPS/blob/master/Tutorial_v1.0.0.pdf).

After the GPS project has been configured via the GUI, it can be run via the GUI or can also be run in "headless" mode to enable submitting jobs via a batch-scheduling system on HPC clusters.  An example job script is given below, but will need to be adapted to the user's cluster environment (e.g. moab, slurm, lfs, etc.).

```
#!/bin/bash
#SBATCH ...

conda activate gps

export ZERORK_HOME=/path/to/zerork
export GPS_HOME=/path/to/GPS
export PYTHONPATH=${GPS_HOME}:${PYTHONPATH}

python ${GPS_HOME}/gps_headless.py project_dir/project.json >& gps.log
```

## How to cite
To improve the accuracy of reduced mechanisms, GPS considers all-generation relation between species, and minimizes the risk of broken pathways and dead-ends. This algorithm was originally developed by Prof. Wenting Sun's group at Georgia Tech [[link](http://sun.gatech.edu/)]

* X. Gao, S. Yang, W. Sun, "A global pathway selection algorithm for the reduction of detailed chemical kinetic mechanisms", **Combustion and Flame**, 167 (2016) 238–247 [[link](https://www.sciencedirect.com/science/article/pii/S0010218016000638)]
* X. Gao, X. Gou, W. Sun, "Global Pathway Analysis: a hierarchical framework to understand complex chemical kinetics", **Combustion Theory and Modelling**, 2018 pp.1-23. [[link](https://www.tandfonline.com/doi/abs/10.1080/13647830.2018.1560503)]

When using GPS-Z version of GPS, please also site the Zero-RK theory paper:

* M.J. McNenly, R.A. Whitesides, and D.L. Flowers, Faster solvers for large kinetic mechanisms using adaptive preconditioners. Proceedings of the Combustion Institute, 35(1) (2015) 581-587. [[link](https://doi.org/10.1016/j.proci.2014.05.113)]
   

## Related publication
* Gao, X., Gou, X. and Sun, W., 2018. Global Pathway Analysis: a hierarchical framework to understand complex chemical kinetics. Combustion Theory and Modelling, pp.1-23.
* X. Gao, W. Sun, Using Global Pathway Selection Method to Understand Chemical Kinetics, 55th AIAA Aerospace Sciences Meeting, (2017) AIAA 2017-0836.
* X. Gao, W. Sun, Global Pathway Analysis of the Autoignition and Extinction of Aromatic/Alkane mixture,  53rd AIAA/SAE/ASEE Joint Propulsion Conference, Atlanta, Georgia, 2017.
* S. Yang, X. Gao, W. Sun, Global Pathway Analysis of the Extinction and Re-ignition of a Turbulent Non-Premixed Flame,  53rd AIAA/SAE/ASEE Joint Propulsion Conference, Atlanta, Georgia, 2017.


License
----------------

GPS-Z is distributed under the terms of the MIT license.

All new contributions must be made under the same license.

See LICENSE and NOTICE for details.

SPDX-License-Identifier: (MIT)


LLNL-CODE-2010843
