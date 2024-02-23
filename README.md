# Capacity Estimation of Lithium-Ion batteries Through a Machine Learning Approach

This github repository contains the Python code to reproduce the results of the paper Capacity Estimation of Lithium-Ion batteries Through a Machine Learning Approach by Simone Barcellona, Lorenzo Codecasa, Silvia Colnago, Loris Cannelli, Christian
Laurano, Gabriele Maroni and Emil Petkovski.

This paper has been submitted for publication for *Electrimacs 2024, the International Conference on Modeling and Simulation of Electric Machines, Converters and Systems*.

![License Badge](https://img.shields.io/badge/license-MIT-blue)

## Abstract
Lithium-ion batteries (LiBs) are widely employed in various application fields, including renewable energy sources and electric vehicles, which heavily rely on LiBs.
This has spurred research to develop battery models capable of predicting and estimating battery behavior to optimize battery usage and decrease degradation. State of charge (SOC) and state of health are important state parameters for this purpose. In the literature, many estimation methods are based on the knowledge of the open circuit voltage-SOC relationship. The latter can be modeled in different ways, organized into three main approaches: table-based, analytical, and machine learning approaches. Among them, machine learning approaches have become popular and interesting for this goal. Therefore, the present paper develops and validates a machine learning algorithm to estimate the battery capacity of a LiB considering different levels of cycle aging.

## Software implementation
All the source code used to generate the results and figures in the paper are in the `src` and `notebooks` folders. Computations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/).

## Getting the code
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/gabribg88/LiB_Capacity_Estimation.git

or [download a zip archive](https://github.com/gabribg88/LiB_Capacity_Estimation/archive/refs/heads/master.zip).

## Requirements
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.

We recommend to use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install the dependencies without causing conflicts, with your
setup (even with different Python versions), with the pip package-management system.

Run the following command in the repository folder (where `requirements.txt`
is located) to create a separate environment and install all required
dependencies in it:

    conda create --name <env_name>
    source activate <env_name>
    pip install -r requirements.txt
