# EHT Sgr A* Theory Paper Analysis Toolkit

This repository contains tools and Jupyter notebooks for data analysis
and visualization for the EHT Sgr A* Theory Paper.
All user scripts and notebooks are in the top directory.
To use the scripts and notebooks in this package, simply run

    pip3 install -r requirements.txt

to obtain the necessary packages.


## Dependency

This repository tries to keep its dependency minimal.
We only require standard libraries:

* astropy
* click (only used in the top level driver scripts)
* h5py
* jupyterlab
* matplotlib (only used to open the notebooks)
* numpy
* pandas
* parse
* tqdm (only used in the top level driver scripts)

Because of this, instead of using custom libraries such as
[`hallmark`](https://github.com/l6a/hallmark)
and
[`mockservation`](https://github.com/focisrc/mockservation),
we put all the dependent functions in the "common/" directory.
This makes this repository easy to use and self-contained.

When we are about to finish the Sgr A* papers, it is possible that we
will push the code upstream back to the individual packages, and turn
this into a meta-package that help us manage all upstream packages.


## Design

We design this toolkit so it allows multiple people to work together.
To make it a smooth experience, we define three data stage.

* `models/`: the "input models" of this toolkit include images at
  different frequencies or full SEDs.
  These are usually outputs of high performance C or Fortran codes such as
  [`ipole`](https://github.com/AFD-Illinois/ipole) and
  [`igrmonty`](https://github.com/AFD-Illinois/ipole)
  that require significant computing resource to run.
  The number of files is of order of million and total size may reach
  10s of TB.
  The intention is that the people who generate the models are
  responsible to their own models, who may copy or link the input
  models in the `models/` directory in their own clones of this
  repository.

* `cache/`: this toolkit then preprocesses the input models to summary
  tables and/or compressed movies that are directly relevant to the
  science.
  Examples include total flux, image size, spectral filtered movies.
  The number and total size of these cache files is usually a few
  orders of mangitude smaller than the files in `models/`.
  Because of the more managable size, the intension is that people can
  share/rsync the full copy of these cache files to their own laptops
  to perform further analysis.

* `output/`: stores outputs including LaTeX tables, plots, and movies,
  that are generated from files in `cache/` and possibility `models/`.
  The intention is these output files should be identical no matter
  who generate them.
  These files should be at publication qualitity that we can place
  them directly to the Sgr A* Theory Paper.


## Other Tools

* `SYMBA` Pegasus Workflow: https://github.com/bhpire/symba-osg

* `ipole` Condor Workflow: https://github.com/bhpire/ipole-osg

* `igrmonty` Condor Workflow: https://github.com/bhpire/igrmonty-osg

* `calsz` Condor Workflow: https://github.com/bhpire/calsz-osg
