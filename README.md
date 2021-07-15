# EHT Sgr A* Theory Paper Analysis Tools

This repository contains tools and Jupyter notebooks for data analysis
and visualization for the EHT Sgr A* Theory Paper.


## Dependency

This repository tries to keep its dependency minimal.  We only require
standard libraries:

* astropy
* click (only used in the top level scripts)
* h5py
* matplotlib
* numpy
* pandas
* parse
* tqdm (only used in the top level scripts)

Because of this, instead of using custom libraries such as
[`hallmark`](https://github.com/l6a/hallmark)
and
[`mockservation`](https://github.com/focisrc/mockservation),
we put all the dependent functions in the "common/" directory.
This makes this repository easy to use and self-contain.


## Other Tools

* `SYMBA` Pegasus Workflow: https://github.com/bhpire/symba-osg

* `ipole` Condor Workflow: https://github.com/bhpire/ipole-osg

* `igrmonty` Condor Workflow: https://github.com/bhpire/igrmonty-osg

* `calsz` Condor Workflow: https://github.com/bhpire/calsz-osg
