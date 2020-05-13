# OptCat

OptCat is a codebase relating consensus optimization hosted by Dr.Han, supported by

* Shuo Han [https://hanshuo.people.uic.edu/site](https://hanshuo.people.uic.edu/site/)
* Yinbin Ma (mayinbing12@gmail.com) 01/2020 - 05/2020

## Programing principle

Why Python?

    Python has many elaborate open-source libraries for optimization.

Required libraries and reasons?

    numpy, scipy: for linear algebra and solving linear system equation.
    mpi4py: providing distributed functions.
    cvxpy: providing proximal solutions. 

## Miscellany

* Document is in `doc` folder.
* To run a specific program, `mpirun -n [num_proc] ...`.
