# z0_local_void_BAO

In this repository, we provide the files required to reproduce the results in the paper "Constraints on the gravitational potential from DESI DR2 BAO and its implications for the local void scenario". In this paper, we explore constraints on a potential hill using BAO+CMB+CC to determine whether one of the local void predictions is consistent with the data. The primary statistical analysis is conducted using the algorithm dynesty_z0_main.py.

To run CLASS in a grid to obtain the recombination and decoupling temperatures, follow this order of operations:

1. CLASS_Tstar_decoupling.py runs the parameter grid and saves the values
2. Quadratic_fitting_grid.py fits the values using a quadratic form
3. Quadratic_fitting_rmse.py saves a file with the rms and maximum absolute deviation from the above quadratic fit.


Versions of these files marked FIRAS assume z_0 = 0 and thus T_CMB = T_FIRAS without uncertainty. Output directories will be created as part of the operation, with separate folders for recombination and decoupling. The main output files are the reference, gradient, and Hessian CSV files, along with the error file from step (3) above. The parameter range is also printed out. To set a different one, alter the algorithm in step (1). The algorithms should all be in one directory. Please set up a different directory if using the versions of these algorithms with z_0 = 0. All algorithms operate with Python.

Users will need to install the CLASS package. To make best use of the recombination and decoupling temperatures, please follow the protocol described in Najera, Banik & Desmond 2026 (Arxiv:2510.20964) and substitute the corresponding cosmic scale factors into the Hu & Sugiyama (1996) exact analytic result for the sound horizon at each epoch, as described in the above papers.
