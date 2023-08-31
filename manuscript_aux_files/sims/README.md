## Simulations scripts

This directory contains examples of simulation scripts used in the experiments of the [manuscript](https://doi.org/10.1101/2023.03.01.529396).

__ReLERNN experiments__

* `rho_bkgd.slim` : Simulations of various recombination rates with background selection in SLiM.
* `rho_dem.py` : Simulations of various recombination rates under differen demographic models in msprime.

__SIA experiments__

* `swp_bkgd.slim` : Simulations of sweeps in the presence of background selection in SLiM.
* `swp_blnk_init.py` : Initial coalescent simulation of a demography with bottleneck using msprime.
* `swp_blnk_cont.slim` : Continue from a coalescent simulation to simulate sweeps under a demography with bottleneck using SLiM.
* `swp_IwM_init.py` : Initial coalescent simulation of an isolation-with-migration (IwM) demography using msprime.
* `swp_IwM_cont.slim` : Continue from a coalescent simulation to simulate sweeps under an IwM demography using SLiM.