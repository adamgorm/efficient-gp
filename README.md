# Efficient multi-level Gaussian process regression

This repository contains the efficient implementation of multilevel Gaussian
process regression described in:

**Hoffmann, Ekstrøm & Jensen (2025).**

*Computationally efficient multi-level Gaussian process regression for
functional data observed under completely or partially regular sampling designs.*

Published in **TEST**.

- DOI: https://doi.org/10.1007/s11749-025-00996-4
- Preprint: https://arxiv.org/abs/2406.13691

# Overview

- `functions.stan` contains functions for all parts of the implementation: the
  log-likelihood, the conditional posterior simulation, etc.
- `regular_model.stan` is a model template for using the functions to sample
  from the posterior in the completely regular sampling design.
- `irregular_model.stan` is a model template for the partially regular
  (irregular) sampling design.
- `example.R` shows how to sample from the models or expose the functions in R.
- `sim.stan` is used to simulate data from the model.
- `sim_irregular.stan` contains a helper function for doing the simulations.
- `sim_list_to_tib.R` contains a helper function for transforming the list
  output of the simulation to a tibble for convenience.

# A note on modifying the models for your case

All the functions in `functions.stan` use the exponentiated quadratic kernel,
but this is not necessary for our simplifications. If you want to modify the
code to use a different kernel, you should search for `gp_exp_quad_cov` in
`functions.stan` and change it to your kernel of choice. You may have to change
the kernel hyperparameters as well. If you do want to use the exponentiated
quadratic kernel, then you can use `regular_model.stan` or
`irregular_model.stan` directly by simply changing the priors to suit your case.
