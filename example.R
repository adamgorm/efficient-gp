set.seed(1157)

## Load packages
library(tidyverse)
options(mc.cores = parallel::detectCores())
source("sim_irregular.R")
source("sim_list_to_tib.R")

## Compile Stan code for simulation of GP
sim_mod <- cmdstanr::cmdstan_model("sim.stan")
## Compile and expose Stan functions to R
stan_funs <- cmdstanr::cmdstan_model("functions.stan",
                                     force_recompile = TRUE,
                                     compile_standalone = TRUE)$functions
## Compile complete Stan models that use our implementation
irreg_mod <- cmdstanr::cmdstan_model("irregular_model.stan")
reg_mod <- cmdstanr::cmdstan_model("regular_model.stan")



##### Illustration for partially regular (irregular) sampling design

## Simulate data with 3 regularly sampled and 2 irregularly sampled functions
n_a <- 3
n_b <- 2
n <- n_a + n_b
J_a <- 75
J_b <- c(50, 25)
J <- J_a + sum(J_b)
t_a <- seq(-10, 10, length.out = J_a)
t_b <- list()
for (i in 1:n_b) {
    t_b <- c(t_b, list(seq(-10, 10, length.out = J_b[i])))
}
magnitude_mu <- 1
length_scale_mu <- 1
magnitude_eta <- 0.5
length_scale_eta <- 0.5
sigma <- 0.2
sim_res <- sim_irregular(n_a, J_a, t_a,
                         n_b, J_b, t_b,
                         magnitude_mu, length_scale_mu,
                         magnitude_eta, length_scale_eta,
                         sigma, sim_mod, stanseed = 1533)
## Convert list output to tibble and manually calculate mu as average of fs
sim_tib_full <- sim_list_to_tib(sim_res[[2]]) |>
    mutate(mu_true = sim_res[[2]]$f[1, , 1:n] |> apply(1, mean) |> rep(n))

## Draw samples from posterior at equidistant time points between -10 and 10.
J_pred <- 100
t_pred <- seq(-10, 10, length.out = J_pred)
post_samples <- irreg_mod$sample(c(sim_res[[1]],
                                   list(J_pred = J_pred,
                                        t_pred = t_pred)))

## Extract draws of mu and f
draws <- posterior::as_draws_rvars(post_samples$draws())
f_pred <- posterior::draws_of(draws$f_pred)
mu <- posterior::draws_of(draws$mu)

## Calculate posterior mean and 90%-interval for mu and f
mu_ci <- apply(mu, 2, \(x) quantile(x, c(0.05, 0.95)))
mu_mean <- apply(mu, 2, mean)
f_mean <- apply(f_pred, c(2, 3), mean)
f_ci <- apply(f_pred, c(2, 3), \(x) quantile(x, c(0.05, 0.95)))

## Plot posterior means and intervals for mu and f_1
ggplot() +
    geom_line(aes(x = t_pred, y = mu_mean, col = "mu", linetype = "posterior mean")) +
    geom_ribbon(aes(x = t_pred, ymin = mu_ci[1, ], ymax = mu_ci[2, ], fill = "mu"),
                alpha = 0.1) +
    geom_line(aes(x = t_pred, y = f_mean[, 1], col = "f1", linetype = "posterior mean")) +
    geom_ribbon(aes(x = t_pred, ymin = f_ci[1, , 1], ymax = f_ci[2, , 1], fill = "f1"),
                alpha = 0.1) +
    geom_line(aes(x = t, y = f, col = "f1", linetype = "truth"),
              filter(sim_tib_full, group_id == "a1")) +
    geom_line(aes(x = t, y = mu_true, col = "mu", linetype = "truth"),
              filter(sim_tib_full, group_id == "a1")) +
    theme(axis.title = element_blank())



##### Illustration for completely regular sampling design

## Simulate regularly sampled data (by reusing our function for irregular
## sampling, but setting the sampling grid to be the same for the irregular
## part).

n_a <- 4
n_b <- 1
n <- n_a + n_b
J_a <- 75
J_b <- 75
J <- J_a + sum(J_b)
t_a <- seq(-10, 10, length.out = J_a)
t_b <- list()
for (i in 1:n_b) {
    t_b <- c(t_b, list(seq(-10, 10, length.out = J_b[i])))
}
magnitude_mu <- 1
length_scale_mu <- 1
magnitude_eta <- 0.5
length_scale_eta <- 0.5
sigma <- 0.2
sim_res <- sim_irregular(n_a, J_a, t_a,
                         n_b, J_b, t_b,
                         magnitude_mu, length_scale_mu,
                         magnitude_eta, length_scale_eta,
                         sigma, sim_mod, stanseed = 1533)
## Convert list output to tibble and manually calculate mu as average of fs
sim_tib_full <- sim_list_to_tib(sim_res[[2]]) |>
    mutate(mu_true = sim_res[[2]]$f[1, , 1:n] |> apply(1, mean) |> rep(n))

t_obs <- filter(sim_tib_full, group_id == "a1") |> pull(t)
J_obs <- length(t_obs)
y_obs_vec <- sim_tib_full$y
dat_reg <- with(sim_res[[1]], {
    list(
        y_obs_vec = y_obs_vec,
        t_obs = t_obs,
        n_obs = J_obs,
        n_group = n,
        n_obs_total = J_obs * n,
        magnitude_mu = magnitude_mu,
        length_scale_mu = length_scale_mu,
        magnitude_eta = magnitude_eta,
        length_scale_eta = length_scale_eta,
        sigma = sigma,
        zero_n_obs_total = rep(0, J_obs * n),
        one_mat_n_a = matrix(1, ncol = n_a, nrow = n_a),
        y_obs = matrix(y_obs_vec, ncol = n),
        t_pred = t_pred,
        n_pred = J_pred
    )})

## Draw samples from posterior at equidistant time points between -10 and 10.
post_samples <- reg_mod$sample(dat_reg)

## Extract draws of mu and f
draws <- posterior::as_draws_rvars(post_samples$draws())
f_pred <- posterior::draws_of(draws$f_pred)
mu <- posterior::draws_of(draws$mu)

## Calculate posterior mean and 90%-interval for mu and f
mu_ci <- apply(mu, 2, \(x) quantile(x, c(0.05, 0.95)))
mu_mean <- apply(mu, 2, mean)
f_mean <- apply(f_pred, c(2, 3), mean)
f_ci <- apply(f_pred, c(2, 3), \(x) quantile(x, c(0.05, 0.95)))

## Plot posterior means and intervals for mu and f_1
ggplot() +
    geom_line(aes(x = t_pred, y = mu_mean, col = "mu", linetype = "posterior mean")) +
    geom_ribbon(aes(x = t_pred, ymin = mu_ci[1, ], ymax = mu_ci[2, ], fill = "mu"),
                alpha = 0.1) +
    geom_line(aes(x = t_pred, y = f_mean[, 1], col = "f1", linetype = "posterior mean")) +
    geom_ribbon(aes(x = t_pred, ymin = f_ci[1, , 1], ymax = f_ci[2, , 1], fill = "f1"),
                alpha = 0.1) +
    geom_line(aes(x = t, y = f, col = "f1", linetype = "truth"),
              filter(sim_tib_full, group_id == "a1")) +
    geom_line(aes(x = t, y = mu_true, col = "mu", linetype = "truth"),
              filter(sim_tib_full, group_id == "a1")) +
    theme(axis.title = element_blank())
