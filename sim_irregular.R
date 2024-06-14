## Simulates data in irregular (partially regular) format.  Can, as special
## cases, be used for generating regular data and completely irregular data.
sim_irregular <- function(n_a, J_a, t_a,
                          n_b, J_b, t_b,
                          magnitude_mu, length_scale_mu,
                          magnitude_eta, length_scale_eta,
                          sigma, sim_mod, stanseed = NULL)
{
    t_all <- sort(unique(c(t_a, unlist(t_b))))
    ti_a <- which(t_all %in% t_a)
    ti_b <- list()
    for (i in 1:length(t_b)) {
        ti_b[[i]] <- which(t_all %in% t_b[[i]])
    }

    dat_list <- list(
        n_obs = length(t_all),
        n_group = n_a + n_b,
        t = t_all,
        magnitude_mu = magnitude_mu,
        length_scale_mu = length_scale_mu,
        magnitude_eta = magnitude_eta,
        length_scale_eta = length_scale_eta,
        sigma = sigma
    )

    sim <- sim_mod$sample(dat_list, fixed_param = TRUE, chains = 1, iter_sampling = 1,
                          seed = stanseed)
    draws <- posterior::as_draws_rvars(sim$draws())
    mu <- posterior::draws_of(draws$mu)
    eta <- posterior::draws_of(draws$eta)
    f <- posterior::draws_of(draws$f)
    y <- posterior::draws_of(draws$y)

    y_a <- y[1, ti_a, 1:n_a]
    y_a_vec <- as.vector(y_a)
    y_b_list <- list()
    for (i in 1:length(ti_b))
        y_b_list <- c(y_b_list, list(y[1, ti_b[[i]], n_a + i]))
    y_b_vec <- unlist(y_b_list)

    return(list(
        list(y_a = y_a,
             y_a_vec = y_a_vec,
             y_b_vec = y_b_vec,
             y_ab_vec = c(y_a_vec, y_b_vec),
             t_a = t_a,
             t_b = unlist(t_b),
             n_a = n_a,
             n_b = n_b,
             J_a = J_a,
             J_b = J_b,
             t_b_is = c(0, cumsum(J_b)),
             magnitude_mu = magnitude_mu,
             length_scale_mu = length_scale_mu,
             magnitude_eta = magnitude_eta,
             length_scale_eta = length_scale_eta,
             sigma = sigma,
             one_mat_n_a = matrix(1, ncol = n_a, nrow = n_a)),
        list(f = f, y = y, n_a = n_a,
             n_b = n_b, t_all = t_all,
             ti_a = ti_a, ti_b = ti_b)
        ))
}
