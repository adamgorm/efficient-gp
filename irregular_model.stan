#include functions.stan

data {
  int n_a;
  int n_b;
  int J_a;
  array[n_b] int J_b;
  int J_pred;
  array[J_pred] real t_pred;
  array[n_b + 1] int t_b_is;
  array[J_a] real t_a;
  array[sum(J_b)] real t_b;
  matrix[J_a, n_a] y_a;
  vector[n_a * J_a] y_a_vec;
  vector[sum(J_b)] y_b_vec;
  vector[n_a * J_a + sum(J_b)] y_ab_vec;
  matrix[n_a, n_a] one_mat_n_a;
}

transformed data {
  int n = n_a + n_b;
}

parameters {
  real<lower = 0> magnitude_mu;
  real<lower = 0> length_scale_mu;
  real<lower = 0> magnitude_eta;
  real<lower = 0> length_scale_eta;
  real<lower = 0> sigma;
}

model {
  /* Priors. Change to suit your case. */
  length_scale_mu ~ inv_gamma(2, 2);
  magnitude_mu ~ normal(0, 2);
  length_scale_eta ~ inv_gamma(2, 1);
  magnitude_eta ~ normal(0, 2);
  sigma ~ std_normal();
  
  target += log_lik_irreg_smart(y_a, y_a_vec, y_b_vec, y_ab_vec,
                                t_a, t_b,
                                n_a, n_b,
                                J_a, J_b, t_b_is,
                                magnitude_mu, length_scale_mu,
                                magnitude_eta, length_scale_eta,
                                sigma, one_mat_n_a);
}

generated quantities {
  vector[J_pred] mu_pred;
  matrix[J_pred, n] eta_pred;
  matrix[J_pred, n] f_pred;
  {
    matrix[J_pred, n+1] mu_eta_pred =
      mu_eta_irreg_smart_rng(y_a, y_a_vec, y_b_vec, y_ab_vec,
                             t_a, t_b, t_pred,
                             n_a, n_b, J_pred,
                             J_a, J_b, t_b_is,
                             magnitude_mu, length_scale_mu,
                             magnitude_eta, length_scale_eta,
                             sigma, one_mat_n_a);
    mu_pred = mu_eta_pred[, 1];
    eta_pred = mu_eta_pred[, 2:];
    f_pred = f_draws(mu_eta_pred);
  }
}
