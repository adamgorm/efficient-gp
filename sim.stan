data {
  int<lower = 1> n_obs;
  int<lower = 1> n_group;
  array[n_obs] real t;
  real<lower = 0> magnitude_mu;
  real<lower = 0> length_scale_mu;
  real<lower = 0> magnitude_eta;
  real<lower = 0> length_scale_eta;
  real<lower = 0> sigma;
}

transformed data {
  matrix[n_obs, n_obs] L_eta;
  matrix[n_group, n_group] A;
  matrix[n_obs, n_obs] L_mu;
  vector[n_obs] zero_n_obs;
  {
  matrix[n_obs, n_obs] K_eta;
  matrix[n_obs, n_obs] K_mu;
  zero_n_obs = rep_vector(0, n_obs);
  // add jitter to diagonal for numerical stability
  K_eta = add_diag(gp_exp_quad_cov(t, magnitude_eta, length_scale_eta), 1e-8);
  L_eta = cholesky_decompose(K_eta);
  real diagonal_element = sqrt((n_group - 1.0) / n_group);
  real off_diagonal_element = - 1.0 / sqrt((n_group * (n_group - 1)));
  for (i in 1:n_group)
    A[i, i] = diagonal_element;
  for (i in 1:(n_group-1)) {
    for (j in (i+1):n_group) {
      A[i, j] = off_diagonal_element;
      A[j, i] = off_diagonal_element;
    }
  }
  // add jitter to diagonal for numerical stability
  K_mu = add_diag(gp_exp_quad_cov(t, magnitude_mu, length_scale_mu), 1e-8);
  L_mu = cholesky_decompose(K_mu);
  }
}

generated quantities {
  vector[n_obs] mu;
  matrix[n_obs, n_group] eta;
  matrix[n_obs, n_group] f;
  matrix[n_obs, n_group] y;
  mu = multi_normal_cholesky_rng(zero_n_obs, L_mu);
  matrix[n_obs, n_group] Z;
  for (i_col in 1:n_group)
    for (i_row in 1:n_obs)
      Z[i_row, i_col] = std_normal_rng();
  eta = L_eta * Z * A;
  f = rep_matrix(mu, n_group) + eta;
  for (i_col in 1:n_group)
    for (i_row in 1:n_obs)
      y[i_row, i_col] = normal_rng(f[i_row, i_col], sigma);
}
