functions {

  /* log-lik implementations */

  real
  log_lik_reg_smart(vector y_obs_vec, matrix y_obs,
                     array[] real t_obs,
                     int n_obs, int n_group, int n_obs_total,
                     real magnitude_mu, real length_scale_mu,
                     real magnitude_eta, real length_scale_eta,
                     real sigma,
                     matrix one_mat_n_group)
  {
    matrix[n_obs, n_obs] K_mu =
      gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
    matrix[n_obs, n_obs] K_eta =
      gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
    matrix[n_obs, n_obs] Sigma0 = add_diag(n_group / (n_group - 1.0) * K_eta,
                                           max([square(sigma), 1e-8]));
    matrix[n_obs, n_obs] L0 = cholesky_decompose(Sigma0);
    matrix[n_obs, n_obs] Sigma1 = add_diag(n_group * K_mu,
                                           max([square(sigma), 1e-8]));
    matrix[n_obs, n_obs] L1 = cholesky_decompose(Sigma1);
    real log_det0 = cholesky_log_det(L0);
    real log_det1 = cholesky_log_det(L1);
    real log_det = (n_group - 1.0) * log_det0 + log_det1;
    matrix[n_obs, n_group] prod0 = cholesky_left_divide_mat(L0, y_obs);
    matrix[n_obs, n_group] prod1 = cholesky_left_divide_mat(L1, y_obs);
    real quad_form = y_obs_vec' *
      to_vector(prod0 + 1.0 / n_group * (prod1 - prod0) * one_mat_n_group);

    return multi_normal_lpdf_prepared(n_obs_total, log_det, quad_form);
  }

  real
  log_lik_reg_base(vector y_obs_vec, array[] real t_obs,
                   int n_obs, int n_group, int n_obs_total,
                   real magnitude_mu, real length_scale_mu,
                   real magnitude_eta, real length_scale_eta,
                   real sigma,
                   vector zero_n_obs_total)
  {
    matrix[n_obs, n_obs] K_mu =
      gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
    matrix[n_obs, n_obs] K_eta =
      gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
    matrix[n_obs, n_obs] diagonal_block = add_diag(K_mu + K_eta, max([square(sigma), 1e-8]));
    matrix[n_obs, n_obs] off_diagonal_block = K_mu - K_eta / (n_group - 1.0);
    matrix[n_obs_total, n_obs_total] cov_mat =
      block_mat_AB(diagonal_block, off_diagonal_block, n_group);
    matrix[n_obs_total, n_obs_total] L = cholesky_decompose(cov_mat);
    return multi_normal_lpdf_prepared(n_obs_total, cholesky_log_det(L),
                                      y_obs_vec' * cholesky_left_divide_vec(L, y_obs_vec));
  }

  real
  log_lik_irreg_base(matrix y_a, vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                     array[] real t_a, array[] real t_b,
                     int n_a, int n_b,
                     int J_a, array[] int J_b, array[] int t_b_is,
                     real magnitude_mu, real length_scale_mu,
                     real magnitude_eta, real length_scale_eta,
                     real sigma)
  {
    int N = n_a * J_a + sum(J_b);
    matrix[N, N] L = Sigma_chol_irreg_base(t_a, t_b,
                                           n_a, n_b,
                                           J_a, J_b, t_b_is,
                                           magnitude_mu, length_scale_mu,
                                           magnitude_eta, length_scale_eta,
                                           sigma);
    real log_det = cholesky_log_det(L);
    real quad_form = y_ab_vec' * cholesky_left_divide_vec(L, y_ab_vec);
    
    return multi_normal_lpdf_prepared(N, log_det, quad_form);
  }

  real
  log_lik_irreg_smart(matrix y_a, vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                      array[] real t_a, array[] real t_b,
                      int n_a, int n_b,
                      int J_a, array[] int J_b, array[] int t_b_is,
                      real magnitude_mu, real length_scale_mu,
                      real magnitude_eta, real length_scale_eta,
                      real sigma,
                      matrix one_mat_n_a)
  {
    int col_start, col_end, row_start, row_end;
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;
    
    tuple(vector[N_a], vector[N_b],
          matrix[J_a, J_a], matrix[J_a, J_a],
          matrix[N_b, N_b]) Saiy_Sbiy_L0_L1_LSchur =
      irreg_smart_Sainvy_Sbinvy_L0_L1_LSchur(y_a, y_a_vec,
                                             y_b_vec, y_ab_vec,
                                             t_a, t_b,
                                             n_a, n_b,
                                             J_a, J_b, t_b_is,
                                             magnitude_mu, length_scale_mu,
                                             magnitude_eta, length_scale_eta,
                                             sigma,
                                             one_mat_n_a);

    real log_det_A = (n_a - 1.0) *
      cholesky_log_det(Saiy_Sbiy_L0_L1_LSchur.3) +
      cholesky_log_det(Saiy_Sbiy_L0_L1_LSchur.4);
    real log_det = log_det_A +
      cholesky_log_det(Saiy_Sbiy_L0_L1_LSchur.5);

    real quad_form = y_a_vec' * Saiy_Sbiy_L0_L1_LSchur.1 +
      y_b_vec' * Saiy_Sbiy_L0_L1_LSchur.2;

    return multi_normal_lpdf_prepared(N, log_det, quad_form);
  }

  real
  log_lik_completely_irreg(vector y_vec,
                           array[] real t,
                           int n,
                           array[] int J,
                           array[] int t_is,
                           real magnitude_mu, real length_scale_mu,
                           real magnitude_eta, real length_scale_eta,
                           real sigma)
  {
    int N = sum(J);

    matrix[N, N] B = irregular_cov_mat_B(N, n, n,
                                         J, t, t_is,
                                         magnitude_mu, length_scale_mu,
                                         magnitude_eta, length_scale_eta,
                                         sigma);
    matrix[N, N] L = cholesky_decompose(B);

    real log_det = cholesky_log_det(L);
    real quad_form = y_vec' * cholesky_left_divide_vec(L, y_vec);
    
    return multi_normal_lpdf_prepared(N, log_det, quad_form);
  }

  /* Posterior draw implementations */

  /* Functions for drawing */

  /* Get f from mu and eta */
  /* Note: If you also want to be able to get draws of y, then you just have to
     use normal_rng with mean f_pred and sd sigma */
  matrix
  f_draws(matrix mu_eta_pred)
  {
    int n_pred = rows(mu_eta_pred);
    int n_group = cols(mu_eta_pred) - 1;
    matrix[n_pred, n_group] f_pred;
    for (i in 1:n_group) {
      f_pred[, i] = mu_eta_pred[, 1] + mu_eta_pred[, i + 1];
    }
    return f_pred;
  }

  /* General function for drawing from posterior */
  matrix
  mu_eta_post_rng(int n_pred, int n_group,
                  tuple(vector, matrix, vector, matrix) mu_eta_mean_cov_post)
  {
    /* first row is mu_pred, the other n_group rows are eta_pred */
    matrix[n_pred, n_group + 1] mu_eta_pred;
    mu_eta_pred[, 1] = multi_normal_rng(mu_eta_mean_cov_post.1,
                                        add_diag(mu_eta_mean_cov_post.2, 1e-8));
    mu_eta_pred[, 2:n_group] =
      to_matrix(multi_normal_rng(mu_eta_mean_cov_post.3,
                                 add_diag(mu_eta_mean_cov_post.4, 1e-8)),
                n_pred, n_group - 1);
    mu_eta_pred[, n_group+1] = get_last_eta(mu_eta_pred);

    return mu_eta_pred;
  }

  /* In irregular case: call with n_group = n_a + n_b. */
  matrix
  mu_eta_post_irreg_rng(int n_pred, int n_group,
                  tuple(vector, matrix) mu_eta_mean_cov_post)
  {    
    /* first row is mu_pred, the other n_group rows are eta_pred */
    matrix[n_pred, n_group + 1] mu_eta_pred;
    mu_eta_pred[, :n_group] =
      to_matrix(multi_normal_rng(mu_eta_mean_cov_post.1,
                                 add_diag(mu_eta_mean_cov_post.2, 1e-8)),
                n_pred, n_group);
    mu_eta_pred[, n_group+1] = get_last_eta(mu_eta_pred);

    return mu_eta_pred;
  }


  /* Version that takes precalculated chol(cov(eta(t_pred))) instead of
     cov(eta(t_pred)) */
  matrix
  mu_eta_post_chol_rng(int n_pred, int n_group,
                       tuple(vector, matrix, vector, matrix) mu_eta_mean_cov_post_chol)
  {
    /* first row is mu_pred, the other n_group rows are eta_pred */
    matrix[n_pred, n_group + 1] mu_eta_pred;    
    mu_eta_pred[, 1] = multi_normal_rng(mu_eta_mean_cov_post_chol.1,
                                        add_diag(mu_eta_mean_cov_post_chol.2, 1e-8));
    mu_eta_pred[, 2:n_group] =
      to_matrix(multi_normal_cholesky_rng(mu_eta_mean_cov_post_chol.3,
                                          mu_eta_mean_cov_post_chol.4),
                n_pred, n_group - 1);
    mu_eta_pred[, n_group + 1] = mu_eta_pred[:, 2:n_group] * rep_vector(-1, n_group - 1);
    
    return mu_eta_pred;
  }

  /* Irregular joint draw, where the covariance cholesky must be in the order
     (eta, mu), unlike above where it is (mu, eta) - this is to make it easier
     to do block Cholesky. */
  matrix
  mu_eta_post_chol_irreg_rng(int n_pred, int n_group,
                             tuple(vector, matrix) eta_mu_mean_cov_post_chol)
  {
    /* first row is mu_pred, the other n_group rows are eta_pred */
    matrix[n_pred, n_group + 1] mu_eta_pred;    
    mu_eta_pred[, 2:(n_group+1)] =
      to_matrix(multi_normal_cholesky_rng(eta_mu_mean_cov_post_chol.1,
                                          eta_mu_mean_cov_post_chol.2),
                n_pred, n_group);
    /* shift mu to first column */
    mu_eta_pred[, 1] = mu_eta_pred[, n_group+1];
    /* calculate last eta for last column (to sum to 0) */
    mu_eta_pred[, n_group + 1] = get_last_eta(mu_eta_pred);
    
    return mu_eta_pred;
  }


  matrix
  mu_eta_reg_base_rng(vector y_obs_vec, array[] real t_obs, array[] real t_pred,
                      int n_obs, int n_group, int n_obs_total, int n_pred,
                      real magnitude_mu, real length_scale_mu,
                      real magnitude_eta, real length_scale_eta,
                      real sigma)
  {
    int n_pred_except_1_group = n_pred * (n_group - 1);
    tuple(vector[n_pred], matrix[n_pred, n_pred],
          vector[n_pred_except_1_group],
          matrix[n_pred_except_1_group, n_pred_except_1_group]) mu_eta_mean_cov_post =
      mu_eta_reg_base_post(y_obs_vec, t_obs, t_pred,
                           n_obs, n_group, n_obs_total, n_pred,
                           magnitude_mu, length_scale_mu,
                           magnitude_eta, length_scale_eta,
                           sigma);
    return mu_eta_post_rng(n_pred, n_group, mu_eta_mean_cov_post);
  }

  matrix
  mu_eta_reg_smart_rng(vector y_obs_vec, matrix y_obs,
                       array[] real t_obs, array[] real t_pred,
                       int n_obs, int n_group, int n_obs_total, int n_pred,
                       real magnitude_mu, real length_scale_mu,
                       real magnitude_eta, real length_scale_eta,
                       real sigma, matrix one_mat_n_group)
  {
    int n_pred_except_1_group = n_pred * (n_group - 1);
    tuple(vector[n_pred], matrix[n_pred, n_pred],
          vector[n_pred_except_1_group],
          matrix[n_pred_except_1_group, n_pred_except_1_group]) mu_eta_mean_cov_post =
      mu_eta_reg_smart_post(y_obs_vec, y_obs, t_obs, t_pred,
                            n_obs, n_group, n_obs_total, n_pred,
                            magnitude_mu, length_scale_mu,
                            magnitude_eta, length_scale_eta,
                            sigma, one_mat_n_group);
    return mu_eta_post_rng(n_pred, n_group, mu_eta_mean_cov_post);
  }

  matrix
  mu_eta_reg_iter_rng(vector y_obs_vec, matrix y_obs,
                      array[] real t_obs, array[] real t_pred,
                      int n_obs, int n_group, int n_obs_total, int n_pred,
                      real magnitude_mu, real length_scale_mu,
                      real magnitude_eta, real length_scale_eta,
                      real sigma, matrix one_mat_n_group)
  {
    int n_pred_except_1_group = n_pred * (n_group - 1);
    tuple(vector[n_pred], matrix[n_pred, n_pred],
          vector[n_pred_except_1_group],
          matrix[n_pred_except_1_group,
                 n_pred_except_1_group]) mu_eta_mean_cov_post_chol =
      mu_eta_reg_iter_post_chol(y_obs_vec, y_obs, t_obs, t_pred,
                                n_obs, n_group, n_obs_total, n_pred,
                                magnitude_mu, length_scale_mu,
                                magnitude_eta, length_scale_eta,
                                sigma, one_mat_n_group);
    return mu_eta_post_chol_rng(n_pred, n_group, mu_eta_mean_cov_post_chol);
  }

  matrix
  mu_eta_reg_smart_iter_rng(vector y_obs_vec, matrix y_obs,
                            array[] real t_obs, array[] real t_pred,
                            int n_obs, int n_group, int n_obs_total, int n_pred,
                            real magnitude_mu, real length_scale_mu,
                            real magnitude_eta, real length_scale_eta,
                            real sigma, matrix one_mat_n_group)
  {
    int n_pred_except_1_group = n_pred * (n_group - 1);
    tuple(vector[n_pred], matrix[n_pred, n_pred],
          vector[n_pred_except_1_group],
          matrix[n_pred_except_1_group,
                 n_pred_except_1_group]) mu_eta_mean_cov_post_chol =
      mu_eta_reg_smart_iter_post_chol(y_obs_vec, y_obs, t_obs, t_pred,
                                      n_obs, n_group, n_obs_total, n_pred,
                                      magnitude_mu, length_scale_mu,
                                      magnitude_eta, length_scale_eta,
                                      sigma, one_mat_n_group);
    return mu_eta_post_chol_rng(n_pred, n_group, mu_eta_mean_cov_post_chol);
  }

  matrix
  mu_eta_irreg_base_rng(vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                        array[] real t_a, array[] real t_b, array[] real t_pred,
                        int n_a, int n_b, int J_pred,
                        int J_a, array[] int J_b, array[] int t_b_is,
                        real magnitude_mu, real length_scale_mu,
                        real magnitude_eta, real length_scale_eta,
                        real sigma)
  {
    int n = n_a + n_b;
    int N_pred_except_1_group = J_pred * (n - 1);
    int mu_eta_dim = J_pred + N_pred_except_1_group;
    tuple(vector[mu_eta_dim],
          matrix[mu_eta_dim, mu_eta_dim]) mu_eta_mean_cov_post =
      mu_eta_irreg_base_post(y_a_vec, y_b_vec, y_ab_vec,
                             t_a, t_b, t_pred,
                             n_a, n_b, J_pred,
                             J_a, J_b, t_b_is,
                             magnitude_mu, length_scale_mu,
                             magnitude_eta, length_scale_eta,
                             sigma);
    return mu_eta_post_irreg_rng(J_pred, n, mu_eta_mean_cov_post);
  }

  matrix
  mu_eta_completely_irreg_rng(vector y_vec,
                              array[] real t, array[] real t_pred,
                              int n, int J_pred,
                              array[] int J, array[] int t_is,
                              real magnitude_mu, real length_scale_mu,
                              real magnitude_eta, real length_scale_eta,
                              real sigma)
  {
    int N_pred_except_1_group = J_pred * (n - 1);
    tuple(vector[J_pred], matrix[J_pred, J_pred],
          vector[N_pred_except_1_group],
          matrix[N_pred_except_1_group,
                 N_pred_except_1_group]) mu_eta_mean_cov_post_chol =
      mu_eta_completely_irreg_post(y_vec,
                                   t, t_pred,
                                   n, J_pred,
                                   J, t_is,
                                   magnitude_mu, length_scale_mu,
                                   magnitude_eta, length_scale_eta,
                                   sigma);
    return mu_eta_post_rng(J_pred, n, mu_eta_mean_cov_post_chol);
  }

  matrix
  mu_eta_irreg_smart_rng(matrix y_a, vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                         array[] real t_a, array[] real t_b, array[] real t_pred,
                         int n_a, int n_b, int J_pred,
                         int J_a, array[] int J_b, array[] int t_b_is,
                         real magnitude_mu, real length_scale_mu,
                         real magnitude_eta, real length_scale_eta,
                         real sigma, matrix one_mat_n_a)
  {
    int n = n_a + n_b;
    int N_pred_except_1_group = J_pred * (n - 1);
    int mu_eta_dim = J_pred + N_pred_except_1_group;
    tuple(vector[mu_eta_dim],
          matrix[mu_eta_dim, mu_eta_dim]) eta_mu_mean_cov_post_chol =
      eta_mu_irreg_smart_post(y_a, y_a_vec, y_b_vec, y_ab_vec,
                              t_a, t_b, t_pred,
                              n_a, n_b,
                              J_pred, J_a, J_b,
                              t_b_is,
                              magnitude_mu, length_scale_mu,
                              magnitude_eta, length_scale_eta,
                              sigma, one_mat_n_a);
    return mu_eta_post_chol_irreg_rng(J_pred, n, eta_mu_mean_cov_post_chol);
  }

   /* Functions for calculating the posterior distribution */

   tuple(vector, matrix, vector, matrix)
   mu_eta_reg_base_post(vector y_obs_vec, array[] real t_obs, array[] real t_pred,
                        int n_obs, int n_group, int n_obs_total, int n_pred,
                        real magnitude_mu, real length_scale_mu,
                        real magnitude_eta, real length_scale_eta,
                        real sigma)
   {
     matrix[n_obs, n_obs] K_mu_obs =
       gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_obs] K_mu_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_pred] K_mu_pred =
       gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);
     matrix[n_obs, n_obs] K_eta_obs =
       gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_obs] K_eta_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_pred] K_eta_pred =
       gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);
     matrix[n_obs, n_obs] y_var = add_diag(K_mu_obs + K_eta_obs, max([square(sigma), 1e-8]));
     // construct var matrix for (y_1, ..., y_n)
     matrix[n_obs, n_obs] off_diagonal_y_var = K_mu_obs - K_eta_obs / (n_group - 1);
     matrix[n_obs_total, n_obs_total] all_y_var =
       block_mat_AB(y_var, off_diagonal_y_var, n_group);
     // off-diagonal cov matrix cov((y_1, ..., y_n), mu)
     matrix[n_pred, n_obs_total] cov_mu_all_y = block_rep(K_mu_pred_obs, 1, n_group);
     matrix[n_obs_total, n_obs_total] chol_all_y_var = cholesky_decompose(all_y_var);
     vector[n_obs_total] y_var_inv_times_y = cholesky_left_divide_vec(chol_all_y_var, y_obs_vec);
     vector[n_pred] mu_mean_post = cov_mu_all_y * y_var_inv_times_y;
     matrix[n_pred, n_pred] mu_cov_post =
       K_mu_pred - crossprod(mdivide_left_tri_low(chol_all_y_var, cov_mu_all_y'));

      /* eta */
     int n_group_minus_1 = n_group - 1;
     int n_pred_except_1_group = n_pred * (n_group - 1);
     matrix[n_pred, n_pred] off_diagonal_block_eta = - K_eta_pred / (n_group - 1);
     matrix[n_pred_except_1_group, n_pred_except_1_group] cov_eta_pred =
       block_mat_AB(K_eta_pred, off_diagonal_block_eta, n_group_minus_1);
     matrix[n_pred, n_obs] off_diagonal_block_eta_y = - K_eta_pred_obs / (n_group - 1);
     int row_start, row_end, col_start, col_end;
     matrix[n_pred_except_1_group, n_obs_total] cov_eta_all_y;
     for (i in 1:n_group_minus_1) {
       row_start = (i-1) * n_pred + 1;
       row_end = row_start + n_pred - 1;
       for (j in 1:n_group) {
         col_start = (j-1) * n_obs + 1;
         col_end = col_start + n_obs - 1;
         if (i == j)
           cov_eta_all_y[row_start:row_end, col_start:col_end] = K_eta_pred_obs;
         else
           cov_eta_all_y[row_start:row_end, col_start:col_end] = off_diagonal_block_eta_y;
       }
     }

     vector[n_pred_except_1_group] eta_mean_post = cov_eta_all_y * y_var_inv_times_y;
     matrix[n_pred_except_1_group, n_pred_except_1_group] eta_cov_post =
       cov_eta_pred - crossprod(mdivide_left_tri_low(chol_all_y_var, cov_eta_all_y'));

     return (mu_mean_post, mu_cov_post, eta_mean_post, eta_cov_post);
   }

   tuple(vector, matrix, vector, matrix)
   mu_eta_reg_smart_post(vector y_obs_vec, matrix y_obs, array[] real t_obs, array[] real t_pred,
                         int n_obs, int n_group, int n_obs_total, int n_pred,
                         real magnitude_mu, real length_scale_mu,
                         real magnitude_eta, real length_scale_eta,
                         real sigma, matrix one_mat_n_group)
   {
     int n_pred_except_1_group = n_pred * (n_group - 1);
     int n_pred_total = n_pred * n_group;
     matrix[n_obs, n_obs] K_mu_obs =
       gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_obs] K_mu_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_pred] K_mu_pred =
       gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);

     matrix[n_obs, n_obs] K_eta_obs =
       gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_obs] K_eta_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_pred] K_eta_pred =
       gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);

     matrix[n_obs, n_obs] Sigma0 = add_diag(n_group / (n_group - 1.0) * K_eta_obs,
                                            max([square(sigma), 1e-8]));
     matrix[n_obs, n_obs] Sigma1 = add_diag(n_group * K_mu_obs,
                                            max([square(sigma), 1e-8]));
     matrix[n_obs, n_obs] L0 = cholesky_decompose(Sigma0);
     matrix[n_obs, n_obs] L1 = cholesky_decompose(Sigma1);

     vector[n_pred] mu_mean_post =
       K_mu_pred_obs * cholesky_left_divide_vec(L1, y_obs * one_mat_n_group[, 1]);
     matrix[n_pred, n_pred] mu_cov_post =
       K_mu_pred - n_group * K_mu_pred_obs * cholesky_left_divide_mat(L1, K_mu_pred_obs');

     matrix[n_obs, n_group] prod0 = cholesky_left_divide_mat(L0, y_obs);
     vector[n_pred_except_1_group] eta_mean_post = to_vector((n_group * K_eta_pred_obs * prod0 -
                                                        K_eta_pred_obs * prod0 *
                                                        one_mat_n_group) /
                                                       (n_group - 1.0))[1:n_pred_except_1_group];


     matrix[n_pred, n_pred] quad_form_eta =
       K_eta_pred_obs * cholesky_left_divide_mat(L0, K_eta_pred_obs');
     matrix[n_pred, n_pred] off_diagonal_block_eta =
       1/(n_group-1.0) * (n_group/(n_group-1.0) * quad_form_eta - K_eta_pred);
     matrix[n_pred, n_pred] diagonal_block_eta =
       n_group/(n_group-1.0) * (K_eta_pred -
                                n_group/(n_group-1.0) * quad_form_eta) +
       off_diagonal_block_eta;
     int n_group_minus_1 = n_group - 1;
     matrix[n_pred_except_1_group, n_pred_except_1_group] eta_cov_post =
       block_mat_AB(diagonal_block_eta, off_diagonal_block_eta, n_group_minus_1);

     return (mu_mean_post, mu_cov_post, eta_mean_post, eta_cov_post);
   }

   /* Old version, using iterative block Cholesky,
      without the smart division simplification */
   tuple(vector, matrix, vector, matrix)
   mu_eta_reg_iter_post_chol(vector y_obs_vec, matrix y_obs,
                             array[] real t_obs, array[] real t_pred,
                             int n_obs, int n_group, int n_obs_total, int n_pred,
                             real magnitude_mu, real length_scale_mu,
                             real magnitude_eta, real length_scale_eta,
                             real sigma, matrix one_mat_n_group)
   {
     int n_pred_except_1_group = n_pred * (n_group - 1);
     int n_pred_total = n_pred * n_group;
     matrix[n_obs, n_obs] K_mu_obs =
       gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_obs] K_mu_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_mu, length_scale_mu);
     matrix[n_pred, n_pred] K_mu_pred =
       gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);

     matrix[n_obs, n_obs] K_eta_obs =
       gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_obs] K_eta_pred_obs =
       gp_exp_quad_cov(t_pred, t_obs, magnitude_eta, length_scale_eta);
     matrix[n_pred, n_pred] K_eta_pred =
       gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);

     matrix[n_obs, n_obs] Sigma0 = add_diag(n_group / (n_group - 1.0) * K_eta_obs,
                                            max([square(sigma), 1e-8]));
     matrix[n_obs, n_obs] Sigma1 = add_diag(n_group * K_mu_obs,
                                            max([square(sigma), 1e-8]));
     matrix[n_obs, n_obs] L0 = cholesky_decompose(Sigma0);
     matrix[n_obs, n_obs] L1 = cholesky_decompose(Sigma1);

     vector[n_pred] mu_mean_post =
       K_mu_pred_obs * cholesky_left_divide_vec(L1, y_obs * one_mat_n_group[, 1]);
     matrix[n_pred, n_pred] mu_cov_post =
       K_mu_pred - n_group * K_mu_pred_obs * cholesky_left_divide_mat(L1, K_mu_pred_obs');

     matrix[n_obs, n_group] prod0 = cholesky_left_divide_mat(L0, y_obs);
     vector[n_pred_except_1_group] eta_mean_post = to_vector((n_group * K_eta_pred_obs * prod0 -
                                                        K_eta_pred_obs * prod0 *
                                                        one_mat_n_group) /
                                                       (n_group - 1.0))[1:n_pred_except_1_group];

     matrix[n_pred, n_pred] quad_form_eta =
       K_eta_pred_obs * cholesky_left_divide_mat(L0, K_eta_pred_obs');
     matrix[n_pred, n_pred] off_diagonal_block_eta =
       1/(n_group-1.0) * (n_group/(n_group-1.0) * quad_form_eta - K_eta_pred);
     matrix[n_pred, n_pred] diagonal_block_eta =
       n_group/(n_group-1.0) * (K_eta_pred -
                                n_group/(n_group-1.0) * quad_form_eta) +
                                     off_diagonal_block_eta;
     int row_start, row_end, col_start, col_end;
     int n_group_minus_1 = n_group - 1;
     matrix[n_pred_except_1_group, n_pred_except_1_group] eta_cov_post =
       block_mat_AB(diagonal_block_eta, off_diagonal_block_eta, n_group_minus_1);

    /* Do Cholesky decomposition of eta_cov_post by iterative 2x2 block
       Cholesky. */
    matrix[n_pred, n_pred] S_i;
    matrix[n_pred_except_1_group, n_pred_except_1_group] L_cov_eta;
    int row_col_end_old = 0;
    int row_col_start = 1;
    int row_col_end = n_pred;
    matrix[n_pred, n_pred] BAB = off_diagonal_block_eta *
      diagonal_block_eta * off_diagonal_block_eta;
    matrix[n_pred, n_pred] Bcubed = matrix_power(off_diagonal_block_eta, 3);
    L_cov_eta[1:n_pred, 1:n_pred] = cholesky_decompose(add_diag(diagonal_block_eta, 1e-8));
    for (i in 2:n_group_minus_1) {
      row_col_end_old = row_col_end;
      row_col_start = row_col_end + 1;
      row_col_end = row_col_start + n_pred - 1;

      /* Upper right = 0 */
      L_cov_eta[:row_col_end_old, row_col_start:row_col_end] =
        rep_matrix(0, row_col_end_old, n_pred);

      /* Lower left */
      L_cov_eta[row_col_start:row_col_end, :row_col_end_old] =
        mdivide_left_tri_low(L_cov_eta[:row_col_end_old, :row_col_end_old],
                             eta_cov_post[(n_pred+1):(n_pred + 1 + (i-1)*n_pred - 1), :n_pred])';
      
      /* Lower right */
      L_cov_eta[row_col_start:row_col_end, row_col_start:row_col_end] =
        cholesky_decompose(add_diag(diagonal_block_eta -
                                    tcrossprod(L_cov_eta[row_col_start:row_col_end,
                                                         :row_col_end_old]),
                                    1e-8));
    }

    return (mu_mean_post, mu_cov_post, eta_mean_post, L_cov_eta);
  }

  tuple(vector, matrix, vector, matrix)
  mu_eta_reg_smart_iter_post_chol(vector y_obs_vec, matrix y_obs,
                                  array[] real t_obs, array[] real t_pred,
                                  int n_obs, int n_group, int n_obs_total, int n_pred,
                                  real magnitude_mu, real length_scale_mu,
                                  real magnitude_eta, real length_scale_eta,
                                  real sigma, matrix one_mat_n_group)
  {
    int n_pred_except_1_group = n_pred * (n_group - 1);
    int n_pred_total = n_pred * n_group;
    matrix[n_obs, n_obs] K_mu_obs =
      gp_exp_quad_cov(t_obs, magnitude_mu, length_scale_mu);
    matrix[n_pred, n_obs] K_mu_pred_obs =
      gp_exp_quad_cov(t_pred, t_obs, magnitude_mu, length_scale_mu);
    matrix[n_pred, n_pred] K_mu_pred =
      gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);

    matrix[n_obs, n_obs] K_eta_obs =
      gp_exp_quad_cov(t_obs, magnitude_eta, length_scale_eta);
    matrix[n_pred, n_obs] K_eta_pred_obs =
      gp_exp_quad_cov(t_pred, t_obs, magnitude_eta, length_scale_eta);
    matrix[n_pred, n_pred] K_eta_pred =
      gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);

    matrix[n_obs, n_obs] Sigma0 = add_diag(n_group / (n_group - 1.0) * K_eta_obs,
                                           max([square(sigma), 1e-8]));
    matrix[n_obs, n_obs] Sigma1 = add_diag(n_group * K_mu_obs, max([square(sigma), 1e-8]));
    matrix[n_obs, n_obs] L0 = cholesky_decompose(Sigma0);
    matrix[n_obs, n_obs] L1 = cholesky_decompose(Sigma1);

    vector[n_pred] mu_mean_post =
      K_mu_pred_obs * cholesky_left_divide_vec(L1, y_obs * one_mat_n_group[, 1]);
    matrix[n_pred, n_pred] mu_cov_post =
      K_mu_pred - n_group * K_mu_pred_obs * cholesky_left_divide_mat(L1, K_mu_pred_obs');

    matrix[n_obs, n_group] prod0 = cholesky_left_divide_mat(L0, y_obs);
    vector[n_pred_except_1_group] eta_mean_post = to_vector((n_group * K_eta_pred_obs * prod0 -
                                                       K_eta_pred_obs * prod0 *
                                                       one_mat_n_group) /
                                                      (n_group - 1.0))[1:n_pred_except_1_group];

    matrix[n_pred, n_pred] quad_form_eta = crossprod(mdivide_left_tri_low(L0, K_eta_pred_obs'));
    matrix[n_pred, n_pred] off_diagonal_block_eta =
      1/(n_group-1.0) * (n_group/(n_group-1.0) * quad_form_eta - K_eta_pred);
    matrix[n_pred, n_pred] diagonal_block_eta =
      n_group/(n_group-1.0) * (K_eta_pred -
                               n_group/(n_group-1.0) * quad_form_eta) +
                                    off_diagonal_block_eta;
    
    return (mu_mean_post, mu_cov_post, eta_mean_post,
            cholesky_iter_AB(diagonal_block_eta, off_diagonal_block_eta, n_group-1));
  }

  tuple(vector, matrix)
  mu_eta_irreg_base_post(vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                         array[] real t_a, array[] real t_b, array[] real t_pred,
                         int n_a, int n_b, int J_pred,
                         int J_a, array[] int J_b, array[] int t_b_is,
                         real magnitude_mu, real length_scale_mu,
                         real magnitude_eta, real length_scale_eta,
                         real sigma)
  {
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;

    matrix[J_pred, J_pred] K_mu_pred =
      gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);
    matrix[J_pred, J_pred] K_eta_pred =
      gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);
    matrix[J_pred, J_a] K_mu_pred_a =
      gp_exp_quad_cov(t_pred, t_a, magnitude_mu, length_scale_mu);
    matrix[J_pred, J_a] K_eta_pred_a =
      gp_exp_quad_cov(t_pred, t_a, magnitude_eta, length_scale_eta);
    matrix[J_pred, N_b] K_eta_pred_b =
      gp_exp_quad_cov(t_pred, t_b, magnitude_eta, length_scale_eta);

    matrix[N, N] L = Sigma_chol_irreg_base(t_a, t_b,
                                           n_a, n_b,
                                           J_a, J_b, t_b_is,
                                           magnitude_mu, length_scale_mu,
                                           magnitude_eta, length_scale_eta,
                                           sigma);

    matrix[J_pred, N_a] C_mua = block_rep(K_mu_pred_a, 1, n_a);
    matrix[J_pred, N_b] C_mub = gp_exp_quad_cov(t_pred, t_b, magnitude_mu, length_scale_mu);

    matrix[J_pred, N_a + N_b] C_muy = append_col(C_mua, C_mub);
    matrix[N_a + N_b, J_pred] Sigmainv_Cmuyt = cholesky_left_divide_mat(L, C_muy');
    vector[J_pred] mu_mean_post = Sigmainv_Cmuyt' * y_ab_vec;
    matrix[J_pred, J_pred] mu_cov_post =
      K_mu_pred - C_muy * Sigmainv_Cmuyt;

    matrix[J_pred, J_pred] off_diagonal_block_eta = - K_eta_pred / (n - 1);
    int N_a_pred = n_a * J_pred;
    int N_b_pred = n_b * J_pred;
    int N_pred = N_a_pred + N_b_pred;
    matrix[N_pred, N_pred] C_eta = block_mat_AB(K_eta_pred, off_diagonal_block_eta, n);

    matrix[J_pred, J_a] off_diag_etay = - K_eta_pred_a / (n - 1);
    matrix[N_b_pred, N_b] C_etab_yb;
    int col_start, col_end, row_start, row_end;
    for (i_col in 1:n_b) {
      col_start = t_b_slice_lwr(i_col, t_b_is);
      col_end = t_b_slice_upr(i_col, t_b_is);
      matrix[J_pred, J_b[i_col]] diag_i = K_eta_pred_b[, col_start:col_end];
      matrix[J_pred, J_b[i_col]] off_diag_i = - diag_i / (n - 1);
      for (i_row in 1:n_b) {
        row_start = (i_row-1) * J_pred + 1;
        row_end = row_start + J_pred - 1;
        if (i_row == i_col)
          C_etab_yb[row_start:row_end, col_start:col_end] = diag_i;
        else
          C_etab_yb[row_start:row_end, col_start:col_end] = off_diag_i;
      }
    }

    matrix[N_pred, N] C_etay =
      block_mat_2x2(
                    block_mat_AB(K_eta_pred_a, off_diag_etay, n_a),
                    block_rep(off_diag_etay, n_b, n_a),
                    block_rep(- K_eta_pred_b / (n - 1), n_a, 1),
                    C_etab_yb
                    );

    matrix[N, N_pred] Cyinv_Cyeta = cholesky_left_divide_mat(L, C_etay');
    vector[N_pred] eta_mean_post = Cyinv_Cyeta' * y_ab_vec;
    matrix[N_pred, N_pred] eta_cov_post = C_eta - C_etay * Cyinv_Cyeta;

    /* Cross covariance */
    matrix[J_pred, N_pred] minus_Cmuy_Cyinv_Cyeta = - C_muy * Cyinv_Cyeta;

    int N_pred_except_1_group = N_pred - J_pred;
    int mu_eta_dim = J_pred + N_pred_except_1_group;

    vector[mu_eta_dim] mu_eta_mean_post =
      append_row(mu_mean_post, eta_mean_post[:N_pred_except_1_group]);
    matrix[mu_eta_dim, mu_eta_dim] mu_eta_cov_post;
    mu_eta_cov_post[:J_pred, :J_pred] = mu_cov_post;
    mu_eta_cov_post[(J_pred+1):, (J_pred+1):] =
      eta_cov_post[:N_pred_except_1_group, :N_pred_except_1_group];
    mu_eta_cov_post[:J_pred, (J_pred+1):] =
      minus_Cmuy_Cyinv_Cyeta[, :N_pred_except_1_group];
    mu_eta_cov_post[(J_pred+1):, :J_pred] =
      mu_eta_cov_post[:J_pred, (J_pred+1):]';

    return (mu_eta_mean_post, mu_eta_cov_post);
  }

  tuple(vector, matrix, vector, matrix)
  mu_eta_completely_irreg_post(vector y_vec,
                               array[] real t, array[] real t_pred,
                               int n, int J_pred,
                               array[] int J, array[] int t_is,
                               real magnitude_mu, real length_scale_mu,
                               real magnitude_eta, real length_scale_eta,
                               real sigma)
  {
    int N = sum(J);
    int N_pred = n * J_pred;

    matrix[J_pred, J_pred] K_mu_pred =
      gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);
    matrix[J_pred, J_pred] K_eta_pred =
      gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);
    matrix[J_pred, N] K_eta_pred_t =
      gp_exp_quad_cov(t_pred, t, magnitude_eta, length_scale_eta);

    matrix[N, N] B = irregular_cov_mat_B(N, n, n,
                                         J, t, t_is,
                                         magnitude_mu, length_scale_mu,
                                         magnitude_eta, length_scale_eta,
                                         sigma);

    matrix[N, N] L = cholesky_decompose(B);

    matrix[J_pred, N] C_muy = gp_exp_quad_cov(t_pred, t, magnitude_mu, length_scale_mu);

    matrix[N, J_pred] Sigmainv_Cmuyt = cholesky_left_divide_mat(L, C_muy');
    vector[J_pred] mu_mean_post = Sigmainv_Cmuyt' * y_vec;
    matrix[J_pred, J_pred] mu_cov_post =
      K_mu_pred - C_muy * Sigmainv_Cmuyt;

    matrix[J_pred, J_pred] off_diagonal_block_eta = - K_eta_pred / (n - 1);      

    matrix[N_pred, N_pred] C_eta =
      block_mat_AB(K_eta_pred, off_diagonal_block_eta, n);

    matrix[N_pred, N] C_etay;
    int col_start, col_end, row_start, row_end;
    for (i_col in 1:n) {
      col_start = t_b_slice_lwr(i_col, t_is);
      col_end = t_b_slice_upr(i_col, t_is);
      matrix[J_pred, J[i_col]] diag_i = K_eta_pred_t[, col_start:col_end];
      matrix[J_pred, J[i_col]] off_diag_i = - diag_i / (n - 1);
      for (i_row in 1:n) {
        row_start = (i_row-1) * J_pred + 1;
        row_end = row_start + J_pred - 1;
        if (i_row == i_col)
          C_etay[row_start:row_end, col_start:col_end] = diag_i;
        else
          C_etay[row_start:row_end, col_start:col_end] = off_diag_i;
      }
    }
      
    matrix[N, N_pred] Cyinv_Cyeta = cholesky_left_divide_mat(L, C_etay');
    vector[N_pred] eta_mean_post = Cyinv_Cyeta' * y_vec;
    matrix[N_pred, N_pred] eta_cov_post = C_eta - C_etay * Cyinv_Cyeta;

    int N_pred_except_1_group = N_pred - J_pred;
    return (mu_mean_post, mu_cov_post,
            eta_mean_post[:N_pred_except_1_group],
            eta_cov_post[:N_pred_except_1_group, :N_pred_except_1_group]);
  }

  tuple(vector, matrix)
  eta_mu_irreg_smart_post(matrix y_a, vector y_a_vec, vector y_b_vec, vector y_ab_vec,
                          array[] real t_a, array[] real t_b, array[] real t_pred,
                          int n_a, int n_b, int J_pred,
                          int J_a, array[] int J_b, array[] int t_b_is,
                          real magnitude_mu, real length_scale_mu,
                          real magnitude_eta, real length_scale_eta,
                          real sigma,
                          matrix one_mat_n_a)
  {
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;

    tuple(vector[N_a],
          vector[N_b],
          matrix[J_a, J_a],
          matrix[J_a, J_a],
          matrix[N_b, N_b],
          matrix[J_pred, N]) Saiy_Sbiy_L0_L1_LSchur =
      irreg_smart_Sainvy_Sbinvy_L0_L1_LSchur_extra(y_a, y_a_vec,
                                                   y_b_vec, y_ab_vec,
                                                   t_a, t_b,
                                                   n_a, n_b,
                                                   J_a, J_b, t_b_is,
                                                   magnitude_mu, length_scale_mu,
                                                   magnitude_eta, length_scale_eta,
                                                   sigma,
                                                   one_mat_n_a,
                                                   J_pred, t_pred);

    matrix[J_pred, J_a] K_mu_pred_a =
      gp_exp_quad_cov(t_pred, t_a, magnitude_mu, length_scale_mu);
    matrix[J_pred, N_b] K_mu_pred_b =
      gp_exp_quad_cov(t_pred, t_b, magnitude_mu, length_scale_mu);

    vector[J_pred] mu_mean_post =
      K_mu_pred_a *
      (to_matrix(Saiy_Sbiy_L0_L1_LSchur.1, J_a, n_a) * one_mat_n_a[, 1]) +
      K_mu_pred_b * Saiy_Sbiy_L0_L1_LSchur.2;

    matrix[N_b, J_pred] Sinv_Cbmu =
      cholesky_left_divide_mat(
                               Saiy_Sbiy_L0_L1_LSchur.5,
                               K_mu_pred_b'
                               );

    matrix[N_b, J_a] K_mu_b_a =
      gp_exp_quad_cov(t_b, t_a, magnitude_mu, length_scale_mu);
    matrix[N_b, J_a] K_eta_b_a =
      gp_exp_quad_cov(t_b, t_a, magnitude_eta, length_scale_eta);
    matrix[J_a, J_pred] A1inv_Kmuapred =
      cholesky_left_divide_mat(Saiy_Sbiy_L0_L1_LSchur.4, K_mu_pred_a');
    matrix[N_b, J_pred] C_Ainv_Camu =
      n_a * (K_mu_b_a - K_eta_b_a / (n - 1)) * A1inv_Kmuapred;
    matrix[N_b, J_pred] Sinv_C_Ainv_Camu =
      cholesky_left_divide_mat(Saiy_Sbiy_L0_L1_LSchur.5, C_Ainv_Camu);

    matrix[J_pred, J_pred] K_mu_pred =
      gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);

    matrix[J_pred, J_pred] Cmua_Ainv_Ct_Sinv_Cbmu =
      Sinv_C_Ainv_Camu' * K_mu_pred_b';

    matrix[J_pred, J_pred] mu_cov_post =
      K_mu_pred
      - n_a * K_mu_pred_a * A1inv_Kmuapred
      - C_Ainv_Camu' * Sinv_C_Ainv_Camu
      + Cmua_Ainv_Ct_Sinv_Cbmu
      + Cmua_Ainv_Ct_Sinv_Cbmu'
      - K_mu_pred_b * Sinv_Cbmu;

    matrix[J_pred, J_a] K_eta_pred_a =
      gp_exp_quad_cov(t_pred, t_a, magnitude_eta, length_scale_eta);
    matrix[J_pred, N_b] K_eta_pred_b =
      gp_exp_quad_cov(t_pred, t_b, magnitude_eta, length_scale_eta);

    matrix[J_pred, n_a] Ketapreda_veciSaiy =
      K_eta_pred_a * to_matrix(Saiy_Sbiy_L0_L1_LSchur.1, J_a, n_a);

    int N_a_pred = n_a * J_pred;
    int N_b_pred = n_b * J_pred;

    vector[N_b_pred] nmin1_Cetabyb_Sbiy = (n - 1) *
      to_vector(Cetabyb_slice_multiply(K_eta_pred_b,
                                       to_matrix(Saiy_Sbiy_L0_L1_LSchur.2, N_b, 1),
                                       t_b, t_b_is, n_b, n));

    int N_pred = N_a_pred + N_b_pred;
    vector[N_pred] eta_mean_post =
      append_row(
                 n * to_vector(Ketapreda_veciSaiy) -
                 to_vector(Ketapreda_veciSaiy * one_mat_n_a) -
                 to_vector(block_rep(to_matrix(K_eta_pred_b * Saiy_Sbiy_L0_L1_LSchur.2, J_pred, 1),
                                     n_a, 1)),
                 -to_vector(Ketapreda_veciSaiy * rep_matrix(1, n_a, n_b)) +
                 nmin1_Cetabyb_Sbiy
                 ) / (n - 1);

    matrix[J_pred, J_pred] K_eta_pred =
      gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);

    matrix[J_a, J_pred] A1inv_Ketaapred = cholesky_left_divide_mat(Saiy_Sbiy_L0_L1_LSchur.4,
                                                                   K_eta_pred_a');

    matrix[J_a, N_b] Cb = gp_exp_quad_cov(t_a, t_b, magnitude_mu, length_scale_mu) -
      gp_exp_quad_cov(t_a, t_b, magnitude_eta, length_scale_eta) / (n-1);

    matrix[J_pred, J_pred] Ketapreda_A0inv_Ketaapred =
      crossprod(mdivide_left_tri_low(Saiy_Sbiy_L0_L1_LSchur.3, K_eta_pred_a'));

    matrix[J_pred, N_b] Ketapreda_A1inv_Cb_Sinv = A1inv_Ketaapred' *
      cholesky_left_divide_mat(Saiy_Sbiy_L0_L1_LSchur.5, Cb')';

    matrix[J_pred, J_pred] Ketapreda_A1inv_Cb_Sinv_Ketabpred =
      Ketapreda_A1inv_Cb_Sinv * K_eta_pred_b';

    matrix[J_pred, J_pred] term_to_transpose =
      (n-n_a)/(n-1)^2 * Ketapreda_A1inv_Cb_Sinv_Ketabpred;

    matrix[J_pred, J_pred] Ketapreda_A1inv_Cb_Sinv_Cbt_A1inv_Ketaapred =
      Ketapreda_A1inv_Cb_Sinv * Cb' * A1inv_Ketaapred;

    matrix[N_b, J_pred] LSinv_Ketabpred =
      mdivide_left_tri_low(Saiy_Sbiy_L0_L1_LSchur.5, K_eta_pred_b');

    matrix[J_pred, J_pred] Ketapreda_A1inv_Ketaapred =
      K_eta_pred_a * A1inv_Ketaapred;

    matrix[J_pred, J_pred] off_diag_eta = - K_eta_pred / (n-1) -
      (n-n_a)^2/(n_a*(n-1.0)^2) * Ketapreda_A1inv_Ketaapred +
      n^2/(n_a*(n-1.0)^2) * Ketapreda_A0inv_Ketaapred -
      ((n-n_a)/(n-1.0))^2 * Ketapreda_A1inv_Cb_Sinv_Cbt_A1inv_Ketaapred -
      term_to_transpose -
      term_to_transpose' -
      crossprod(LSinv_Ketabpred) / (n-1)^2;

    matrix[J_pred, J_pred] diag_eta =
      n/(n-1.0) * K_eta_pred - (n/(n-1.0))^2 * Ketapreda_A0inv_Ketaapred +
      off_diag_eta;

    matrix[N_b_pred, N_a_pred] eta_cov_post_lower_left =
      block_rep(block_rep(- K_eta_pred / (n-1) - (
                          - (n - n_a) / (n - 1.0)^2 * K_eta_pred_a * A1inv_Ketaapred -
                          n_a*(n-n_a)/(n-1)^2 * Ketapreda_A1inv_Cb_Sinv_Cbt_A1inv_Ketaapred -
                          n_a / (n-1)^2 * Ketapreda_A1inv_Cb_Sinv_Ketabpred
                                                  ),
                          n_b, 1) -
                Cetabyb_slice_multiply(K_eta_pred_b,
                                       -(n-n_a)/(n-1.0) * Ketapreda_A1inv_Cb_Sinv' -
                                       mdivide_right_tri_low(LSinv_Ketabpred',
                                                             Saiy_Sbiy_L0_L1_LSchur.5)' / (n-1),
                                       t_b, t_b_is, n_b, n),
                1, n_a);

    matrix[N_pred, N_pred] L_eta_cov_post;

    /* Upper left block */
    L_eta_cov_post[:N_a_pred, :N_a_pred] =
      cholesky_iter_AB(diag_eta, off_diag_eta, n_a);

    /* Lower left block */    
    L_eta_cov_post[(N_a_pred+1):, :N_a_pred] =
      mdivide_left_tri_low(L_eta_cov_post[:N_a_pred, :N_a_pred],
                           eta_cov_post_lower_left')';

    /* Smart division to find lower right block */
    
    matrix[N_b_pred, J_pred] scaled_Cetabyb_Sinv_Cbt_A1inv_Ketaapred =
      Cetabyb_slice_multiply(K_eta_pred_b,
                             n_a/(n-1.0) * Ketapreda_A1inv_Cb_Sinv',
                             t_b, t_b_is, n_b, n);

    matrix[N_b, N_b_pred] LSinv_Cybetab;
    int col_start, col_end, col_start_old, col_end_old;
    col_start = 1;
    col_end = J_pred;
    matrix[N_b, J_pred] Delta = rep_matrix(0, N_b, J_pred);
    {
      matrix[N_b, J_pred] rh_side1;
      rh_side1[1:t_b_slice_upr(1, t_b_is), ] =
        K_eta_pred_b[, 1:t_b_slice_upr(1, t_b_is)]';
      rh_side1[t_b_slice_lwr(2, t_b_is):, ] =
        - K_eta_pred_b[, t_b_slice_lwr(2, t_b_is):]' / (n-1);
      LSinv_Cybetab[, col_start:col_end] =
        mdivide_left_tri_low(Saiy_Sbiy_L0_L1_LSchur.5, rh_side1);
      Delta[1:t_b_slice_upr(1, t_b_is), ] =
        n/(n-1.0) * rh_side1[1:t_b_slice_upr(1, t_b_is), ];
    }
    for (i in 2:n_b) {
      col_start_old = col_start;
      col_end_old = col_end;
      col_start = col_start_old + J_pred;
      col_end = col_end_old + J_pred;
      Delta[t_b_slice_lwr(i-1, t_b_is):t_b_slice_upr(i-1, t_b_is), ] *= -1;
      Delta[t_b_slice_lwr(i, t_b_is):t_b_slice_upr(i, t_b_is), ] =
        n/(n-1.0) * K_eta_pred_b[, t_b_slice_lwr(i, t_b_is):
                                 t_b_slice_upr(i, t_b_is)]';
      LSinv_Cybetab[, col_start:col_end] =
        LSinv_Cybetab[, col_start_old:col_end_old];
      LSinv_Cybetab[t_b_slice_lwr(i-1, t_b_is):, col_start:col_end] +=
        mdivide_left_tri_low(Saiy_Sbiy_L0_L1_LSchur.5[t_b_slice_lwr(i-1, t_b_is):,
                                                      t_b_slice_lwr(i-1, t_b_is):],
                             Delta[t_b_slice_lwr(i-1, t_b_is):, ]);
    }

    matrix[N_b_pred, N_b_pred] eta_cov_post_lower_right =
      block_mat_AB(K_eta_pred, - K_eta_pred / (n-1), n_b)
      - (
         block_rep(block_rep(n_a/(n-1.0)^2 * Ketapreda_A1inv_Ketaapred +
                             (n_a / (n-1.0))^2 *
                             Ketapreda_A1inv_Cb_Sinv_Cbt_A1inv_Ketaapred,
                             1, n_b) +
                   scaled_Cetabyb_Sinv_Cbt_A1inv_Ketaapred',
                   n_b, 1) +
         block_rep(scaled_Cetabyb_Sinv_Cbt_A1inv_Ketaapred, 1, n_b) +
         crossprod(LSinv_Cybetab)
         );
      
    /* Lower right block */
    /* the tcrossprod gives C * Ainv * Ct */
    /* eta_cov_post_lower_right is B */
    L_eta_cov_post[(N_a_pred+1):, (N_a_pred+1):] =
      cholesky_decompose(add_diag(eta_cov_post_lower_right -
                                  tcrossprod(L_eta_cov_post[(N_a_pred+1):, :N_a_pred]),
                                  1e-8));

    /* Upper right block is 0 */
    L_eta_cov_post[:N_a_pred, (N_a_pred+1):] =
      rep_matrix(0, N_a_pred, N_b_pred);

    /* Cross covariance */

    matrix[J_pred, J_a] off_diag_etay = - K_eta_pred_a / (n - 1);
    matrix[N_b_pred, N_b] C_etab_yb;
    int row_start, row_end;
    for (i_col in 1:n_b) {
      col_start = t_b_slice_lwr(i_col, t_b_is);
      col_end = t_b_slice_upr(i_col, t_b_is);
      matrix[J_pred, J_b[i_col]] diag_i = K_eta_pred_b[, col_start:col_end];
      matrix[J_pred, J_b[i_col]] off_diag_i = - diag_i / (n - 1);
      for (i_row in 1:n_b) {
        row_start = (i_row-1) * J_pred + 1;
        row_end = row_start + J_pred - 1;
        if (i_row == i_col)
          C_etab_yb[row_start:row_end, col_start:col_end] = diag_i;
        else
          C_etab_yb[row_start:row_end, col_start:col_end] = off_diag_i;
      }
    }
    
    matrix[N_pred, N] C_etay =
      block_mat_2x2(
                    block_mat_AB(K_eta_pred_a, off_diag_etay, n_a),
                    block_rep(off_diag_etay, n_b, n_a),
                    block_rep(- K_eta_pred_b / (n - 1), n_a, 1),
                    C_etab_yb
                    );

    matrix[J_pred, N_pred] cross_cov_mu_eta = -Saiy_Sbiy_L0_L1_LSchur.6 * C_etay';

    /* Construct final result */
    /* We do a single block Cholesky step to combine the chol(cov_eta) with the
       cross covariance and chol(mu) to get Cholesky of combine covariance
       matrix for (eta, mu) */

    int N_pred_except_1_group = N_pred - J_pred;
    int mu_eta_dim = J_pred + N_pred_except_1_group; /* this is just N_pred */
    vector[mu_eta_dim] eta_mu_mean_post =
      append_row(eta_mean_post[:N_pred_except_1_group], mu_mean_post);
    matrix[mu_eta_dim, mu_eta_dim] L_eta_mu_cov_post;

    /* Upper left  corner is just chol(cov_eta) */
    L_eta_mu_cov_post[:N_pred_except_1_group, :N_pred_except_1_group] =
      L_eta_cov_post[:N_pred_except_1_group, :N_pred_except_1_group];

    /* Lower left corner */
    L_eta_mu_cov_post[(N_pred_except_1_group+1):, :N_pred_except_1_group] =
      mdivide_left_tri_low(L_eta_mu_cov_post[:N_pred_except_1_group,
                                             :N_pred_except_1_group],
                           cross_cov_mu_eta[, :N_pred_except_1_group]')';

    /* Lower right corner */
    /* Here we can reuse the lower left corner to calculate the quadratic term
       for the Schur complement. */
    L_eta_mu_cov_post[(N_pred_except_1_group+1):, (N_pred_except_1_group+1):] =
      cholesky_decompose(add_diag(mu_cov_post -
                                  tcrossprod(L_eta_mu_cov_post[(N_pred_except_1_group+1):,
                                                               :N_pred_except_1_group]),
                                  1e-8));

    /* Upper right corner is just zero */
    L_eta_mu_cov_post[:N_pred_except_1_group, (N_pred_except_1_group+1):] =
      rep_matrix(0, N_pred_except_1_group, J_pred);

    return (eta_mu_mean_post, L_eta_mu_cov_post);
  }

  /* Utility functions */

  /* Calculate Cetabyb times mat, by using sliced calculation where you multiply
     each K_eta_pred_bi by relevant columns of mat and then take linear
     combinations to avoid recalculating the same matrix product twice */
  matrix
  Cetabyb_slice_multiply(matrix K_eta_pred_b, matrix mat,
                         array[] real t_b, array[] int t_b_is,
                         int n_b, int n)
  {
    int J_pred = rows(K_eta_pred_b);
    int mat_cols = cols(mat);
    array[n_b] matrix[J_pred, mat_cols] Ketapredb_mat_slices;
    int col_start, col_end;
    for (i in 1:n_b) {
      col_start = t_b_slice_lwr(i, t_b_is);
      col_end = t_b_slice_upr(i, t_b_is);
      Ketapredb_mat_slices[i] =
        K_eta_pred_b[, col_start:col_end] * mat[col_start:col_end, ];
    }
    
    /* initiate Cetabyb_mat and fill out first part */
    int N_b_pred = n_b * J_pred;
    matrix[N_b_pred, mat_cols] Cetabyb_mat;
    int start_old, end_old, start, end;
    start = 1;
    end = J_pred;
    Cetabyb_mat[start:end, ] = Ketapredb_mat_slices[1];
    for (i in 2:n_b)
      Cetabyb_mat[start:end, ] =
        Cetabyb_mat[start:end, ] -
        Ketapredb_mat_slices[i] / (n-1);
    /* calculate other products by iterative modifications */
    for (i in 2:n_b) {
      start_old = start;
      end_old = end;
      start = end_old + 1;
      end = end_old + J_pred;
      Cetabyb_mat[start:end, ] =
        Cetabyb_mat[start_old:end_old, ] -
        Ketapredb_mat_slices[i-1] * (n/(n-1.0)) +
        Ketapredb_mat_slices[i] * (n/(n-1.0));
    }

    return Cetabyb_mat;
  }

  /* Performs iterative Cholesky decomposition of an AB-block matrix (meaning
     with a single matrix A in all diagonal blocks and a single matrix B in all
     off diagonal blocks).
  */
  matrix
  cholesky_iter_AB(matrix diagonal_block, matrix off_diagonal_block, int n_blocks)
  {
    int J = rows(diagonal_block);
    /* Fill out upper 2x2 matrix (to prepare for general iterations) */
    int N = J * n_blocks;
    matrix[J, J] S_i;
    matrix[N, N] L;
    int row_col_start_old = 1;
    int row_col_end_old = J;
    int row_col_start = row_col_start_old + J;
    int row_col_end = row_col_end_old + J;
    /* Upper left = chol(A) */
    L[:row_col_end_old, :row_col_end_old] =
      cholesky_decompose(add_diag(diagonal_block, 1e-8));
    /* Upper right block is 0 */
    L[:row_col_end_old, row_col_start:row_col_end] =
      rep_matrix(0, row_col_end_old, J);
    /* Lower left = B chol(A)^(-T) */
    L[row_col_start:row_col_end, :row_col_end_old] =
      mdivide_left_tri_low(L[:row_col_end_old, :row_col_end_old],
                           off_diagonal_block)';
    /* Lower right = chol(S_2) */
    matrix[J, J] quad_term_Schur =
      tcrossprod(L[row_col_start:row_col_end, :row_col_end_old]);
    L[row_col_start:row_col_end, row_col_start:row_col_end] =
      cholesky_decompose(add_diag(diagonal_block - quad_term_Schur, 1e-8));

    for (i in 3:n_blocks) {
      row_col_start_old = row_col_start;
      row_col_end_old = row_col_end;
      row_col_start = row_col_end + 1;
      row_col_end = row_col_start + J - 1;

      /* Upper right block is 0 */
      L[:row_col_end_old, row_col_start:row_col_end] =
        rep_matrix(0, row_col_end_old, J);

      /* Lower left corner */
      /* Add previously calculated chol(M_{i-2}) \ (1_{i-2} kronecker B) */
      L[row_col_start:row_col_end, :(row_col_end_old-J)] =
        L[row_col_start_old:row_col_end_old, :(row_col_end_old-J)];
      /* Do small left division to fill out the rest */
      L[row_col_start:row_col_end, (row_col_end_old-J+1):row_col_end_old] =
        mdivide_left_tri_low(
                             /* chol(S_{i-1}) (lower right corner from previous
                                iteration) */
                             L[row_col_start_old:row_col_end_old,
                                       row_col_start_old:row_col_end_old],
                             off_diagonal_block - quad_term_Schur
                             )';

      /* Lower right corner */
      quad_term_Schur = tcrossprod(L[row_col_start:row_col_end, :row_col_end_old]);
      /* chol(S_i) */      
      L[row_col_start:row_col_end, row_col_start:row_col_end] =
        cholesky_decompose(add_diag(diagonal_block - quad_term_Schur,
                                    1e-8));
    }

    return L;    
  }

  tuple(vector, vector, matrix, matrix, matrix)
  irreg_smart_Sainvy_Sbinvy_L0_L1_LSchur(matrix y_a, vector y_a_vec,
                                         vector y_b_vec, vector y_ab_vec,
                                         array[] real t_a, array[] real t_b,
                                         int n_a, int n_b,
                                         int J_a, array[] int J_b, array[] int t_b_is,
                                         real magnitude_mu, real length_scale_mu,
                                         real magnitude_eta, real length_scale_eta,
                                         real sigma,
                                         matrix one_mat_n_a)
  {
    int col_start, col_end, row_start, row_end;
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;
    
    matrix[J_a, J_a] K_mu_a = gp_exp_quad_cov(t_a, magnitude_mu, length_scale_mu);
    matrix[J_a, J_a] K_eta_a = gp_exp_quad_cov(t_a, magnitude_eta, length_scale_eta);
    matrix[J_a, J_a] A0 = add_diag(n / (n - 1.0) * K_eta_a, max([square(sigma), 1e-8]));
    matrix[J_a, J_a] A1 = add_diag(
      n_a * K_mu_a + (n_b / (n - 1.0)) * K_eta_a,
      max([square(sigma), 1e-8]));
    matrix[J_a, J_a] L0 = cholesky_decompose(A0);
    matrix[J_a, J_a] L1 = cholesky_decompose(A1);
    
    matrix[N_a, N_b] Ainv_Ct;
    matrix[N_b, N_b] C_Ainv_Ct;
    col_start = 0;
    col_end = 0;
    for (i_b in 1:n_b) {
      col_start = col_end + 1;
      col_end = col_start + J_b[i_b] - 1;
      array[J_b[i_b]] real t_b_slice_i = t_b_slice(i_b, t_b, t_b_is);
      matrix[J_a, J_b[i_b]] C_b_i =
        gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_mu, length_scale_mu) -
          gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_eta, length_scale_eta) /
          (n - 1.0);
      matrix[J_a, J_b[i_b]] A1inv_C_b_i = cholesky_left_divide_mat(L1, C_b_i);
      for (j in 1:n_a) {
        Ainv_Ct[((j-1)*J_a + 1):(j*J_a), col_start:col_end] = A1inv_C_b_i;
      }
      row_start = 0;
      row_end = 0;
      for (j_b in 1:i_b) {
        row_start = row_end + 1;
        row_end = row_start + J_b[j_b] - 1;
        matrix[col_end - col_start + 1, row_end - row_start + 1] block_i_j =
          C_b_i' * Ainv_Ct[1:J_a, row_start:row_end];
        C_Ainv_Ct[col_start:col_end, row_start:row_end] = block_i_j;
        if (j_b != i_b) {
          C_Ainv_Ct[row_start:row_end, col_start:col_end] = block_i_j';
        }
      }
    }
    C_Ainv_Ct *= n_a;

    matrix[N_b, N_b] B = irregular_cov_mat_B(N_b, n_b, n,
                                             J_b, t_b, t_b_is,
                                             magnitude_mu, length_scale_mu,
                                             magnitude_eta, length_scale_eta,
                                             sigma);

    matrix[N_b, N_b] L_Schur = cholesky_decompose(B - C_Ainv_Ct);

    vector[N_b] Schurinv_yb = cholesky_left_divide_vec(L_Schur, y_b_vec);
    vector[N_b] Schurinv_CAinv_ya = cholesky_left_divide_vec(L_Schur, Ainv_Ct' * y_a_vec);
    
    vector[N_a] Sigmaainv_y = calc_Ainv_y_a(L0, L1, y_a, one_mat_n_a) + Ainv_Ct *
      Schurinv_CAinv_ya - Ainv_Ct * Schurinv_yb;
    vector[N_b] Sigmabinv_y = Schurinv_yb - Schurinv_CAinv_ya;

    return (Sigmaainv_y, Sigmabinv_y, L0, L1, L_Schur);
  }

  /* The last entry is Cmua_Piainv + Cmub_Pibinv for irregular
     cross-covariance*/
  tuple(vector, vector, matrix, matrix, matrix, matrix)
  irreg_smart_Sainvy_Sbinvy_L0_L1_LSchur_extra(matrix y_a, vector y_a_vec,
                                               vector y_b_vec, vector y_ab_vec,
                                               array[] real t_a, array[] real t_b,
                                               int n_a, int n_b,
                                               int J_a, array[] int J_b, array[] int t_b_is,
                                               real magnitude_mu, real length_scale_mu,
                                               real magnitude_eta, real length_scale_eta,
                                               real sigma,
                                               matrix one_mat_n_a,
                                               int J_pred,
                                               array[] real t_pred)
  {
    int col_start, col_end, row_start, row_end;
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;
    
    matrix[J_a, J_a] K_mu_a = gp_exp_quad_cov(t_a, magnitude_mu, length_scale_mu);
    matrix[J_a, J_a] K_eta_a = gp_exp_quad_cov(t_a, magnitude_eta, length_scale_eta);
    matrix[J_a, J_a] A0 = add_diag(n / (n - 1.0) * K_eta_a, max([square(sigma), 1e-8]));
    matrix[J_a, J_a] A1 = add_diag(
      n_a * K_mu_a + (n_b / (n - 1.0)) * K_eta_a,
      max([square(sigma), 1e-8]));
    matrix[J_a, J_a] L0 = cholesky_decompose(A0);
    matrix[J_a, J_a] L1 = cholesky_decompose(A1);
    
    matrix[N_a, N_b] Ainv_Ct;
    matrix[N_b, N_b] C_Ainv_Ct;
    col_start = 0;
    col_end = 0;
    for (i_b in 1:n_b) {
      col_start = col_end + 1;
      col_end = col_start + J_b[i_b] - 1;
      array[J_b[i_b]] real t_b_slice_i = t_b_slice(i_b, t_b, t_b_is);
      matrix[J_a, J_b[i_b]] C_b_i =
        gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_mu, length_scale_mu) -
          gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_eta, length_scale_eta) /
          (n - 1.0);
      matrix[J_a, J_b[i_b]] A1inv_C_b_i = cholesky_left_divide_mat(L1, C_b_i);
      for (j in 1:n_a) {
        Ainv_Ct[((j-1)*J_a + 1):(j*J_a), col_start:col_end] = A1inv_C_b_i;
      }
      row_start = 0;
      row_end = 0;
      for (j_b in 1:i_b) {
        row_start = row_end + 1;
        row_end = row_start + J_b[j_b] - 1;
        matrix[col_end - col_start + 1, row_end - row_start + 1] block_i_j =
          C_b_i' * Ainv_Ct[1:J_a, row_start:row_end];
        C_Ainv_Ct[col_start:col_end, row_start:row_end] = block_i_j;
        if (j_b != i_b) {
          C_Ainv_Ct[row_start:row_end, col_start:col_end] = block_i_j';
        }
      }
    }
    C_Ainv_Ct *= n_a;

    matrix[N_b, N_b] B = irregular_cov_mat_B(N_b, n_b, n,
                                             J_b, t_b, t_b_is,
                                             magnitude_mu, length_scale_mu,
                                             magnitude_eta, length_scale_eta,
                                             sigma);

    matrix[N_b, N_b] L_Schur = cholesky_decompose(B - C_Ainv_Ct);

    vector[N_b] Schurinv_yb = cholesky_left_divide_vec(L_Schur, y_b_vec);
    vector[N_b] Schurinv_CAinv_ya = cholesky_left_divide_vec(L_Schur, Ainv_Ct' * y_a_vec);
    
    vector[N_a] Sigmaainv_y = calc_Ainv_y_a(L0, L1, y_a, one_mat_n_a) + Ainv_Ct *
      Schurinv_CAinv_ya - Ainv_Ct * Schurinv_yb;
    vector[N_b] Sigmabinv_y = Schurinv_yb - Schurinv_CAinv_ya;

    /* Extra terms needed only for cross covariance */

    matrix[J_pred, J_a] K_mu_pred_a =
      gp_exp_quad_cov(t_pred, t_a, magnitude_mu, length_scale_mu);

    matrix[J_pred, N] Cmua_Piainv;
    /* Term (3), using that Ainv_Ct[:J_a, ] is A1inv * Cb */
    Cmua_Piainv[, (N_a+1):] =
      -n_a * cholesky_left_divide_mat(L_Schur,
                                      (K_mu_pred_a * Ainv_Ct[:J_a, ])')';
    Cmua_Piainv[, :N_a] =
      block_rep(cholesky_left_divide_mat(L1, K_mu_pred_a')', 1, n_a) + /* Term (1) */
      -Cmua_Piainv[, (N_a+1):] * Ainv_Ct';                             /* Term (2) */

    matrix[J_pred, N] Cmub_Pibinv;
    matrix[N_b, J_pred] Cbmu =
      gp_exp_quad_cov(t_b, t_pred, magnitude_mu, length_scale_mu);
    Cmub_Pibinv[, (N_a + 1):] =
      cholesky_left_divide_mat(L_Schur, Cbmu)';
    Cmub_Pibinv[, :N_a] =
      -Cmub_Pibinv[, (N_a + 1):] * Ainv_Ct';
    
    return (Sigmaainv_y, Sigmabinv_y, L0, L1, L_Schur, Cmua_Piainv + Cmub_Pibinv);
  }

  real
  multi_normal_lpdf_prepared(int n_obs_total, real log_det, real quad_form)
  {
    return -0.5 * (n_obs_total * log(2*pi()) + log_det + quad_form);
  }

  /* returns last eta to make them sum to 0 */
  vector
  get_last_eta(matrix mu_eta_pred)
  {
    int n_group = cols(mu_eta_pred) - 1;
    return mu_eta_pred[, 2:n_group] * rep_vector(-1, n_group - 1);
  }

  /* Return block matrix with repetitions of same diagonal block and
     off diagonal block */
  matrix
  block_mat_AB(matrix diagonal_block, matrix off_diagonal_block, int n_blocks)
  {
    int block_rows = rows(diagonal_block);
    int block_cols = cols(diagonal_block);
    matrix[n_blocks * block_rows, n_blocks * block_cols] block_mat;
    int col_start, col_end, row_start, row_end;
    for (i_col in 1:n_blocks) {
      col_start = (i_col-1) * block_cols + 1;
      col_end = col_start + block_cols - 1;
      for (i_row in 1:n_blocks) {
        row_start = (i_row-1) * block_rows + 1;
        row_end = row_start + block_rows - 1;
        if (i_row == i_col)
          block_mat[row_start:row_end, col_start:col_end] = diagonal_block;
        else          
          block_mat[row_start:row_end, col_start:col_end] = off_diagonal_block;
      }
    }

    return block_mat;
  }

  matrix block_mat_2x2(matrix upper_left, matrix lower_left,
                       matrix upper_right, matrix lower_right)
  {
    int ul_rows = rows(upper_left);
    int ul_cols = cols(upper_left);
    int lr_rows = rows(lower_right);
    int lr_cols = cols(lower_right);
    matrix[ul_rows + lr_rows, ul_cols + lr_cols] block_mat;
    block_mat[:ul_rows, :ul_cols] = upper_left;
    block_mat[:ul_rows, (ul_cols+1):] = upper_right;
    block_mat[(ul_rows+1):, :ul_cols] = lower_left;
    block_mat[(ul_rows+1):, (ul_cols+1):] = lower_right;
    return block_mat;
  }

  matrix block_mat_2x2_symmetric(matrix upper_left, matrix upper_right,
                                 matrix lower_right)
  {
    return block_mat_2x2(upper_left, upper_right', upper_right, lower_right);
  }

  /* Return block matrix with n_rows x n_cols blocks of the matrix block */
  matrix
  block_rep(matrix block, int n_rows, int n_cols)
  {
    int block_cols = cols(block);
    int block_rows = rows(block);
    int col_start, col_end, row_start, row_end;
    matrix[n_rows * block_rows, n_cols * block_cols] result;
    for (i_row in 1:n_rows) {
      row_start = (i_row-1) * block_rows + 1;
      row_end = row_start + block_rows - 1;
      for (i_col in 1:n_cols) {
        col_start = (i_col-1) * block_cols + 1;
        col_end = col_start + block_cols - 1;
        result[row_start:row_end, col_start:col_end] = block;
      }
    }

    return result;
  }

  matrix
  Sigma_chol_irreg_base(array[] real t_a, array[] real t_b,
                        int n_a, int n_b,
                        int J_a, array[] int J_b, array[] int t_b_is,
                        real magnitude_mu, real length_scale_mu,
                        real magnitude_eta, real length_scale_eta,
                        real sigma)
  {
    int col_start, col_end, row_start, row_end;
    int n = n_a + n_b;
    int N_a = n_a * J_a;
    int N_b = sum(J_b);
    int N = N_a + N_b;
    matrix[J_a, J_a] K_mu_a = gp_exp_quad_cov(t_a, magnitude_mu, length_scale_mu);
    matrix[J_a, J_a] K_eta_a = gp_exp_quad_cov(t_a, magnitude_eta, length_scale_eta);
    matrix[J_a, J_a] diagonal_block_A = add_diag(K_mu_a + K_eta_a,
                                                 max([square(sigma), 1e-8]));
    matrix[J_a, J_a] off_diagonal_block_A = K_mu_a - K_eta_a / (n - 1.0);
    matrix[N_a, N_a] A = block_mat_AB(diagonal_block_A, off_diagonal_block_A, n_a);
    matrix[N_b, N_b] B = irregular_cov_mat_B(N_b, n_b, n,
                                             J_b, t_b, t_b_is,
                                             magnitude_mu, length_scale_mu,
                                             magnitude_eta, length_scale_eta,
                                             sigma);
    
    matrix[J_a, N_b] Ct_row;
    col_start = 0;
    col_end = 0;
    for (i in 1:n_b) {
      col_start = col_end + 1;
      col_end = col_start + J_b[i] - 1;
      array[J_b[i]] real t_b_slice_i = t_b_slice(i, t_b, t_b_is);
      Ct_row[, col_start:col_end] =
        gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_mu, length_scale_mu) -
        gp_exp_quad_cov(t_a, t_b_slice_i, magnitude_eta, length_scale_eta) /
        (n - 1.0);
    }

    matrix[N, N] Sigma;
    Sigma[:N_a, :N_a] = A;
    Sigma[(1+N_a):, (1+N_a):] = B;
    Sigma[:N_a, (1+N_a):] = block_rep(Ct_row, n_a, 1);
    Sigma[(1+N_a):, :N_a] = Sigma[:N_a, (1+N_a):]';

    return cholesky_decompose(Sigma);
  }

  matrix
  irregular_cov_mat_B(int N_b, int n_b, int n,
                      array[] int J_b, array[] real t_b, array[] int t_b_is,
                      real magnitude_mu, real length_scale_mu,
                      real magnitude_eta, real length_scale_eta,
                      real sigma)
  {
    matrix[N_b, N_b] B;
    int col_start = 0;
    int col_end = 0;
    int row_start = 0;
    int row_end = 0;
    for (i in 1:n_b) {
      col_start = col_end + 1;
      col_end = col_start + J_b[i] - 1;
      row_start = 0;
      row_end = 0;
      array[J_b[i]] real t_b_slice_i = t_b_slice(i, t_b, t_b_is);
      for (j in 1:n_b) {
        row_start = row_end + 1;
        row_end = row_start + J_b[j] - 1;
        if (i == j) {
          B[col_start:col_end, row_start:row_end] = add_diag(
            gp_exp_quad_cov(t_b_slice_i, magnitude_mu, length_scale_mu) +
              gp_exp_quad_cov(t_b_slice_i, magnitude_eta, length_scale_eta),
            max([square(sigma), 1e-8]));
        } else {
          array[J_b[j]] real t_b_slice_j = t_b_slice(j, t_b, t_b_is);
          B[col_start:col_end, row_start:row_end] =
            gp_exp_quad_cov(t_b_slice_i,
                            t_b_slice_j,
                            magnitude_mu, length_scale_mu) -
            gp_exp_quad_cov(t_b_slice_i,
                            t_b_slice_j,
                            magnitude_eta, length_scale_eta) / (n - 1.0);
        }
      }
    }

    return B;
  }
  
  vector
  calc_Ainv_y_a(matrix L0, matrix L1, matrix y, matrix one_mat)
  {
    int n = cols(y); /* y is y_a, so this is n_a in thesis notation */
    int J = rows(y);
    int N = num_elements(y);
    matrix[J, n] prod0 = cholesky_left_divide_mat(L0, y);
    matrix[J, n] prod1 = cholesky_left_divide_mat(L1, y);
    return to_vector(prod0 + 1.0 / n * (prod1 - prod0) * one_mat);
  }

  int
  t_b_slice_lwr(int i, array[] int t_b_is)
  {
    return t_b_is[i]+1;
  }

  int
  t_b_slice_upr(int i, array[] int t_b_is)
  {
    return t_b_is[i+1];
  }

  /* Slice i is of size J_b[i] */
  array[] real
  t_b_slice(int i, array[] real t_b, array[] int t_b_is)
  {
    return t_b[t_b_slice_lwr(i, t_b_is):t_b_slice_upr(i, t_b_is)];
  }

  matrix
  cholesky_left_divide_mat(matrix L, matrix A)
  {
    return mdivide_right_tri_low(
      mdivide_left_tri_low(L, A)',
      L
    )';
  }

  vector
  cholesky_left_divide_vec(matrix L, vector v)
  {
    return mdivide_right_tri_low(
      mdivide_left_tri_low(L, v)',
      L
    )';
  }

  real
  cholesky_log_det(matrix L)
  {
    return 2 * sum(log(diagonal(L)));
  }

  /* Experiments */

  matrix
  post_cross_covariance(vector y_vec,
                        array[] real t, array[] real t_pred,
                        int n, int J_pred,
                        array[] int J, array[] int t_is,
                        real magnitude_mu, real length_scale_mu,
                        real magnitude_eta, real length_scale_eta,
                        real sigma)
  {
    int N = sum(J);
    int N_pred = n * J_pred;

    matrix[J_pred, J_pred] K_mu_pred =
      gp_exp_quad_cov(t_pred, magnitude_mu, length_scale_mu);
    matrix[J_pred, J_pred] K_eta_pred =
      gp_exp_quad_cov(t_pred, magnitude_eta, length_scale_eta);
    matrix[J_pred, N] K_eta_pred_t =
      gp_exp_quad_cov(t_pred, t, magnitude_eta, length_scale_eta);

    matrix[N, N] B = irregular_cov_mat_B(N, n, n,
                                         J, t, t_is,
                                         magnitude_mu, length_scale_mu,
                                         magnitude_eta, length_scale_eta,
                                         sigma);

    matrix[J_pred, N] C_muy = gp_exp_quad_cov(t_pred, t, magnitude_mu, length_scale_mu);

    matrix[N_pred, N] C_etay;
    int col_start, col_end, row_start, row_end;
    for (i_col in 1:n) {
      col_start = t_b_slice_lwr(i_col, t_is);
      col_end = t_b_slice_upr(i_col, t_is);
      matrix[J_pred, J[i_col]] diag_i = K_eta_pred_t[, col_start:col_end];
      matrix[J_pred, J[i_col]] off_diag_i = - diag_i / (n - 1);
      for (i_row in 1:n) {
        row_start = (i_row-1) * J_pred + 1;
        row_end = row_start + J_pred - 1;
        if (i_row == i_col)
          C_etay[row_start:row_end, col_start:col_end] = diag_i;
        else
          C_etay[row_start:row_end, col_start:col_end] = off_diag_i;
      }
    }

    return (C_muy / B) * C_etay';
  }
}

