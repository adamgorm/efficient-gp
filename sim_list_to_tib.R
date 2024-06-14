sim_list_to_tib <- function(sim_list) {
    
    tib <- tibble()
    observed <- rep(FALSE, length(sim_list$t_all))
    observed[sim_list$ti_a] <- TRUE
    for (i in 1:sim_list$n_a) {
        fi <- sim_list$f[1, , i]
        yi <- sim_list$y[1, , i]
        tib <- bind_rows(tib,
                         tibble(t = sim_list$t_all,
                                type = "a",
                                group = i,
                                group_id = paste0("a", i),
                                f = fi,
                                y = yi,
                                observed = observed))
    }
    for (i in 1:sim_list$n_b) {
        observed <- rep(FALSE, length(sim_list$t_all))
        observed[sim_list$ti_b[[i]]] <- TRUE
        fi <- sim_list$f[1, , sim_list$n_a + i]
        yi <- sim_list$y[1, , sim_list$n_a + i]
        tib <- bind_rows(tib,
                         tibble(t = sim_list$t_all,
                                type = "b",
                                group = i,
                                group_id = paste0("b", i),
                                f = fi,
                                y = yi,
                                observed = observed))
    }

    return(tib)
}
