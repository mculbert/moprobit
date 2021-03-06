#library(dplyr)
#library(tidyr)

# Helper function to find the mid-points between thresholds
mid.pt <- function(tau) c(-.5, tau[-length(tau)] + diff(tau)/2, tau[length(tau)] + .5)

#' MCMC Sampling for the Multivariate Ordinal Probit Model
#' 
#' @param state A moprobit state container, generated by [moprobit_init()].
#' @param N The number of states to return.
#' @param thin The number of iterations to run between returned states. A value of 1 will return a state container for every iteration; a value of 2 will return the state for every other iteration, etc.
#' @param clean If `TRUE`, the returned states (except for the final state) will be passed through [clean_state()] to conserve memory.
#' @param ... Additional parameters to pass to [moprobit_init()], such as `fixSigma` or `fixCrossBlockCov`
#' 
#' @return A list of `N` moprobit state containers
#' 
#' @seealso [moprobit_init()], [moprobit_iter()], [clean_state()]
moprobit_chain <- function(state, N, thin = 1, clean = FALSE, ...) {
  if (N < 1) return(list())
  trace <- list(moprobit_iter(state, thin, ...))
  for (i in 2:n) {
    trace[[i]] <- moprobit_iter(trace[[i-1]], thin, ...)
    if (clean) trace[[i-1]] <- clean_state(trace[[i-1]])
  }
  return(trace)
}

state_to_vector <- function(state) {
  setNames(c(as.vector(state$beta),
             as.vector(state$Sigma[lower.tri(state$Sigma, diag=F)]),
             diag(state$Sigma)[state$env$Y.continuous],
             do.call('c', lapply(state$tau[which(state$env$Y.factor & state$env$K > 2)], function(tau) tau[-1]))),
           c(paste0('beta[', outer(rownames(state$beta), colnames(state$beta), paste, sep=', '), ']'),
             paste0('Sigma[', outer(rownames(state$Sigma), colnames(state$Sigma), paste, sep=', ')[lower.tri(state$Sigma, diag=F)], ']'),
             if (sum(state$env$Y.continuous) > 0)
               paste0('sigma[', rownames(state$Sigma)[state$env$Y.continuous], ']') else NULL,
             if (sum(state$env$Y.factor & (state$env$K > 2)) > 0)
               paste0('tau[', paste(rep(names(state$tau)[which(state$env$Y.factor & state$env$K > 2)],
                                        state$env$K[which(state$env$Y.factor & state$env$K > 2)]-2),
                                    do.call('c', lapply(state$env$K[which(state$env$Y.factor & state$env$K > 2)],
                                                        function(K) 2:(K-1))),
                                    sep=', '), ']') else NULL ))
}

#' Extract the parameters from a list of moprobit state containers
#' 
#' @param trace A list of moprobit state containers
#' 
#' @return A matrix of parameter values, rows corresponding to iterations and columns to parameters
#' 
#' @seealso [moprobit_chain()], [moprobit_iter()], [get_toc()], [as.mcmc()]
chain_to_mcmc <- function(trace) as.mcmc(t(sapply(trace, state_to_vector)))

normalize_parameters <- function(par, env = NULL) {
  if (!is.null(env))
    q.block <- setNames(colnames(env$q.block)[apply(env$q.block, 1, which)], rownames(env$q.block))
  as.data.frame(par) %>%
    mutate(i = row_number()) %>%
    gather(param, val, -i) %>%
    group_by(param) %>%
    mutate(val_std = (val - mean(val)) / sd(val)) %>%
    ungroup() %>%
    mutate(param = factor(param, levels=colnames(par)),
           var = sub('^(\\w+)\\b.*$', '\\1', param),
           var = factor(var, levels=unique(var)),
           row = ifelse(grepl('\\[', param), sub('^.*\\[([^,]+),?.*\\].*$', '\\1', param), ''),
           row = factor(row, levels=unique(row)),
           col = ifelse(grepl(',', param), sub('^.*\\[.*,\\s*(.+)\\].*$', '\\1', param), ''),
           col = factor(col, levels=unique(col)),
           block = if (is.null(env)) NA else
             ifelse(var == 'beta', q.block[as.character(col)],
                    ifelse(var == 'sigma' | var == 'tau', q.block[as.character(row)],
                           ifelse(var == 'Sigma' & q.block[as.character(row)] == q.block[as.character(col)],
                                  q.block[as.character(row)], NA)))
           ) %>%
    arrange(as.integer(param), i)
}

#' Obtain metainformation about the multivariate ordinal probit model parameters
#' 
#' @param par A matrix returned from [chain_to_mcmc()]
#' @param env The MCMC environment from a state container returned from [moprobit_init()]
#' 
#' @return A data frame describing the parameters
#' 
#' @details FIXME
#' 
#' @seealso [chain_to_mcmc()]
get_toc <- function(par, env = NULL) {
  if (!is.null(env))
    q.block <- setNames(colnames(env$q.block)[apply(env$q.block, 1, which)], rownames(env$q.block))
  data.frame(j = 1:ncol(par), param = factor(colnames(par), levels=colnames(par))) %>%
    mutate(var = sub('^(\\w+)\\b.*$', '\\1', param),
           var = factor(var, levels=unique(var)),
           row = ifelse(grepl('\\[', param), sub('^.*\\[([^,]+),?.*\\].*$', '\\1', param), ''),
           row = factor(row, levels=unique(row)),
           col = ifelse(grepl(',', param), sub('^.*\\[.*,\\s*(.+)\\].*$', '\\1', param), ''),
           col = factor(col, levels=unique(col)),
           block = if (is.null(env)) NA else
             ifelse(var == 'beta', q.block[as.character(col)],
                    ifelse(var == 'sigma' | var == 'tau', q.block[as.character(row)],
                           ifelse(var == 'Sigma' & q.block[as.character(row)] == q.block[as.character(col)],
                                  q.block[as.character(row)], NA))),
           # Structural zeroes
           zero = if (is.null(env)) F else
             ifelse(var == 'beta', !env$p.block[cbind(match(as.character(row), rownames(env$p.block)),
                                                      match(as.character(block), colnames(env$p.block)) )], F)
           ) %>%
    filter(!zero) %>%
    select(-zero) %>%
    arrange(j)
}

#' Get the autocorrelation of the MCMC parameters
#' 
#' @param par A matrix of parameters, from [chain_to_mcmc()]
#' 
#' @return FIXME
get_acf <- function(par) {
  sapply(1:ncol(par), function(i) acf(par[,i], plot=F)$acf) %>%
    as.data.frame() %>%
    setNames(colnames(par)) %>%
    mutate(lag = row_number()-1) %>%
    gather(param, val, -lag) %>%
    mutate(param = factor(param, levels=colnames(par)),
           var = sub('^(\\w+)\\b.*$', '\\1', param),
           var = factor(var, levels=unique(var)),
           row = ifelse(grepl('\\[', param), sub('^.*\\[([^,]+),?.*\\].*$', '\\1', param), ''),
           row = factor(row, levels=unique(row)),
           col = ifelse(grepl(',', param), sub('^.*\\[.*,\\s*(.+)\\].*$', '\\1', param), ''),
           col = factor(col, levels=unique(col)) ) %>%
    arrange(as.integer(param), lag)
}

#' Rhat diagnostic
#'
#' @param par A matrix of parameters, from [chain_to_mcmc()]
#' 
#' @return A vector of Rhat for each parameter
#' 
#' FIXME. Gelman et al. 2014, p 284-285
Rhat <- function(par) {
  # If we have (n.iter x n.param), reformat into (n.iter x n.param x n.chain=1)
  if (length(dim(par)) == 2)
    par <- array(par, c(dim(par), 1), list(rownames(par), colnames(par), NULL))
  # Ensure number of iterations is even
  if (dim(par)[1] %% 2 == 1) par <- par[-1,,]
  
  n.chain <- dim(par)[3]*2
  n.iter <- dim(par)[1]/2
  n.param <- dim(par)[2]
  
  # Split chains into two halves, n.iter x n.chain x n.param
  x <- array(aperm(par, c(1,3,2)), c(n.iter, n.chain, n.param))
  # Chain means, n.chain x n.param
  x.bar <- apply(x, c(2, 3), mean)
  # Grand mean, n.param
  x.grand <- apply(x.bar, 2, mean)
  
  B <- n.iter / (n.chain-1) * apply((t(x.bar) - x.grand)^2, 1, sum)
  W <- apply((aperm(x, c(2, 3, 1)) - as.vector(x.bar))^2, 2, sum) / n.chain / (n.iter - 1)
  var.plus <- W * (n.iter - 1) / n.iter  +  B / n.iter
  return( setNames(sqrt(var.plus / W), colnames(par)) )
}
