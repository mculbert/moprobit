
#' Constrained Inverse Wishart Distribution Sampling
#' 
#' Returns one random sample from the Constrained Inverse Wishart distribution
#' 
#' @param nu The degrees of freedom parameter.
#' @param S The sums-and-cross-products parameter.
#' @param fixed A logical vector (recycled, as necessary) indicating which dimensions should be constrained.
#' @param eps A value taken as "machine zero".
#' 
#' @return A list containing the lower Cholesky factor of a draw from the variance-constrained inverse-Wishart distribution, CIW(nu, S), with its inverse
rCIW <- function(nu, S, fixed = T, eps = sqrt(.Machine$double.eps)) {
  stopifnot(nrow(S) == ncol(S), length(nu) == 1, nu > nrow(S)-1, eps > 0)
  if (length(fixed) > nrow(S)) {
    warn("fixed is longer than the dimension of S, only using the first ", nrow(S), "elements")
    fixed <- fixed[1:nrow(S)]
  }
  if (nrow(S) %% length(fixed) != 0) {
    warn("The dimension of S is not a multiple of the length of fixed")
  }
  
  internal_rCIW(as.numeric(nu), as.matrix(S), as.logical(fixed), as.numeric(eps))
}
