% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rCIW.R
\name{rCIW}
\alias{rCIW}
\title{Constrained Inverse Wishart Distribution Sampling}
\usage{
rCIW(nu, S, fixed = T, eps = sqrt(.Machine$double.eps))
}
\arguments{
\item{nu}{The degrees of freedom parameter.}

\item{S}{The sums-and-cross-products parameter.}

\item{fixed}{A logical vector (recycled, as necessary) indicating which dimensions should be constrained.}

\item{eps}{A value taken as "machine zero".}
}
\value{
A list containing the lower Cholesky factor of a draw from the variance-constrained inverse-Wishart distribution, CIW(nu, S), with its inverse
}
\description{
Returns one random sample from the Constrained Inverse Wishart distribution
}
