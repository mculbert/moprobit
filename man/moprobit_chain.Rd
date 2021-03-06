% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auxiliary.R
\name{moprobit_chain}
\alias{moprobit_chain}
\title{MCMC Sampling for the Multivariate Ordinal Probit Model}
\usage{
moprobit_chain(state, N, thin = 1, clean = FALSE, ...)
}
\arguments{
\item{state}{A moprobit state container, generated by \code{\link[=moprobit_init]{moprobit_init()}}.}

\item{N}{The number of states to return.}

\item{thin}{The number of iterations to run between returned states. A value of 1 will return a state container for every iteration; a value of 2 will return the state for every other iteration, etc.}

\item{clean}{If \code{TRUE}, the returned states (except for the final state) will be passed through \code{\link[=clean_state]{clean_state()}} to conserve memory.}

\item{...}{Additional parameters to pass to \code{\link[=moprobit_init]{moprobit_init()}}, such as \code{fixSigma} or \code{fixCrossBlockCov}}
}
\value{
A list of \code{N} moprobit state containers
}
\description{
MCMC Sampling for the Multivariate Ordinal Probit Model
}
\seealso{
\code{\link[=moprobit_init]{moprobit_init()}}, \code{\link[=moprobit_iter]{moprobit_iter()}}, \code{\link[=clean_state]{clean_state()}}
}
