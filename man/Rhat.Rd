% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auxiliary.R
\name{Rhat}
\alias{Rhat}
\title{Rhat diagnostic}
\usage{
Rhat(par)
}
\arguments{
\item{par}{A matrix of parameters, from \code{\link[=chain_to_mcmc]{chain_to_mcmc()}}}
}
\value{
A vector of Rhat for each parameter

FIXME. Gelman et al. 2014, p 284-285
}
\description{
Rhat diagnostic
}
