
#ifndef MOPROBIT_DIST_H
#define MOPROBIT_DIST_H

// FIXME: Switch to Threefry RNG
#include <Rcpp.h>

#ifdef MOPROBIT_DIST_USE_R
#include <Rcpp.h>
#else
// [[Rcpp::depends(BH)]]
#include <math.h>
#include <boost/math/special_functions.hpp>
#endif

#ifndef NAN
#define NAN (0./0.)
#endif

#ifndef INFINITY
#define INFINITY (1./0.)
#endif

#define SQRT_TWO (1.414213562373095048801688724209698078)
#define ONE_OVER_SQRT_TWO (0.7071067811865475244008443621048490392)

namespace moprobit { namespace dist {


/************ Uniform distribution ************/


inline double runif()
{
  return R::runif(0, 1);
}

inline double runif(double a, double b)
{
  if (a > b || isinf(a) || isinf(b)) return NAN;
  if (a == b) return a;
  return a + (b-a) * runif();
}


/************ Normal distribution ************/


inline double pnorm(double x)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::pnorm(x, 0, 1, 1, 0);
#else
  if (isinf(x)) return (x < 0 ? 0 : 1);
  return 0.5 * boost::math::erfc( -ONE_OVER_SQRT_TWO * x );
#endif
}


inline double pnorm(double x, double mu, double sd)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::pnorm(x, mu, sd, 1, 0);
#else
  if (sd <= 0)
  {
    if (sd == 0) return (x < mu ? 0 : 1);
    return NAN;
  }
  return pnorm( (x - mu) / sd );
#endif
}


inline double qnorm(double x)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::qnorm(x, 0, 1, 1, 0);
#else
  if (x <= 0 || x >= 1)
  {
    if (x == 0) return -INFINITY;
    if (x == 1) return INFINITY;
    return NAN;
  }
  return -SQRT_TWO * boost::math::erfc_inv(2*x);
#endif
}


inline double qnorm(double x, double mu, double sd)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::qnorm(x, mu, sd, 1, 0);
#else
  if (sd <= 0)
  {
    if (sd == 0) return mu;
    return NAN;
  }
  return mu + sd * qnorm(x);
#endif
}


inline double rnorm()
{
#ifdef MOPROBIT_DIST_USE_R
  return R::rnorm(0, 1);
#else
  return qnorm(runif());
#endif
}

inline double rnorm(double mu, double sd)
{
  if (isinf(sd) || sd < 0) return NAN;
  if (sd == 0 || isinf(mu)) return mu;
  return mu + sd * rnorm();
}


/************ Truncated Normal distribution ************/


inline double rtruncnorm(double a, double b)
{
  if (a >= b)
  {
    if (a == b) return a;
    return NAN;
  }
  return qnorm(runif(pnorm(a), pnorm(b)));
}

inline double rtruncnorm_lowertail(double b) // a = -Inf
{
  return qnorm(runif(0, pnorm(b)));
}

inline double rtruncnorm_uppertail(double a) // b = Inf
{
  return qnorm(runif(pnorm(a), 1));
}

inline double rtruncnorm(double a, double b, double mu, double sd)
{
  if (sd <= 0)
  {
    if (sd == 0 && a <= mu && mu <= b) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm( (a - mu) / sd, (b - mu) / sd );
}

inline double rtruncnorm_lowertail(double b, double mu, double sd) // a = -Inf
{
  if (sd <= 0)
  {
    if (sd == 0 && mu <= b) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm_lowertail( (b - mu) / sd );
}

inline double rtruncnorm_uppertail(double a, double mu, double sd) // b = Inf
{
  if (sd <= 0)
  {
    if (sd == 0 && a <= mu) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm_uppertail( (a - mu) / sd );
}


/************ Inverse Gamma distribution ************/


inline double pInvGamma(double x, double shape, double scale)
{
#ifdef MOPROBIT_DIST_USE_R
  // Note: R's {pqd}gamma functions use the scale = 1/rate parameterization
  return R::pgamma(1./x, shape, 1./scale, 0, 0);
#else
  if (x <= 0 || shape <= 0 || scale <= 0)
  {
    if (scale > 0 && (x == 0 || shape == 0)) return 0;
    return NAN;
  }
  if (isinf(x))
  {
    if (isinf(scale)) return NAN;
    return 1;
  }
  return boost::math::gamma_q(shape, scale / x);
#endif
}

inline double qInvGamma(double x, double shape, double scale)
{
#ifdef MOPROBIT_DIST_USE_R
  // Note: R's {pqd}gamma functions use the scale = 1/rate parameterization
  return 1. / R::qgamma(x, shape, 1./scale, 0, 0);
#else
  if (x < 0 || x > 1 || shape < 0 || scale <= 0) return NAN;
  if (shape == 0 || x == 0) return 0;
  if (x == 1) return INFINITY;
  return scale / boost::math::gamma_q_inv(shape, x);
#endif
}

inline double rInvGamma(double shape, double scale)
{
#ifdef MOPROBIT_DIST_USE_R
  return 1. / R::rgamma(shape, 1. / scale);
#else
  return qInvGamma(runif(), shape, scale);
#endif
}


/************ Truncated Inverse Gamma distribution ************/


inline double rtruncInvGamma(double a, double b, double shape, double scale)
{
  if (a < 0) a = 0;
  if (shape <= 0 || scale <= 0 || a >= b || b < 0)
  {
    if (scale > 0 && shape == 0 && a == 0 && b >= 0) return 0;
    if (scale > 0 && shape > 0 && a == b && a > 0) return a;
    return NAN;
  }
  return qInvGamma(runif(pInvGamma(a, shape, scale), pInvGamma(b, shape, scale)), shape, scale);
}

inline double rtruncInvGamma_lowertail(double b, double shape, double scale) // a = 0
{
  if (shape <= 0 || scale <= 0 || b <= 0)
  {
    if (scale > 0 && shape >= 0 && b >= 0) return 0;
    return NAN;
  }
  return qInvGamma(runif(0, pInvGamma(b, shape, scale)), shape, scale);
}

inline double rtruncInvGamma_uppertail(double a, double shape, double scale) // b = Inf
{
  if (a < 0) a = 0;
  if (shape <= 0 || scale <= 0 || isinf(a))
  {
    if (scale > 0 && shape == 0 && a == 0) return 0;
    return NAN;
  }
  return qInvGamma(runif(pInvGamma(a, shape, scale), 1), shape, scale);
}


} }

#endif
