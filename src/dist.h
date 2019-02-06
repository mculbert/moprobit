
#ifndef MOPROBIT_DIST_H
#define MOPROBIT_DIST_H

#ifdef MOPROBIT_DIST_USE_R
#include <Rcpp.h>
#else
// [[Rcpp::depends(BH)]]
#include <math.h>
#include <boost/math/special_functions.hpp>
#include "threefry.h"
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

#ifdef MOPROBIT_DIST_USE_R
  // Counter and Key are ignored when using R implementation
  typedef uint64_t Counter;
  typedef uint64_t Key;
#endif


/************ Uniform distribution ************/


inline double runif(const Counter &n, const Key &k)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::runif(0, 1);
#else
  return U64_TO_D01(threefry2x64_20(n, k).i0);
#endif
}

inline void runif2(const Counter &n, const Key &k, double &u1, double &u2)
{
#ifdef MOPROBIT_DIST_USE_R
  u1 = R::runif(0, 1);
  u2 = R::runif(0, 1);
#else
  Counter x = threefry2x64_20(n, k);
  u1 = U64_TO_D01(x.i0);
  u2 = U64_TO_D01(x.i1);
#endif
}

inline double runif(double a, double b, const Counter &n, const Key &k, double u = -1)
{
  if (a > b || isinf(a) || isinf(b)) return NAN;
  if (a == b) return a;
  if (u == -1) u = runif(n, k);
  return a + (b-a) * u;
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


inline double rnorm(const Counter &n, const Key &k, double u = -1)
{
#ifdef MOPROBIT_DIST_USE_R
  return R::rnorm(0, 1);
#else
  if (u == -1) u = runif(n, k);
  return qnorm(u);
#endif
}

inline double rnorm(double mu, double sd, const Counter &n, const Key &k, double u = -1)
{
  if (isinf(sd) || sd < 0) return NAN;
  if (sd == 0 || isinf(mu)) return mu;
  return mu + sd * rnorm(n, k, u);
}


/************ Truncated Normal distribution ************/


inline double rtruncnorm(double a, double b, const Counter &n, const Key &k, double u = -1)
{
  if (a >= b)
  {
    if (a == b) return a;
    return NAN;
  }
  a = pnorm(a);
  b = pnorm(b);
  if (u == -1) u = runif(n, k);
  return qnorm(a + (b-a) * u);
}

inline double rtruncnorm_lowertail(double b, const Counter &n, const Key &k, double u = -1) // a = -Inf
{
  b = pnorm(b);
  if (u == -1) u = runif(n, k);
  return qnorm(b * u);
}

inline double rtruncnorm_uppertail(double a, const Counter &n, const Key &k, double u = -1) // b = Inf
{
  a = pnorm(a);
  if (u == -1) u = runif(n, k);
  return qnorm(a + (1-a) * u);
}

inline double rtruncnorm(double a, double b, double mu, double sd, const Counter &n, const Key &k, double u = -1)
{
  if (sd <= 0)
  {
    if (sd == 0 && a <= mu && mu <= b) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm( (a - mu) / sd, (b - mu) / sd, n, k, u);
}

inline double rtruncnorm_lowertail(double b, double mu, double sd, const Counter &n, const Key &k, double u = -1) // a = -Inf
{
  if (sd <= 0)
  {
    if (sd == 0 && mu <= b) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm_lowertail( (b - mu) / sd, n, k, u);
}

inline double rtruncnorm_uppertail(double a, double mu, double sd, const Counter &n, const Key &k, double u = -1) // b = Inf
{
  if (sd <= 0)
  {
    if (sd == 0 && a <= mu) return mu;
    return NAN;
  }
  return mu + sd * rtruncnorm_uppertail( (a - mu) / sd, n, k, u);
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

inline double rInvGamma(double shape, double scale, const Counter &n, const Key &k, double u = -1)
{
#ifdef MOPROBIT_DIST_USE_R
  return 1. / R::rgamma(shape, 1. / scale);
#else
  if (u == -1) u = runif(n, k);
  return qInvGamma(u, shape, scale);
#endif
}


/************ Truncated Inverse Gamma distribution ************/


inline double rtruncInvGamma(double a, double b, double shape, double scale, const Counter &n, const Key &k, double u = -1)
{
  if (a < 0) a = 0;
  if (shape <= 0 || scale <= 0 || a >= b || b < 0)
  {
    if (scale > 0 && shape == 0 && a == 0 && b >= 0) return 0;
    if (scale > 0 && shape > 0 && a == b && a > 0) return a;
    return NAN;
  }
  a = pInvGamma(a, shape, scale);
  b = pInvGamma(b, shape, scale);
  if (u == -1) u = runif(n, k);
  return qInvGamma(a + (b-a) * u, shape, scale);
}

inline double rtruncInvGamma_lowertail(double b, double shape, double scale, const Counter &n, const Key &k, double u = -1) // a = 0
{
  if (shape <= 0 || scale <= 0 || b <= 0)
  {
    if (scale > 0 && shape >= 0 && b >= 0) return 0;
    return NAN;
  }
  b = pInvGamma(b, shape, scale);
  if (u == -1) u = runif(n, k);
  return qInvGamma(b * u, shape, scale);
}

inline double rtruncInvGamma_uppertail(double a, double shape, double scale, const Counter &n, const Key &k, double u = -1) // b = Inf
{
  if (a < 0) a = 0;
  if (shape <= 0 || scale <= 0 || isinf(a))
  {
    if (scale > 0 && shape == 0 && a == 0) return 0;
    return NAN;
  }
  a = pInvGamma(a, shape, scale);
  if (u == -1) u = runif(n, k);
  return qInvGamma(a + (1-a) * u, shape, scale);
}


} }

#endif
