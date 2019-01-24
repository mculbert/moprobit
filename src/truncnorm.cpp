#include <Rcpp.h>

/***************
*  Adapted from truncnorm package (GPL-2.0-or-later)
*  by Björn Bornkamp <bornkamp@statistik.tu-dortmund.de> and Olaf Mersmann <olafm@statistik.uni-dortmund.de>
*  https://github.com/olafmersmann/truncnorm/blob/master/src/rtruncnorm.c
*/

#ifdef DEBUG
#define SAMPLER_DEBUG(N, A, B) Rprintf("%8s(%f, %f)\n", N, A, B)
#else
#define SAMPLER_DEBUG(N, A, B)
#endif

static const double t1 = 0.15;
static const double t2 = 2.18;
static const double t3 = 0.725;
static const double t4 = 0.45;

/* Exponential rejection sampling (a,inf) */
static R_INLINE double ers_a_inf(double a) {
  SAMPLER_DEBUG("ers_a_inf", a, R_PosInf);
  const double ainv = 1.0 / a;
  double x, rho;
  do {
    x = R::rexp(ainv) + a; /* rexp works with 1/lambda */
    rho = exp(-0.5 * pow((x - a), 2));
  } while (R::runif(0, 1) > rho);
  return x;
}

/* Exponential rejection sampling (a,b) */
static R_INLINE double ers_a_b(double a, double b) {
  SAMPLER_DEBUG("ers_a_b", a, b);
  const double ainv = 1.0 / a;
  double x, rho;
  do {
    x = R::rexp(ainv) + a; /* rexp works with 1/lambda */
    rho = exp(-0.5 * pow((x - a), 2));
  } while (R::runif(0, 1) > rho || x > b);
  return x;
}

/* Normal rejection sampling (a,b) */
static R_INLINE double nrs_a_b(double a, double b) {
  SAMPLER_DEBUG("nrs_a_b", a, b);
  double x = -DBL_MAX;
  while (x < a || x > b) {
    x = R::rnorm(0, 1);
  }
  return x;
}

/* Normal rejection sampling (a,inf) */
static R_INLINE double nrs_a_inf(double a) {
  SAMPLER_DEBUG("nrs_a_inf", a, R_PosInf);
  double x = -DBL_MAX;
  while (x < a) {
    x = R::rnorm(0, 1);
  }
  return x;
}

/* Half-normal rejection sampling */
static R_INLINE double hnrs_a_b(double a, double b) {
  SAMPLER_DEBUG("hnrs_a_b", a, b);
  double x = a - 1.0;
  while (x < a || x > b) {
    x = R::rnorm(0, 1);
    x = fabs(x);
  }
  return x;
}

/* Uniform rejection sampling */
static R_INLINE double urs_a_b(double a, double b) {
  SAMPLER_DEBUG("urs_a_b", a, b);
  const double phi_a = R::dnorm(a, 0.0, 1.0, FALSE);
  double x = 0.0, u = 0.0;

  /* Upper bound of normal density on [a, b] */
  const double ub = a < 0 && b > 0 ? M_1_SQRT_2PI : phi_a;
  do {
    x = R::runif(a, b);
  } while (R::runif(0, 1) * ub > R::dnorm(x, 0, 1, 0));
  return x;
}

/* Previously this was refered to as type 1 sampling: */
/* upper = Inf */
static R_INLINE double r_lefttruncnorm(double a, double mean, double sd) {
  const double alpha = (a - mean) / sd;
  if (alpha < t4) {
    return mean + sd * nrs_a_inf(alpha);
  } else {
    return mean + sd * ers_a_inf(alpha);
  }
}

/* lower = -Inf */
static R_INLINE double r_righttruncnorm(double b, double mean, double sd) {
  const double beta = (b - mean) / sd;
  /* Exploit symmetry: */
  return mean - sd * r_lefttruncnorm(-beta, 0.0, 1.0);
}

/* lower and upper finite */
static R_INLINE double r_truncnorm(double a, double b, double mean, double sd) {
  const double alpha = (a - mean) / sd;
  const double beta = (b - mean) / sd;
  const double phi_a = R::dnorm(alpha, 0.0, 1.0, FALSE);
  const double phi_b = R::dnorm(beta, 0.0, 1.0, FALSE);
  if (beta <= alpha) {
    return NA_REAL;
  } else if (alpha <= 0 && 0 <= beta) { /* 2 */
    if (phi_a <= t1 || phi_b <= t1) {   /* 2 (a) */
      return mean + sd * nrs_a_b(alpha, beta);
    } else { /* 2 (b) */
      return mean + sd * urs_a_b(alpha, beta);
    }
  } else if (alpha > 0) {      /* 3 */
    if (phi_a / phi_b <= t2) { /* 3 (a) */
      return mean + sd * urs_a_b(alpha, beta);
    } else {
      if (alpha < t3) { /* 3 (b) */
        return mean + sd * hnrs_a_b(alpha, beta);
      } else { /* 3 (c) */
        return mean + sd * ers_a_b(alpha, beta);
      }
    }
  } else {                     /* 3s */
    if (phi_b / phi_a <= t2) { /* 3s (a) */
      return mean - sd * urs_a_b(-beta, -alpha);
    } else {
      if (beta > -t3) { /* 3s (b) */
        return mean - sd * hnrs_a_b(-beta, -alpha);
      } else { /* 3s (c) */
        return mean - sd * ers_a_b(-beta, -alpha);
      }
    }
  }
}

/************************/
