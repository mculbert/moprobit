
#ifndef __MOPROBIT_CIW_H__
#define __MOPROBIT_CIW_H__

#include <RcppEigen.h>
#include "dist.h"

void internal_rCIW_eigen(double nu, const Eigen::MatrixXd &S, const LogicalVector fix, double eps,
                        Eigen::MatrixXd &Lambda, Eigen::MatrixXd &Lambda_inv,
                        moprobit::dist::Counter &n, const moprobit::dist::Key &key);

#ifndef MOPROBIT_DIST_USE_R
inline void internal_rCIW_eigen(double nu, const Eigen::MatrixXd &S, const LogicalVector fix, double eps,
                                Eigen::MatrixXd &Lambda, Eigen::MatrixXd &Lambda_inv,
                                uint64_t context, const moprobit::dist::Key &key)
{
  moprobit::dist::Counter n(context);
  internal_rCIW_eigen(nu, S, fix, eps, Lambda, Lambda_inv, n, key);
}
#endif

inline void internal_rCIW_eigen(double nu, const Eigen::MatrixXd &S, const LogicalVector fix, double eps,
                                Eigen::MatrixXd &Lambda, Eigen::MatrixXd &Lambda_inv,
                                const moprobit::dist::Key &key)
{
  moprobit::dist::Counter n;
  internal_rCIW_eigen(nu, S, fix, eps, Lambda, Lambda_inv, n, key);
}


List internal_rCIW(double nu, NumericMatrix S, LogicalVector fix, double eps);

#endif
