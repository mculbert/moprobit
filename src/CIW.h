
#ifndef __MOPROBIT_CIW_H__
#define __MOPROBIT_CIW_H__

#include <RcppEigen.h>

void internal_rCIW_eigen(double nu, const Eigen::MatrixXd &S, const LogicalVector fix, double eps,
                        Eigen::MatrixXd &Lambda, Eigen::MatrixXd &Lambda_inv);

List internal_rCIW(double nu, NumericMatrix S, LogicalVector fix, double eps);

#endif
