/* moprobit: MCMC Estimation of Multivariate Ordinal Probit Models
 * Copyright 2019 Michael J. Culbertson <mculbertson@edanalytics.org>
 *
 *  moprobit is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  moprobit is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  To read a copy of the GNU General Public License, see
 *  <http://www.gnu.org/licenses/>.
 */


// [[Rcpp::depends(RcppEigen)]]

#include <cmath>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;

#include "dist.h"

void internal_rCIW_eigen(double nu, const Eigen::MatrixXd &S, const LogicalVector fix, double eps,
                        Eigen::MatrixXd &Lambda, Eigen::MatrixXd &Lambda_inv)
{
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::Lower;
  
  // Dimension of the problem
  const int p = S.rows();
  const double nu_2 = 0.5*nu;
  
  // Wishart_1(nu, S) = Gamma(nu/2, S/2)
  if (p == 1)
  {
    if (fix[0])
    {
      Lambda(0,0) = Lambda_inv(0,0) = 1;
      return;
    }
    else
    {
      const double g = sqrt(moprobit::dist::rInvGamma(nu_2, 0.5*S(0,0)));
      Lambda(0,0) = g;
      Lambda_inv(0,0) = 1./g;
      return;
    }
  }
  
  // Replicate fix as necessary so that it is p long
  const LogicalVector fix_v = rep_len(fix, p);
  const double eps2 = eps*eps;
  
  // Cholesky decomposition of S = LL'
  const Eigen::LLT<MatrixXd> S_llt(S);
  const MatrixXd L(S_llt.matrixL());
  
  // Inverse Cholesky decomposition L^{-1}
  const MatrixXd L_inv(S_llt.matrixL().solve(MatrixXd::Identity(p, p)));
  
  // Storage for covariance of the given row
  MatrixXd D(p, p);
  MatrixXd D_inv(p, p);
  RowVectorXd adjustment(p);
  
  // Initialize
  Lambda.setZero();
  Lambda_inv.setZero();
  D.triangularView<Lower>().setZero();
  D_inv.triangularView<Lower>().setZero();
  
  
#ifdef USE_R_TRUNCNORM
  // rtruncnorm:
  static const Rcpp::Function rtruncnorm("rtruncnorm");
#endif
  
  // First row
  if (fix_v[0])
    Lambda_inv(0,0) = Lambda(0,0) = 1;
  else
  {
    const double L_00 = L(0,0);
    const double g = sqrt(moprobit::dist::rInvGamma(nu_2, 0.5*(L_00*L_00)));
    Lambda(0,0) = g;
    Lambda_inv(0,0) = 1./g;
  }
  
  // Remaining rows, 1:(p-1)
  for (int i=1; i < p; i++)
  {
    
    if (fix_v[i])
    {
      // Generate lambda_ii^2 ~ Inv-Gamma(nu/2, l_ii^2/2) <= 1
      const double L_ii = L(i,i);
      const double L_ii_2 = 0.5*(L_ii*L_ii);
      const double lambda_ii = sqrt(moprobit::dist::rtruncInvGamma_lowertail(1, nu_2, L_ii_2));
      Lambda(i,i) = (lambda_ii < eps ? eps : lambda_ii);
      
      // Remaining portion of Lambda_i'Lambda_i = 1 - lambda_ii^2 available
      double R2 = 1 - lambda_ii*lambda_ii;
      
      // Generate elements (i-1):1
      //   Lambda_i ~ N_{i-1}(L_i (L_ii)^-1 Lambda_ii, lambda_ii^2 Lambda_ii' (L_ii^-1)' L_ii^-1 Lambda_ii)
      
      // Compute right lower-triangular factor of the covariance for this row: cov(Lambda_i) = D_ii'D_ii
      // D <- Lambda[i,i] * L_inv[ii,ii] %*% Lambda[ii,ii]
      // D_inv <- solve(D)
      D.block(i-1,0, 1,i) = L_inv.block(i-1,0, 1,i) * Lambda.topLeftCorner(i,i).triangularView<Lower>();
      D_inv.block(i-1,0, 1,i) = Lambda_inv.block(i-1,0, 1,i) * L.topLeftCorner(i,i).triangularView<Lower>();
      
      // Compute means for this row
      // mu <- L[i,ii] %*% L_inv[ii,ii] %*% Lambda[ii, ii]
      const RowVectorXd mu(L.block(i,0, 1,i) *
                           L_inv.topLeftCorner(i,i).triangularView<Lower>() *
                           Lambda.topLeftCorner(i,i).triangularView<Lower>() );
      
      // Mean for first element of this row (last to be generated)
      double mu0;
      
      if (i > 1)
      {
        // Each element of Lambda_i, except the first: (i-1):1
        for (int j=i-1; j > 0; j--)
        {
          // lambda_ij | lambda_i,(j+1):i ~ N(mu[j] + D_j' (D_jj')^-1 [lambda_i,(j+1):(i-1) - mu_(j+1):(i-1)]),
          //                                  D[j,j]^2) I[ lambda_ij^2 <= R2 ]
          if (R2 > eps2)
          {
            const double mu_j = mu[j] + (j < i-1 ? (adjustment.segment(j+1, i-j-1) * D.block(j+1,j, i-j-1,1))(0,0) : 0);
            const double sd_j = D(j,j) * lambda_ii;
            const double sqrt_R2 = sqrt(R2);
#ifdef USE_R_TRUNCNORM
            PutRNGstate();
            const double lambda_ij = NumericVector(rtruncnorm(1, -sqrt_R2, sqrt_R2, mu_j, sd_j))[0];
            GetRNGstate();
#else
            const double lambda_ij = moprobit::dist::rtruncnorm(-sqrt_R2, sqrt_R2, mu_j, sd_j);
#endif
            Lambda(i,j) = lambda_ij;
            R2 -= lambda_ij*lambda_ij;
          }
          else
            Lambda(i,j) = 0;
          
          // Lambda[i, j:(i-1)] * D_inv[j:(i-1), j]
          adjustment[j] = ((Lambda.block(i,j, 1,i-j) - mu.segment(j, i-j)) * D_inv.block(j,j, i-j,1))(0,0);
        }
        
        // Calculate mean for final element (lambda_i1)
        // mu <- mu[1] + crossprod(D_inv[-1,-1] %*% D[-1,1], Lambda[i,2:(i-1)] - mu[-1])
        //   Here, j == 0
        mu0 = mu[0] + (adjustment.segment(1, i-1) * D.block(1,0, i-1,1))(0,0);
        
      } else mu0 = mu[0];
      
      // Last element gets the remainder
      //   Determine the sign randomly by coin flip with
      //   weights proportional to N(mu, D^2) at +/- sqrt(R2)
      if (R2 > eps2)
      {
        const double r = sqrt(R2);
        const double sd0 = D(0,0)*lambda_ii;
        const double p_pos = R::dnorm(r, mu0, sd0, false);
        const double p_neg = R::dnorm(-r, mu0, sd0, false);
        // Perhaps should this be < eps, instead:
        if (p_pos + p_neg == 0)
        {
          // Both p_pos and p_neg have underflowed, so let's try to infer which is larger
          if (mu0 >= r)
            Lambda(i,0) = r;
          else if (mu0 <= -r)
            Lambda(i,0) = -r;
          else if (r-mu0 < mu0+r)
            Lambda(i,0) = r;
          else if (r-mu0 > mu0+r)
            Lambda(i,0) = -r;
          else
            Lambda(i,0) = (moprobit::dist::runif() < .5 ? r : -r);
        }
        else
          Lambda(i,0) = (moprobit::dist::runif() < p_pos / (p_pos + p_neg) ? r : -r);
      } else // R2 < eps2
        Lambda(i,0) = 0;
      
    } else {  // Not fixed
      
      // Update this row of D for future rows that are fixed
      D.block(i-1,0, 1,i) = L_inv.block(i-1,0, 1,i) * Lambda.topLeftCorner(i,i).triangularView<Lower>();
      D_inv.block(i-1,0, 1,i) = Lambda_inv.block(i-1,0, 1,i) * L.topLeftCorner(i,i).triangularView<Lower>();
      
      // lambda_ii ~ IGamma(nu/2, L[i,i]^2/2)
      const double L_ii = L(i, i);
      const double Lambda_ii = sqrt(moprobit::dist::rInvGamma(nu_2, 0.5*(L_ii*L_ii)));
      Lambda(i,i) = Lambda_ii;
      
      // Lambda_i ~ N_{i-1}(L_i (L_ii)^-1 Lambda_ii, lambda_ii^2 Lambda_ii' (L_ii^-1)' L_ii^-1 Lambda_ii)
      Lambda.block(i,0, 1,i) =
        L.block(i,0, 1,i) *
        L_inv.topLeftCorner(i, i).triangularView<Lower>() *
        Lambda.topLeftCorner(i, i).triangularView<Lower>() +
        as<Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic> > >(rnorm(i)) *
        L_inv.topLeftCorner(i, i).triangularView<Lower>() *
        Lambda.topLeftCorner(i, i).triangularView<Lower>() * Lambda_ii;
    }
    
    
    // Fill in Lambda_inv
    // Lambda_inv[i,ii] <- -(Lambda[i,ii] %*% Lambda_inv[ii,ii]) / Lambda[i,i]
    Lambda_inv.block(i,0, 1,i) = (Lambda.block(i,0, 1,i) * Lambda_inv.block(0,0, i,i)) / -Lambda(i,i);
    Lambda_inv(i,i) = 1./Lambda(i,i);
  } // Each row
  
} // internal_CIW_eigen


// [[Rcpp::export]]
List internal_rCIW(double nu, NumericMatrix S, LogicalVector fix, double eps)
{
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::Lower;
  
  // Dimension of the problem
  const int p = S.nrow();
  
  // Wishart_1(nu, S) = Gamma(nu/2, S/2)
  if (p == 1)
  {
    if (fix[0])
      return List::create(Named("Lambda") = NumericMatrix::diag(1, 1),
                          Named("Lambda_inv") = NumericMatrix::diag(1, 1));
    else
    {
      double g = sqrt(moprobit::dist::rInvGamma(0.5*nu, 0.5*S[0,0]));
      return List::create(Named("Lambda") = NumericMatrix::diag(1, g),
                          Named("Lambda_inv") = NumericMatrix::diag(1, 1./g));
    }
  }
  
  // Results to return
  MatrixXd Lambda(p, p);
  MatrixXd Lambda_inv(p, p);
  
  internal_rCIW_eigen(nu, as<Eigen::Map<MatrixXd> >(S), fix, eps, Lambda, Lambda_inv);
  
  return List::create(Named("Lambda") = wrap(Lambda), Named("Lambda_inv") = wrap(Lambda_inv));
}
