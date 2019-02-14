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
#include "CIW.h"

#ifdef MOPROBIT_DIST_USE_R
#define JUMP {}
#define COPY_I_TO_CNT {}
#else
#define JUMP cnt.jump()
#define COPY_I_TO_CNT cnt.i1 = i
#endif


inline Eigen::MatrixXd rnorm_matrix(int r, int c, moprobit::dist::Counter &cnt, const moprobit::dist::Key &key)
{
  Eigen::ArrayXd x(r, c);
  int num = r*c;
  for (int i = 0; i < num; i++) x[i] = moprobit::dist::rnorm(++cnt, key);
  return x.matrix();
}


// [[Rcpp::export]]
List internal_iter(const List prior, int iters, int TMN_iters, bool fixSigma, bool fixCrossBlockCov, double eps)
{
  using moprobit::dist::pnorm;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::Lower;
  typedef Eigen::Map<MatrixXd> MapMatrix;
  
  static const Function modelmatrix("model.matrix");

#ifdef MOPROBIT_DIST_USE_R
  const moprobit::dist::Key key = 0;
#else
  // Key to R's seed
  const moprobit::dist::Key key = moprobit::dist::Key::from_R();
#endif
  moprobit::dist::Counter cnt;

  // Extract the MCMC setup
  const Environment env(as<Environment>(prior["env"]));      // Constants
  const int G = env["G"];                           // Number of grade-level blocks
  const int *K = as<IntegerVector>(env["K"]).begin();  // Number of levels for each discrete variable (0 = continuous)
  const int *Ybool = as<LogicalVector>(env["Y.logical"]).begin();  // Flag that variable is logical
  
  const LogicalMatrix q_block = env["q.block"];     // Outcomes in each block (q x G)
  const LogicalMatrix p_block = env["p.block"];     // Covariates in each block (p x G)
  const DataFrame Xbase = env["X"];                 // Base predictors
  
  const int a = (env.exists("a") ? env["a"] : 0);   // Prior df for Sigma
  const LogicalVector fixed = env["Y.fixed"];       // Dimensions to constrain to unit variance
  const double sd_tau = env["sd_tau"];              // Standard deviation for proposal distribution for tau
  
  const List Yobs = env["Y.obs"];                   // List of q index vectors for observed cases for each variable
  const List Ymis = env["Y.mis"];                   // List of q index vectors for missing cases for each variable
  const List meas_err = env["meas_err"];          // List of reciprocal measurement error variances
  
  // Allocate memory for return state
  List state = List::create(Named("env") = env,
                            Named("Z") = clone(as<NumericMatrix>(prior["Z"])),
                            Named("Y") = clone(as<DataFrame>(prior["Y"])),
                            // wrap beta at end
                            Named("X") = clone(as<NumericMatrix>(prior["X"])),
                            Named("mu") = clone(as<NumericMatrix>(prior["mu"])),
                            Named("E") = clone(as<NumericMatrix>(prior["E"])),
                            // FIXME: Consider sparse Sigma?
                            Named("Sigma") = clone(as<NumericMatrix>(prior["Sigma"])),
                            Named("Omega") = clone(as<NumericMatrix>(prior["Omega"])),
                            Named("L_Sigma") = clone(as<NumericMatrix>(prior["L_Sigma"])),
                            Named("D_inv_Sigma") = clone(as<NumericMatrix>(prior["D_inv_Sigma"])),
                            Named("tau") = clone(as<List>(prior["tau"]))
  );
  
  // Eigen maps for current state
  MapMatrix Z(as<MapMatrix>(state["Z"]));      // [Latent variables under Y, Continuous outcomes] (n x q)
  DataFrame Y(as<DataFrame>(state["Y"]));      // Complete outcomes data (n x q)
  Eigen::SparseMatrix<double> beta(as<Eigen::SparseMatrix<double> >(prior["beta"])); // Regression coefficients (p x q)
  MapMatrix X(as<MapMatrix>(state["X"]));      // Covariates (n x p)
  MapMatrix mu(as<MapMatrix>(state["mu"]));    // Predicted values XB
  MapMatrix E(as<MapMatrix>(state["E"]));      // Residuals
  MapMatrix Sigma(as<MapMatrix>(state["Sigma"]));              // Residual covariance matrix (q x q)
  MapMatrix Omega(as<MapMatrix>(state["Omega"]));              // Precision matrix (q x q) = Sigma^-1
  MapMatrix L_Sigma(as<MapMatrix>(state["L_Sigma"]));          // Cholesky decomposition of residual covariance
  MapMatrix D_inv_Sigma(as<MapMatrix>(state["D_inv_Sigma"]));  // Inverse Cholesky decomposition of residual covariance (D_inv_Sigma = solve(L_Sigma))
  List tau = state["tau"];            // Thresholds for discrete variables, list of q vectors, each K_j-1 elements long
  
  // Dimensions
  const int n = X.rows();
  const int p = X.cols();
  const int q = Z.cols();
  // Degrees of freedom for Wishart distribution
  const int df = n + a;
  
  // Counters for acceptance ratio
  int num_accepted_tau = 0, num_attempted_tau = 0;
  int num_accepted_Z = 0, num_attempted_Z = 0;
  
  // Working space: Matrix to select columns of X for block g (in step 2)
  Eigen::SparseMatrix<double> C(p,p);
  
  
  // One Gibbs iteration
  for (int iter = 0; iter < iters; iter++)
  {
    // Step 1a: Draw Z^(d)[Y.obs] | Y, Z^(c), beta, Omega ~ Truncated Multivariate Normal
    // Step 1b: Draw Z[Y.mis]|X,B,Sigma for each grade level
    for (int sub_iter = TMN_iters; sub_iter > 0; sub_iter--)
    {
      for (int j = 0; j < q; j++)
      {
        double *Yj_dbl = (K[j] == 0 ? as<NumericVector>(Y[j]).begin() : NULL);
        int *Yj_int = (K[j] == 0 ? NULL :
                         (Ybool[j] ? as<LogicalVector>(Y[j]).begin() : as<IntegerVector>(Y[j]).begin()));
        
        // Conditional mean/sd for Z_j|Z_-j
        // mu_j <- mu[,j] - E[,-j, drop=F] %*% (Omega[-j,j, drop=F] / Omega[j,j])
        // sd_j <- 1/sqrt(Omega[j,j])
        const VectorXd mu_j = mu.col(j) - (E.leftCols(j) * (Omega.topRows(j).col(j) / Omega(j,j)) +
                                           E.rightCols(q-j-1) * (Omega.bottomRows(q-j-1).col(j) / Omega(j,j)) );
        const double sd_j = 1./sqrt(Omega(j,j));
        
        // Discrete variable
        if (K[j] > 0)
        {
          const IntegerVector obsVec = as<IntegerVector>(Yobs[j])-1;                    // 0-based indexes
          const int *obs = obsVec.begin();
          const int Nobs = obsVec.size();
          
          // For more than two categories, propose new thresholds
          if (K[j] > 2)
          {
            const double *old = as<NumericVector>(tau[j]).begin();
            NumericVector proposalVec(K[j]-1);    // 0:(K-2)
            double *proposal = proposalVec.begin();
            JUMP;
            
            // First threshold always 0
            proposal[0] = 0;
            // Intermediate thresholds, k = 1:(K-3)
            for (int k=1; k < K[j]-2; k++)
              proposal[k] = moprobit::dist::rtruncnorm(proposal[k-1], old[k+1], old[k], sd_tau, ++cnt, key);
            // Last threshold, k = K-2
            proposal[K[j]-2] = moprobit::dist::rtruncnorm_uppertail(proposal[K[j]-3], old[K[j]-2], sd_tau, ++cnt, key);
            
            // Calculate the acceptance probability
            double R_thresh = 0, R_obs = 0;
            // prod((pnorm(old[3:K[j]] - old[2:(K[j]-1)], sd = sd_tau) -
            //       pnorm(proposal[1:(K[j]-2)] - old[2:(K[j]-1)], sd = sd_tau)) /
            //      (pnorm(proposal[3:K[j]] - proposal[2:(K[j]-1)], sd = sd_tau) -
            //       pnorm(old[1:(K[j]-2)] - proposal[2:(K[j]-1)], sd = sd_tau)), na.rm=T)
            for (int k = 1; k < K[j]-1; k++)  // 1:(K-2)
            {
              // Note that in 0-based indexing, old[K[j]-1] = proposal[K[j]-1] = Inf
              const double num = (k < K[j]-2 ? pnorm(old[k+1] - old[k], 0, sd_tau) : 1) -
                pnorm(proposal[k-1] - old[k], 0, sd_tau);
              const double denom = (k < K[j]-2 ? pnorm(proposal[k+1] - proposal[k], 0, sd_tau) : 1) -
                pnorm(old[k-1] - proposal[k], 0, sd_tau);
              const double r = num / denom;
              if (std::isfinite(r) && r > 0) R_thresh += log(r);
            }
            for (int i = 0; i < Nobs; i++)
            {
              const int I = obs[i];
              const int y = Yj_int[I] - 1;
              const double num =   (y < K[j]-1 ? pnorm(proposal[y], mu_j[I], sd_j) : 1) -
                (y > 0      ? pnorm(proposal[y-1], mu_j[I], sd_j) : 0);
              const double denom = (y < K[j]-1 ? pnorm(old[y], mu_j[I], sd_j) : 1) -
                (y > 0      ? pnorm(old[y-1], mu_j[I], sd_j) : 0);
              const double r = num / denom;
              if (std::isfinite(r) && r > 0) R_obs += log(r);
            }
            double R = exp(R_thresh + R_obs);
            
            // Determine acceptance
            //   Automatic acceptance if R >= 1
            num_attempted_tau += 1;
#ifdef PEDANTIC
            double u = moprobit::dist::runif(++cnt, key);
            if (R < 1 && u > R)
#else
              if (R < 1 && moprobit::dist::runif(++cnt, key) > R)
#endif
                goto step1b;  // Reject
              num_accepted_tau += 1;
              
              tau[j] = proposalVec;
          } // update thresholds
          
          // Update latent variable, Y.obs part (step 1a)
          const double *tau_j = as<NumericVector>(tau[j]).begin();
          JUMP;
          for (int i = 0; i < Nobs; i++)
          {
            COPY_I_TO_CNT;
            const int I = obs[i];
            const int y = (Ybool[j] ? Yj_int[I] : Yj_int[I]-1);
            // lower <- c(-Inf, tau[[j]])[Yj]
            // upper <- c(tau[[j]], Inf)[Yj]
            if (y == 0)
              // lower = -Inf, upper = tau[Yj]
              Z(I,j) = moprobit::dist::rtruncnorm_lowertail(tau_j[y], mu_j[I], sd_j, cnt, key);
            else if (y == K[j]-1)
              // lower = tau[Yj-1], upper = Inf
              Z(I,j) = moprobit::dist::rtruncnorm_uppertail(tau_j[y-1], mu_j[I], sd_j, cnt, key);
            else
              // lower = tau[Yj-1], upper = tau[Yj]
              Z(I,j) = moprobit::dist::rtruncnorm(tau_j[y-1], tau_j[y], mu_j[I], sd_j, cnt, key);
            
            E(I,j) = Z(I,j) - mu(I,j);
          }
          
        } // if discrete variable
        
        step1b:
          // Y.mis part (step 1b), only on last sub-iteration
          if (sub_iter == 1 && (as<IntegerVector>(Ymis[j]).size() > 0 || !(as<RObject>(meas_err[j]).isNULL())) )
          {
            const bool no_meas_err = as<RObject>(meas_err[j]).isNULL();
            // FIXME: Shouldn't Nmev == n ? Check and replace.
            const NumericVector mevVec = (no_meas_err ? NumericVector() : as<NumericVector>(meas_err[j]));
            const int Nmev = (no_meas_err ? 0 : mevVec.size() );
            const double *mev = (no_meas_err ? NULL : mevVec.begin());
            const IntegerVector misVec = (no_meas_err ?
                                            IntegerVector(as<IntegerVector>(Ymis[j])-1) :
                                            IntegerVector(Range(0, Nmev-1)) ); // 0-based indexes
            const int *mis = misVec.begin();
            const int Nmis = (no_meas_err ? misVec.size() : Nmev);
            // Original observed scores
            const double *obsScore = (no_meas_err ? NULL : as<NumericVector>(as<DataFrame>(env["Y"])).begin());
            
            // Adjust mu_j/sd_j for measurement error, as necessary
            VectorXd adj_mu_j;
            VectorXd adj_sd_j;
            if (no_meas_err)  // No measurement error
            {
              adj_sd_j = Eigen::VectorXd::Constant(n, sd_j);
              adj_mu_j = mu_j;
              
            } else {                                  // Variable j has measurement error
              // adj_sd_j <- 1/sqrt(1/sd_j^2 + mev)
              adj_sd_j = (as<Eigen::Map<Eigen::ArrayXd> >(mevVec) + 1./(sd_j*sd_j)).rsqrt().matrix();
              // adj_mu_j <- adj_sd_j^2 * (mu_j / sd_j^2  +  Y.obs.me[,j]*mev)
              adj_mu_j = adj_sd_j.array().square().matrix().cwiseProduct(mu_j / (sd_j*sd_j) +
                as<Eigen::Map<VectorXd> >(as<NumericVector>(as<List>(env["Y.mev"])[j])) );
            }
            
            if (as<RObject>(as<List>(env["update.formulas"])[j]).isNULL())
            {
              // This outcome is not a predictor for any other outcome, so we can accept the Gibbs proposal
              //   X does not need to be updated, because j does not appear in X
              if (K[j] == 0)
              {
                // Continuous case
                JUMP;
                for (int i = 0; i < Nmis; i++)
                {
                  COPY_I_TO_CNT;
                  const int I = mis[i];
                  Yj_dbl[I] = Z(I,j) = moprobit::dist::rnorm(adj_mu_j[I], adj_sd_j[I], cnt, key);
                  mu(I,j) = (X.row(I) * beta.col(j))(0,0);
                  E(I,j) = Z(I,j) - mu(I,j);
                }
              }
              else if (Ybool[j])
              {
                // Logical case
                JUMP;
                for (int i = 0; i < Nmis; i++)
                {
                  COPY_I_TO_CNT;
                  const int I = mis[i];
                  Z(I,j) = moprobit::dist::rnorm(mu_j[I], sd_j, cnt, key);
                  Yj_int[I] = (Z(I,j) > 0);
                  mu(I,j) = (X.row(I) * beta.col(j))(0,0);
                  E(I,j) = Z(I,j) - mu(I,j);
                }
              }
              else
              {
                // Factor case
                const double *tau_j = as<NumericVector>(tau[j]).begin();
                JUMP;
                for (int i = 0; i < Nmis; i++)
                {
                  COPY_I_TO_CNT;
                  const int I = mis[i];
                  Z(I,j) = moprobit::dist::rnorm(mu_j[I], sd_j, cnt, key);
                  // factor(sapply(Z[mis, j], function(z) levels(env$Y[,j])[sum(z > tau[[j]])+1]), levels=levels(env$Y[,j]))
                  // Yj[I] = sum(Z(I,j) > tau_j) + 1;
                  for (Yj_int[I] = 1; Yj_int[I] < K[j]; Yj_int[I]++)
                    if (Z(I,j) < tau_j[Yj_int[I]-1]) break;
                  mu(I,j) = (X.row(I) * beta.col(j))(0,0);
                  E(I,j) = Z(I,j) - mu(I,j);
                }
              }
              
            } else {
              // This outcome is also a predictor, so we have to vet the proposal with the downstream outcomes
              
              // Z.star <- Z[mis,]; Z.star[,j] <- rnorm(N.mis[j], mean = mu_j[mis], sd = sd_j)
              MatrixXd Zstar(Nmis, q);
              JUMP;
              for (int i = 0; i < Nmis; i++)
              {
                COPY_I_TO_CNT;
                const int I = mis[i];
                Zstar.row(i) = Z.row(I);
                Zstar(i,j) = moprobit::dist::rnorm(adj_mu_j[I], adj_sd_j[I], cnt, key);
              }
              
              // Y.star
              // FIXME: Would be more efficient to copy only those variables relevant for the given update.formula
              DataFrame Ystar = DataFrame::create();
              double *Ystar_j_dbl = NULL;
              int *Ystar_j_int = NULL;
              // Y.star: -j part
              for (int J = 0; J < q; J++)
                if (J != j)
                  switch (as<RObject>(Y[J]).sexp_type())
                  {
                  case LGLSXP:
                    Ystar.push_back(as<LogicalVector>(as<LogicalVector>(Y[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Y.names())[J]));
                    break;
                  case INTSXP:
                    Ystar.push_back(as<IntegerVector>(as<IntegerVector>(Y[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Y.names())[J]));
                    break;
                  case REALSXP:
                    Ystar.push_back(as<NumericVector>(as<NumericVector>(Y[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Y.names())[J]));
                    break;
                  default:
                    break;
                  }
                // Y.star: base predictor part
                for (int J = 0; J < Xbase.ncol(); J++)
                  switch (as<RObject>(Xbase[J]).sexp_type())
                  {
                  case LGLSXP:
                    Ystar.push_back(as<LogicalVector>(as<LogicalVector>(Xbase[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Xbase.names())[J]));
                    break;
                  case INTSXP:
                    Ystar.push_back(as<IntegerVector>(as<IntegerVector>(Xbase[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Xbase.names())[J]));
                    break;
                  case REALSXP:
                    Ystar.push_back(as<NumericVector>(as<NumericVector>(Xbase[J])[misVec]),
                                    as<std::string>(as<CharacterVector>(Xbase.names())[J]));
                    break;
                  default:
                    break;
                  }
                // Y.star: proposed outcome j
                //   Note: This block is not thread-safe, but we're just copying memory, so probably won't parallelize
                if (K[j] == 0)
                {
                  NumericVector y(Nmis);
                  for (int i = 0; i < Nmis; i++)
                    y[i] = Zstar(i,j);
                  Ystar.push_back(y, as<std::string>(as<CharacterVector>(Y.names())[j]));
                  Ystar_j_dbl = as<NumericVector>(Ystar[Ystar.ncol()-1]).begin();
                }
                else if (Ybool[j])
                {
                  LogicalVector y(Nmis);
                  for (int i = 0; i < Nmis; i++)
                    y[i] = (Zstar(i,j) > 0);
                  Ystar.push_back(y, as<std::string>(as<CharacterVector>(Y.names())[j]));
                  Ystar_j_int = as<LogicalVector>(Ystar[Ystar.ncol()-1]).begin();
                }
                else // factor
                {
                  IntegerVector y(Nmis);
                  const NumericVector tau_j = tau[j];
                  for (int i = 0; i < Nmis; i++)
                    y[i] = sum(Zstar(i,j) > tau_j) + 1;
                  y.attr("class") = as<IntegerVector>(Y[j]).attr("class");
                  y.attr("levels") = as<IntegerVector>(Y[j]).attr("levels");
                  Ystar.push_back(y, as<std::string>(as<CharacterVector>(Y.names())[j]));
                  Ystar_j_int = as<IntegerVector>(Ystar[Ystar.ncol()-1]).begin();
                }
                
                // X.star: model.matrix(env$update.formulas[[j]], Y.star, env$update.constrasts[[j]])
                const IntegerVector update_to_vec = as<List>(env["update.to"])[j];
                const int *update_to = update_to_vec.begin();
                const int *update_from = as<IntegerVector>(as<List>(env["update.from"])[j]).begin();
                const int Nupdate = update_to_vec.size();
                // FIXME: Could we avoid passing through the R interface, here?
                const NumericMatrix mm = modelmatrix(as<List>(env["update.formulas"])[j],
                                                     Ystar,
                                                     as<List>(env["update.contrasts"])[j]);
                MatrixXd Xstar(Nmis, p);
                for (int i = 0; i < Nmis; i++)
                {
                  Xstar.row(i) = X.row(mis[i]);
                  for (int J = 0; J < Nupdate; J++)
                    Xstar(i, update_to[J]-1) = mm(i, update_from[J]-1);
                }
                
                // E.star
                const IntegerVector update_cascade_vec = as<List>(env["update.cascade"])[j];
                const int *update_cascade = update_cascade_vec.begin();
                const int Ncascade = update_cascade_vec.size();
                const MatrixXd Estar = Zstar - Xstar * beta;
                
                // For each individual, calculate acceptance and update
                num_attempted_Z += Nmis;
                JUMP;
                for (int i = 0; i < Nmis; i++)
                {
                  COPY_I_TO_CNT;
                  const int I = mis[i];

                  // Calculate acceptance ratio
                  //  R <- dmvnorm(as.matrix(E.star), sigma = as.matrix(Sigma), log=T) -
                  //       dmvnorm(as.matrix(E[mis,]), sigma = as.matrix(Sigma), log=T) +
                  //       dnorm(Z[mis,j], mean = mu_j[mis], sd = sd_j, log=T) -
                  //       dnorm(Z.star[,j], mean = mu_j[mis], sd = sd_j, log=T)
                  const double R = -.5 *
                    ( (Estar.row(i) * Omega * Estar.row(i).transpose())(0,0) -
                      (E.row(I) * Omega * E.row(I).transpose())(0,0) +
                      // If the original observed score is missing, we have set mev to 0 in init().
                      ((no_meas_err || mev[I] == 0) ? 0 : 
                       ( (obsScore[I] - Zstar(i,j)) * (obsScore[I] - Zstar(i,j)) -
                         (obsScore[I] - Z(I,j)) * (obsScore[I] - Z(I,j)) ) * mev[I] ) +
                      (Z(I,j) - adj_mu_j[I]) * (Z(I,j) - adj_mu_j[I]) / (adj_sd_j[I]*adj_sd_j[I]) -
                      (Zstar(i,j) - adj_mu_j[I]) * (Zstar(i,j) - adj_mu_j[I]) / (adj_sd_j[I]*adj_sd_j[I]) );
                  
                  // Accept
#ifdef PEDANTIC
                  double u = moprobit::dist::runif(cnt, key);
                  if (R > 0 || log(u) < R)
#else
                  if (R > 0 || log(moprobit::dist::runif(cnt, key)) < R)
#endif
                  {
                    Z(I,j) = Zstar(i,j);
                    E(I,j) = Estar(i,j);
                    mu(I,j) = Z(I,j) - E(I,j);
                    
                    // Y
                    if (K[j] == 0)
                      Yj_dbl[I] = Ystar_j_dbl[i];
//                    else if (Ybool[j])
//                      Yj_int[I] = Ystar_j_int[i];
                    else
                      Yj_int[I] = Ystar_j_int[i];
                    
                    // X
                    for (int J = 0; J < Nupdate; J++)
                      X(I, update_to[J]-1) = Xstar(i, update_to[J]-1);
                    
                    // Downstream outcomes
                    for (int J = 0; J < Ncascade; J++)
                    {
                      const int JJ = update_cascade[J]-1;
                      E(I,JJ) = Estar(i,JJ);
                      mu(I,JJ) = Z(I,JJ) - E(I,JJ);
                    }
                    
                    num_accepted_Z += 1;
                  } // accept this individual's proposal
                } // each individual with missing j
            } // sample outcome that is also a predictor
            
          } // step 1b (missing outcomes)
      } // each outcome variable
    } // each sub_iter
    
    
    // Step 2: Draw beta | Z, Sigma ~ MN_{p,q}( (X'X)^{-1}X'Z, (X'X)^{-1}, Sigma) for each block
    //   q.block is in order by block, but p.block could be in any arbitrary order
    JUMP;
    int q_lt_g = 0;
    for (int g = 0; g < G; g++)
    {
      const int q_g = sum(q_block.column(g));  // number of outcomes in this block
      
      // Compute matrix C to select columns of X applicable to block g
      int p_g = 0;     // Number of covariates for block g
      C.setZero();
      for (int j = 0; j < p; j++)
        if (p_block(j, g))
          C.coeffRef(j, p_g++) = 1;
        
        // Cholesky decomposition of: Xtilde' Xtilde = (XC)'(XC)
        const Eigen::LLT<MatrixXd> XX_llt(
            MatrixXd(p_g, p_g).setZero().selfadjointView<Lower>().rankUpdate( (X*C.leftCols(p_g)).adjoint() ) );
        
        // beta_g ~ rMatNorm( X.X %*% crossprod(Xtilde, Ztilde), t(chol(X.X)), t(L_Sigma[q.g, q.g]))
        //    Recall that Sigma_{g|<g} = L_gg L_gg'
        const MatrixXd beta_g =
          // (Xtilde'Xtilde)^{-1} Xtilde'Ztilde + L'^{-1} [std norm deviates] LSigma_qg'
          XX_llt.matrixU().solve(
              // L^{-1} Xtilde'Ztilde
              XX_llt.matrixL().solve(
                  // Xtilde'
                  (X*C.leftCols(p_g)).adjoint() *
                    // Ztilde = Z[,q.g] - E[, q_lt_g] %*% crossprod(D_inv_Sigma[q_lt_g, q_lt_g]) %*%
                    //                    L_Sigma[q_lt_g, q_lt_g] %*% t(L_Sigma[q.g, q_lt_g])
                    (Z.middleCols(q_lt_g, q_g) -
                    E.leftCols(q_lt_g) *
                    D_inv_Sigma.topLeftCorner(q_lt_g, q_lt_g).triangularView<Lower>().transpose() *
                    D_inv_Sigma.topLeftCorner(q_lt_g, q_lt_g).triangularView<Lower>() *
                    L_Sigma.topLeftCorner(q_lt_g, q_lt_g).triangularView<Lower>() *
                    L_Sigma.block(q_lt_g,0, q_g,q_lt_g).transpose() )
              ) +
                // [std norm deviates] %*% t(L_Sigma[q.g, q.g])
                rnorm_matrix(p_g, q_g, cnt, key) *
                L_Sigma.block(q_lt_g,q_lt_g, q_g,q_g).triangularView<Lower>().transpose() );
        
        // Copy beta_g into sparse beta[p.g, q.g]
        // FIXME: Any way to make this more efficient?
        for (int i = p_g = 0; i < p; i++)
          if (p_block(i, g))
          {
            for (int j = 0; j < q_g; j++)
              beta.coeffRef(i, q_lt_g+j) = beta_g(p_g, j);
            p_g++;
          }
          
          // mu[,q.g] <- Xtilde %*% beta[p.g, q.g]
          mu.middleCols(q_lt_g, q_g) = X * beta.middleCols(q_lt_g, q_g);
          
          // E[,q.g] <- Z[,q.g] - mu[,q.g]
          E.middleCols(q_lt_g, q_g) = Z.middleCols(q_lt_g, q_g) - mu.middleCols(q_lt_g, q_g);
          
          // cumulative number of outcome variables in prior blocks
          q_lt_g += q_g;
    } // each block in beta
    
    
    // Step 3: Draw Sigma | E ~ CIW(E'E + a*I_q, n+a)
    if (!fixSigma)
    {
      JUMP;
      if (!fixCrossBlockCov)
      {
        // E'E + a*diag(q)
        const MatrixXd resid2( MatrixXd(MatrixXd(q, q).setZero().selfadjointView<Lower>().rankUpdate(E.adjoint())) +
                               a * MatrixXd::Identity(q,q) );
        // Lambda <- rCIW(df, resid2, fixed)
        // FIXME: Figure out templating for internal_CIW_eigen so we don't have to use these temporary objects
        MatrixXd L_Sigma_tmp(q,q);
        MatrixXd D_inv_Sigma_tmp(q,q);
        internal_rCIW_eigen(df, resid2, fixed, eps, L_Sigma_tmp, D_inv_Sigma_tmp, ++cnt, key);
        L_Sigma = L_Sigma_tmp;
        D_inv_Sigma = D_inv_Sigma_tmp;
      }
      else // block-wise
      {
        int q_lt_g = 0;
        for (int g = 0; g < G; g++)
        {
          const int q_g = sum(q_block.column(g));  // number of outcomes in this block
          // crossprod(E[, q.g, drop=F]) + a * diag(sum(q.g))
          const MatrixXd resid2( MatrixXd(MatrixXd(q_g,q_g).setZero().selfadjointView<Lower>().rankUpdate(E.middleCols(q_lt_g,q_g).adjoint())) +
                                 a * MatrixXd::Identity(q_g,q_g) );
          //auto L_Sigma_g = L_Sigma.block(q_lt_g,q_lt_g, q_g,q_g);
          //auto D_inv_Sigma_g = D_inv_Sigma.block(q_lt_g,q_lt_g, q_g,q_g);
          
          // Lambda <- rCIW(df, resid2, fixed[q.g])
          // FIXME: Figure out templating for internal_CIW_eigen so we don't have to use these temporary objects
          MatrixXd L_Sigma_tmp(q_g,q_g);
          MatrixXd D_inv_Sigma_tmp(q_g,q_g);
          internal_rCIW_eigen(df, resid2, fixed[q_block.column(g)], eps, L_Sigma_tmp, D_inv_Sigma_tmp, ++cnt, key);
          L_Sigma.block(q_lt_g,q_lt_g, q_g,q_g) = L_Sigma_tmp;
          D_inv_Sigma.block(q_lt_g,q_lt_g, q_g,q_g) = D_inv_Sigma_tmp;
          
          // cumulative number of outcome variables in prior blocks
          q_lt_g += q_g;
        }
      }
      
      // Compute the corresponding covariance matrix
      Sigma = L_Sigma * L_Sigma.transpose();
      Omega = D_inv_Sigma.transpose() * D_inv_Sigma;
    }

  } // Gibbs iteration
  
  
  // Stash beta
  beta.makeCompressed();
  state["beta"] = wrap(beta);
  as<S4>(state["beta"]).attr("Dimnames") = as<S4>(prior["beta"]).attr("Dimnames");
  
  if (num_attempted_tau > 0) state["acceptance_rate_tau"] = (double)num_accepted_tau / (double)num_attempted_tau;
  if (num_attempted_Z > 0) state["acceptance_rate_Z"] = (double)num_accepted_Z / (double)num_attempted_Z;
  
  return state;
}
