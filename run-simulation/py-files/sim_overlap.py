#!/usr/bin/env python
from causalml.match import NearestNeighborMatch
from causalml.inference.meta import BaseXLearner
from causalml.inference.meta import BaseTLearner
from causalml.inference.meta import BaseRLearner
from causalml.inference.meta import TMLELearner
import statsmodels.formula.api as smf

import os
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression, 
    LogisticRegression
)
from lightgbm import LGBMClassifier, LGBMRegressor
import doubleml as dml
import utils_ate
import utils_dgps
from utils_ate import (
    calibrate_propensity_score, 
    compute_ipw_wls,
    calibration_errors,
    compute_ci_metrics,
    create_results_dict
)
from utils_dgps import dgp_wrapper

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Retrieve the SLURM_JOB_ID environment variable
job_id = os.getenv('SLURM_JOB_ID', 'unknown')

n_rep = 100  # Total number of repetitions
reps_per_process = n_rep // size
extra_reps = n_rep % size

if rank < extra_reps:
    reps_per_process += 1
    start_rep = rank * reps_per_process
else:
    start_rep = rank * reps_per_process + extra_reps

end_rep = start_rep + reps_per_process

start_time = time.time()
warnings.filterwarnings('ignore')

# DGP pars
dgp_type = 'sim_overlap'
n_obs_list = [500,1000,2000,3000,4000]
dim_x = 3
theta = -1

overlaps = [0.1, 0.5, 0.9]
n_folds = 5
score = "ATE"
clipping_thresholds = [1e-12, 0.01, 0.1]

calib_methods = [
    ('alg-1-uncalibrated', 'uncalibrated'),
    #('alg-3-cross-fitted-calib', 'isotonic'),
    #('alg-3-cross-fitted-calib', 'platt'),
    #('alg-3-cross-fitted-calib', 'ivap'),
    ('alg-5-full-sample-calib', 'isotonic'),
    ('alg-5-full-sample-calib', 'platt'),
    ('alg-5-full-sample-calib', 'ivap'),
    #('oracle', 'uncalibrated'),
]

n_folds = 5
score = "ATE"

learner_dict_g = {
    'Linear': LinearRegression(),
    'RF': RandomForestRegressor(),
    'LGBM': LGBMRegressor(verbose=-1)
    }

learner_dict_m = {
    'Logit': LogisticRegression(),
    'RF': RandomForestClassifier(),
    'LGBM': LGBMClassifier(verbose=-1)
    }

def estimate_ate(data, theta, learner_g, learner_m, n_folds, score, clipping_thresholds, calib_methods, m_0):

    n_calib_methods = len(calib_methods)
    n_clipping_thresholds = len(clipping_thresholds)

    # Base array with shape (n_calib_methods, n_clipping_thresholds)
    irm_coefs = np.full((n_calib_methods, n_clipping_thresholds), np.nan)

    # Create multiple arrays of the same shape and fill value (np.nan)
    irm_ses, irm_cover, irm_ci_length, K, rmses, ece_u, ece_q, ece_u_5, ece_q_5, mce, ece_l2, ipw_coefs, ipw_ses, ipw_ci_length,ipw_cover, plr_coefs, plr_ses,plr_ci_length,plr_cover = [
        np.full_like(irm_coefs, np.nan) for _ in range(19)
    ]

    # Arrays with the string "not specified" and object dtype
    name_calib_method, name_method = [
        np.full_like(irm_coefs, "not specified", dtype=object) for _ in range(2)
    ]

    # Create arrays for match_* variables with shape (n_clipping_thresholds,)
    match_coefs = np.full((n_calib_methods, n_clipping_thresholds), np.nan)
    match_ses,match_ci_length, match_cover, K_match, rmses_match, ece_u_match, ece_q_match, ece_u_5_match, ece_q_5_match, mce_match, ece_l2_match = [
        np.full_like(match_coefs, np.nan) for _ in range(11)
    ]

    # Create arrays for meta learners variables with shape (n_clipping_thresholds,)
    X_coefs = np.full((n_calib_methods, n_clipping_thresholds), np.nan)
    X_ci_length, X_cover, TMLE_coefs, TMLE_ci_length, TMLE_cover,  = [
        np.full_like(X_coefs, np.nan) for _ in range(5)
    ]

    # set up the DoubleMLIRM models
    dml_irm = dml.DoubleMLIRM(data,
                            ml_g=learner_g,
                            ml_m=learner_m,
                            score=score,
                            n_folds=n_folds,
                            trimming_threshold=1e-12) 
    # fit standard model without calibration and save predictions
    dml_irm.fit(n_jobs_cv=5)
    smpls = dml_irm.smpls[0]
    dml_irm_ext = dml.DoubleMLIRM(data,
                                ml_g=dml.utils.DMLDummyRegressor(),
                                ml_m=dml.utils.DMLDummyClassifier(),
                                score=score,
                                n_folds=n_folds,
                                trimming_threshold=1e-12)
    # set up the PLR model
    dml_plr = dml.DoubleMLPLR(data,
                              ml_l=learner_g,
                              ml_m=learner_m,
                              score="partialling out",
                              n_folds=n_folds)
    dml_plr.fit(n_jobs_cv=5)
    dml_plr_ext = dml.DoubleMLPLR(data,
                                  ml_l=dml.utils.DMLDummyRegressor(),
                                  ml_m=dml.utils.DMLDummyClassifier(),
                                  score="partialling out",
                                  n_folds=n_folds)

    prop_score = dml_irm.predictions["ml_m"][:, :, 0].squeeze()
    treatment = data.d
    outcome = data.y
    covariates = data.x

    for i_clipping_threshold, clipping_threshold in enumerate(clipping_thresholds):                 

        # re-fit model with calibration
        for i_calib_method, (method, calib_method) in enumerate(calib_methods):
            method = method
            calib_method = calib_method
            calib_prop_score = calibrate_propensity_score(
                method=method,
                covariates=covariates,
                propensity_score=prop_score, 
                treatment=treatment, 
                learner_m=learner_m,
                calib_method=calib_method,
                clipping_threshold=clipping_threshold,
                smpls=smpls,
                true_propensity_score=m_0)
            
            df_X = pd.DataFrame(covariates, columns=[f'x_{i}' for i in range(covariates.shape[1])]) 

            df_pred = pd.DataFrame({
            'd': treatment,
            'y': outcome,
            'prop_score': np.clip(calib_prop_score, 
                                0.0 + clipping_threshold, 
                                1.0 - clipping_threshold)
            })
            df_pred = pd.concat([df_pred, df_X], axis=1)
            
            # Perform nearest neighbor matching using causalml
            psm = NearestNeighborMatch(
                caliper=0.1,
                replace=True,
                ratio=1,
                random_state=42,
                n_jobs = -1
            )
            
            # Perform matching using external propensity scores
            matched_data = psm.match(
                data=df_pred,
                treatment_col='d',
                score_cols=['prop_score'],
            )
            
            # Estimate ATE with OLS
            covariates_list = [str(col) for col in df_X.columns]  # Assuming df_X is your covariate DataFrame
            model_formula = f'y ~ d + {" + ".join(covariates_list)}'
            model = smf.ols(model_formula, data=matched_data).fit()      
            # Store results
            match_coefs[i_calib_method,i_clipping_threshold] = model.params['d']
            match_ses[i_calib_method,i_clipping_threshold] = model.bse['d']

            confint = model.conf_int(alpha=0.05)

            # Calculate the length of the confidence interval for 'd'
            match_ci_length[i_calib_method,i_clipping_threshold] = confint.loc['d', 1] - confint.loc['d', 0]

            # Check coverage: See if the true treatment effect theta is within the confidence interval for 'd'
            match_cover[i_calib_method,i_clipping_threshold] = (confint.loc['d', 0] < theta) and (theta < confint.loc['d', 1])
        
            # Store propensity scores for calibration metrics
            prop_score_match = matched_data['prop_score'].values
            y_match = matched_data['y'].values
            d_match = matched_data['d'].values
            X_match = matched_data[df_X.columns].values

            ece_u_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match, 
                                                                                        strategy = 'uniform',norm='l1',n_bins= float(10))
            ece_q_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match,
                                                                                        strategy = 'quantile',norm='l1',n_bins= float(10))
            ece_u_5_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match, 
                                                                                        strategy = 'uniform',norm='l1',n_bins= float(5))
            ece_q_5_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match,
                                                                                        strategy = 'quantile',norm='l1',n_bins= float(5))            
            ece_l2_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match,
                                                                            strategy = 'uniform',norm='l2',n_bins= float(10))
            mce_match[i_calib_method,i_clipping_threshold] = calibration_errors(prop_score_match, d_match,
                                                                                    strategy = 'uniform',norm='max',n_bins= float(10))
            # store n-unique propensity scores
            prop_values, inverse_map, counts = np.unique(prop_score_match, return_inverse=True, return_counts=True)
            K_match[i_calib_method,i_clipping_threshold] = len(prop_values)
            rmses_match[i_calib_method,i_clipping_threshold] = np.sqrt(((prop_score_match -  d_match) ** 2).mean())


            # Add X Learner, TMLE Learner

            #[X_coefs[i_calib_method,i_clipping_threshold], X_lower, X_upper] = BaseXLearner(learner=learner_g).estimate_ate(X=df_X, treatment=df_pred['d'], 
            #                                                                                                y=df_pred['y'],p=df_pred['prop_score'],
            #                                                                                                bootstrap_ci=True, n_bootstraps =100)
            #X_ci_length[i_calib_method,i_clipping_threshold], X_cover[i_calib_method,i_clipping_threshold] = compute_ci_metrics(theta, X_lower, X_upper).values()

            [TMLE_coefs[i_calib_method,i_clipping_threshold], TMLE_lower, TMLE_upper] = TMLELearner(learner=learner_g,calibrate_propensity=False).estimate_ate(X=df_X, treatment=df_pred['d'], 
                                                                                                                    y=df_pred['y'],p=df_pred['prop_score'],
                                                                                                                    return_ci =True)
            TMLE_ci_length[i_calib_method,i_clipping_threshold], TMLE_cover[i_calib_method,i_clipping_threshold] = compute_ci_metrics(theta, TMLE_lower, TMLE_upper).values() 

            #[T_coefs[i_calib_method,i_clipping_threshold], T_lower, T_upper] = BaseTLearner(learner=learner_g).estimate_ate(X=df_X, treatment=df_pred['d'],
            #                                                                                                                y=df_pred['y'],p=df_pred['prop_score'],bootstrap_ci =True)
            #T_ci_length[i_calib_method,i_clipping_threshold], T_cover[i_calib_method,i_clipping_threshold] = compute_ci_metrics(theta, T_lower, T_upper).values() 
            #[R_coefs[i_calib_method,i_clipping_threshold], R_lower, R_upper] = BaseRLearner(learner=learner_g).estimate_ate(X=df_X, treatment=df_pred['d'],
            #                                                                                                                y=df_pred['y'],p=df_pred['prop_score'],bootstrap_ci =True)  
            #R_ci_length[i_calib_method,i_clipping_threshold], R_cover[i_calib_method,i_clipping_threshold] = compute_ci_metrics(theta, R_lower, R_upper).values() 

            # fit irm model with external predictions
            pred_dict_irm_calib = {"d": {
                "ml_g0": dml_irm.predictions["ml_g0"][:, :, 0],
                "ml_g1": dml_irm.predictions["ml_g1"][:, :, 0],
                "ml_m": calib_prop_score.reshape(-1, 1),
                }
            }
            dml_irm_ext.fit(external_predictions=pred_dict_irm_calib)
            
            # fit plr model with external predictions
            pred_dict_plr_calib = {"d": {
                "ml_l": dml_plr.predictions["ml_l"][:, :, 0],
                "ml_m": calib_prop_score.reshape(-1, 1),
                }
            }
            dml_plr_ext.fit(external_predictions=pred_dict_plr_calib)

            irm_coefs[i_calib_method, i_clipping_threshold] = dml_irm_ext.coef[0]
            irm_ses[i_calib_method, i_clipping_threshold] = dml_irm_ext.se[0]
            irm_confint_calib = dml_irm_ext.confint()
            irm_cover[i_calib_method, i_clipping_threshold] = (irm_confint_calib.loc['d', '2.5 %'] < theta) & (theta < irm_confint_calib.loc['d', '97.5 %'])
            irm_ci_length[i_calib_method, i_clipping_threshold]  = irm_confint_calib.loc['d', '97.5 %'] - irm_confint_calib.loc['d', '2.5 %']
            rmses[i_calib_method, i_clipping_threshold] = np.sqrt(((calib_prop_score - treatment) ** 2).mean())
            name_method[i_calib_method, i_clipping_threshold] = method
            name_calib_method[i_calib_method, i_clipping_threshold] = calib_method
            ece_u[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment, 
                                                                                     strategy = 'uniform',norm='l1',n_bins= float(10))
            ece_q[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment,
                                                                                     strategy = 'quantile',norm='l1',n_bins= float(10))
            ece_u_5[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment, 
                                                                                     strategy = 'uniform',norm='l1',n_bins= float(5))
            ece_q_5[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment,
                                                                                     strategy = 'quantile',norm='l1',n_bins= float(5))            
            ece_l2[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment,
                                                                                      strategy = 'uniform',norm='l2',n_bins= float(10))
            mce[i_calib_method, i_clipping_threshold] = calibration_errors(calib_prop_score, treatment,
                                                                                   strategy = 'uniform',norm='max',n_bins= float(10))
            # store n-unique propensity scores
            prop_values, inverse_map, counts = np.unique(calib_prop_score, return_inverse=True, return_counts=True)
            K[i_calib_method, i_clipping_threshold] = len(prop_values)
            ipw = compute_ipw_wls(calib_prop_score, treatment, outcome,theta)

            ipw_coefs[i_calib_method, i_clipping_threshold] = ipw['IPW_coefs']
            ipw_ses[i_calib_method, i_clipping_threshold] = ipw['IPW_ses']
            ipw_ci_length[i_calib_method, i_clipping_threshold] = ipw['IPW_ci_length']
            ipw_cover[i_calib_method, i_clipping_threshold] = ipw['IPW_cover']
            plr_coefs[i_calib_method, i_clipping_threshold] = dml_plr_ext.coef[0]
            plr_ses[i_calib_method, i_clipping_threshold] = dml_plr_ext.se[0]
            plr_confint_calib = dml_plr_ext.confint()
            plr_cover[i_calib_method, i_clipping_threshold] = (plr_confint_calib.loc['d', '2.5 %'] < theta) & (theta < plr_confint_calib.loc['d', '97.5 %'])
            plr_ci_length[i_calib_method, i_clipping_threshold]  = plr_confint_calib.loc['d', '97.5 %'] - plr_confint_calib.loc['d', '2.5 %']

    # Define common metrics structure
    METRICS = [
        "irm_coefs", "irm_ses", "irm_cover", "irm_ci_length",
        "K", "rmses", "method", "calib_method",
        "ipw_coefs", "ipw_ses", "ipw_cover", "ipw_ci_length",
        "plr_coefs", "plr_ses", "plr_cover", "plr_ci_length",
        "ece_u", "ece_q", "ece_u_5", "ece_q_5", "ece_l2", "mce",
        "match_coefs", "match_ses", "match_ci_length", "match_cover",
        "X_coefs", "X_ci_length", "X_cover", 
        "TMLE_coefs", "TMLE_ci_length", "TMLE_cover"
    ]
    
    n_calib_methods = len(calib_methods)
    n_clipping_thresholds = len(clipping_thresholds)
    base_shape = (n_calib_methods, n_clipping_thresholds)
    results_dict_calib = create_results_dict(
        base_shape=(n_calib_methods, n_clipping_thresholds),
        fill_data={
            "irm_coefs": irm_coefs,
            "irm_ses": irm_ses,
            "irm_cover": irm_cover,
            "irm_ci_length": irm_ci_length,
            "K": K,
            "rmses": rmses,
            "method": name_method,
            "calib_method": name_calib_method,
            "ipw_coefs": ipw_coefs,
            "ipw_ses": ipw_ses,
            "ipw_cover": ipw_cover,
            "ipw_ci_length": ipw_ci_length,
            "plr_coefs": plr_coefs,
            "plr_ses": plr_ses,
            "plr_cover": plr_cover,
            "plr_ci_length": plr_ci_length,
            "ece_u": ece_u,
            "ece_q": ece_q,
            "ece_u_5": ece_u_5,
            "ece_q_5": ece_q_5,        
            "ece_l2": ece_l2,        
            "mce": mce
        },
        metrics_dict=METRICS,
    )

    results_dict_match = create_results_dict(
        base_shape=(n_calib_methods, n_clipping_thresholds),
        fill_data={
            "K": K_match,  
            "rmses": rmses_match,
            "method": name_method,
            "calib_method": name_calib_method,
            "ece_u": ece_u_match,
            "ece_q": ece_q_match,
            "ece_u_5": ece_u_5_match,
            "ece_q_5": ece_q_5_match,
            "ece_l2": ece_l2_match,
            "mce": mce_match,
            "match_coefs": match_coefs,
            "match_ses": match_ses,
            "match_ci_length": match_ci_length,
            "match_cover": match_cover
        },
        metrics_dict=METRICS
    )

    results_dict_meta = create_results_dict(
        base_shape=(n_calib_methods, n_clipping_thresholds),
        fill_data={
            "method": name_method,
            "calib_method": name_calib_method,
            "X_coefs": X_coefs,
            "X_ci_length": X_ci_length,
            "X_cover": X_cover,
            "TMLE_coefs": TMLE_coefs,
            "TMLE_ci_length": TMLE_ci_length,
            "TMLE_cover": TMLE_cover
        },
        metrics_dict=METRICS
    )

    return results_dict_calib, results_dict_match, results_dict_meta


df = pd.DataFrame(columns=[
    "irm_coefs", "irm_ses", "irm_cover", "irm_ci_length", "K", "rmses", "method", "calib_method",
    "ipw_coefs","ipw_ses", "ipw_cover", "ipw_ci_length", "plr_coefs","plr_ses", "plr_cover", "plr_ci_length",
    "match_coefs","match_ses", "match_ci_length","match_cover", 
    "X_coefs", "X_ci_length", "X_cover", "TMLE_coefs", "TMLE_ci_length", "TMLE_cover",
    "learner_g", "learner_m", "n_obs", "dim_x", "overlap", "clipping_threshold", "repetition",
    "ece_u", "ece_q", "ece_u_5", "ece_q_5", "ece_l2", "mce",
])

for i_rep in range(start_rep, end_rep):
    np.random.seed(42 + i_rep)
    for (i_overlap, overlap) in enumerate(overlaps):
        for (i_n_obs, n_obs) in enumerate(n_obs_list):
            print(f'Rank {rank} handling repetition: {i_rep + 1} / {n_rep}\tn_obs: {n_obs}\toverlap: {overlap}')
            n_obs = int(n_obs)
            i_rep = int(i_rep)            
            # generate data
            dgp_dict = {
                'overlap': overlap,
                'n_obs': n_obs,
            }
            data_dict = dgp_wrapper(dgp_type=dgp_type, **dgp_dict)
            # true treatment     
            theta = data_dict['treatment_effect']
            # true propensity score      
            m_0 = data_dict['propensity_score']
            df_data = pd.DataFrame(data_dict['covariates'], columns=[f'x{i+1}' for i in range(dim_x)])
            df_data['y'] = data_dict['outcome']
            df_data['d'] = data_dict['treatment']
            dml_data = dml.DoubleMLData(df_data, y_col='y', d_cols='d')
            for i_learner_g, (name_learner_g, learner_g) in enumerate(learner_dict_g.items()):
                for i_learner_m, (name_learner_m, learner_m) in enumerate(learner_dict_m.items()):
                    random_state_value = 42 + i_rep
                    if hasattr(learner_m, 'random_state'):
                        learner_m.random_state = random_state_value
                    if hasattr(learner_g, 'random_state'):
                        learner_g.random_state = random_state_value

                    results_dict_calib, results_dict_match, results_dict_meta = estimate_ate(
                        dml_data, theta, learner_g, learner_m, n_folds, score, clipping_thresholds, calib_methods, m_0
                    )

                    result_columns = [
                        "irm_coefs", "irm_ses", "irm_cover", "irm_ci_length", "K", "rmses",
                        "method", "calib_method", "ipw_coefs", "ipw_ses", "ipw_cover", "ipw_ci_length",
                        "plr_coefs", "plr_ses", "plr_cover", "plr_ci_length",
                        "match_coefs", "match_ses", "match_ci_length", "match_cover",
                        "X_coefs", "X_ci_length", "X_cover", "TMLE_coefs", "TMLE_ci_length", "TMLE_cover",
                        "ece_u", "ece_q", "ece_u_5", "ece_q_5", "ece_l2", "mce"
                    ]

                    common_metadata = {
                        "learner_g": name_learner_g,
                        "learner_m": name_learner_m,
                        "n_obs": n_obs,
                        "dim_x": dim_x,
                        "overlap": overlap,
                        "repetition": i_rep         
                    }
                    n_calib_methods = len(calib_methods)
                    n_clipping_thresholds = len(clipping_thresholds)
                    
                    for i_clipping, clipping_threshold in enumerate(clipping_thresholds):
                        for results in [results_dict_calib, results_dict_match, results_dict_meta]:
                            # Process calibration results
                            result_data = {
                                col: results[col][:, i_clipping]  # 1D slice
                                for col in result_columns
                            }
                            result_data.update({
                                k: np.full(n_calib_methods, v) 
                                for k, v in common_metadata.items()
                            })
                            result_data["clipping_threshold"] = clipping_threshold
                            df = pd.concat([df, pd.DataFrame(result_data)], ignore_index=True)


output_file = f'02_results_overlap/ranks_{job_id}/output_rank_{rank:03d}.pkl'
df.to_pickle(output_file)

end_time = time.time()

if rank == 0:
    total_runtime = end_time - start_time
    print(f"Total Runtime: {total_runtime:.2f} seconds")


