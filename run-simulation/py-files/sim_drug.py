#!/usr/bin/env python
import os
import datetime
import time
import warnings
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
import utils_calibration
import utils_dgps
from utils_calibration import (
    calibrate_propensity_score, 
    compute_ipw_estimate, 
    calibration_errors
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
dgp_type = 'sim_v06_drug'
n_obs_list = [200,500,1000,2000,4000]
dim_x = 3
theta = -1

clipping_thresholds = [1e-12, 0.01, 0.1]
overlaps = [0.1, 0.5, 0.9]
n_folds = 5
score = "ATE"
clipping_thresholds = [1e-12, 0.01, 0.1]

calib_methods = [
    ('alg-1-uncalibrated', 'uncalibrated'),
    ('alg-2-nested-cross-fitting-calib', 'isotonic'),
    ('alg-2-nested-cross-fitting-calib', 'platt'),
    ('alg-2-nested-cross-fitting-calib', 'ivap'),
    ('alg-3-cross-fitted-calib', 'isotonic'),
    ('alg-3-cross-fitted-calib', 'platt'),
    ('alg-3-cross-fitted-calib', 'ivap'),
    ('alg-4-single-split-calib', 'isotonic'),
    ('alg-4-single-split-calib', 'platt'),
    ('alg-4-single-split-calib', 'ivap'),
    ('alg-5-full-sample-calib', 'isotonic'),
    ('alg-5-full-sample-calib', 'platt'),
    ('alg-5-full-sample-calib', 'ivap'),
    ('oracle', 'uncalibrated'),
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


def estimate_irm(data, theta, learner_g, learner_m, n_folds, score, clipping_thresholds, calib_methods, m_0):
    n_calib_methods = len(calib_methods)
    n_clipping_thresholds = len(clipping_thresholds)
    coefs = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    bias = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    ses = np.full_like(coefs, fill_value=np.nan)
    cover = np.full_like(coefs, fill_value=np.nan)
    ci_length = np.full_like(coefs, fill_value=np.nan)
    K = np.full_like(coefs, fill_value=np.nan)
    rmses = np.full_like(coefs, fill_value=np.nan)
    name_calib_method = np.full_like(coefs, fill_value="not specified", dtype=object)
    name_method = np.full_like(coefs, fill_value="not specified", dtype=object)
    ece_u = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    ece_q = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    ece_u_5 = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    ece_q_5 = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)    
    mce = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)
    ece_l2 = np.full(shape=(n_calib_methods, n_clipping_thresholds), fill_value=np.nan)

    ipw_coefs = np.full_like(coefs, fill_value=np.nan)
    plr_coefs = np.full_like(coefs, fill_value=np.nan)

    # set up the DoubleMLIRM models
    dml_irm = dml.DoubleMLIRM(data,
                            ml_g=learner_g,
                            ml_m=learner_m,
                            score=score,
                            n_folds=n_folds,
                            trimming_threshold=1e-12) 
    dml_irm.fit(n_jobs_cv=5)
    smpls = dml_irm.smpls[0] # only a single repetition & same sample split as in the main model
    dml_irm_ext = dml.DoubleMLIRM(data,
                                ml_g=dml.utils.DMLDummyRegressor(),
                                ml_m=dml.utils.DMLDummyClassifier(),
                                score=score,
                                n_folds=n_folds,
                                trimming_threshold=1e-12)
    # set up PLR model
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

            coefs[i_calib_method, i_clipping_threshold] = dml_irm_ext.coef[0]
            ses[i_calib_method, i_clipping_threshold] = dml_irm_ext.se[0]
            confint_calib = dml_irm_ext.confint()
            cover[i_calib_method, i_clipping_threshold] = (confint_calib.loc['d', '2.5 %'] < theta) & (theta < confint_calib.loc['d', '97.5 %'])
            ci_length[i_calib_method, i_clipping_threshold]  = confint_calib.loc['d', '97.5 %'] - confint_calib.loc['d', '2.5 %']
            rmses[i_calib_method, i_clipping_threshold] = np.sqrt(((calib_prop_score - treatment)**2).mean())
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

            ipw_coefs[i_calib_method, i_clipping_threshold] = compute_ipw_estimate(calib_prop_score, treatment, outcome)
            plr_coefs[i_calib_method, i_clipping_threshold] = dml_plr_ext.coef[0]            

    results_dict = {
        "coefs": coefs,
        "ses": ses,
        "cover": cover,
        "ci_length": ci_length,
        "K": K,
        "rmses": rmses,
        "method": name_method,
        "calib_method": name_calib_method,
        "ipw_coefs": ipw_coefs,
        "plr_coefs": plr_coefs,
        "theta": theta,
        "ece_u": ece_u,
        "ece_q": ece_q,
        "ece_u_5": ece_u_5,
        "ece_q_5": ece_q_5,        
        "ece_l2": ece_l2,        
        "mce": mce}
    return results_dict


# ignore warnings (prop. score is often close to zero or one)
warnings.filterwarnings('ignore')
df = pd.DataFrame(columns=["coefs","ses", "cover", "ci_length", "K", "rmses", "method", "calib_method", 
                           "ipw_coefs", "plr_coefs", "learner_g", "learner_m",
                           "n_obs", "clipping_threshold", "repetition", 'theta', 
                           'ece_u','ece_q','ece_u_5','ece_q_5','ece_l2','mce', 'overlap'])


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
            # create DoubleMLData object 
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
                    dml_results = estimate_irm(dml_data, theta, learner_g, learner_m, n_folds, score, clipping_thresholds, calib_methods, m_0)
                    for i_clipping_threshold, clipping_threshold in enumerate(clipping_thresholds):
                        # store results
                        df_small = pd.DataFrame({
                            "coefs": dml_results["coefs"][:, i_clipping_threshold],
                            "ses": dml_results["ses"][:, i_clipping_threshold],
                            "cover": dml_results["cover"][:, i_clipping_threshold],
                            "ci_length": dml_results["ci_length"][:, i_clipping_threshold],
                            "K": dml_results["K"][:, i_clipping_threshold],
                            "rmses": dml_results["rmses"][:, i_clipping_threshold],
                            "method": dml_results["method"][:, i_clipping_threshold],
                            "calib_method": dml_results["calib_method"][:, i_clipping_threshold],
                            "ipw_coefs": dml_results["ipw_coefs"][:, i_clipping_threshold],
                            "plr_coefs": dml_results["plr_coefs"][:, i_clipping_threshold],
                            "ece_u": dml_results["ece_u"][:, i_clipping_threshold],
                            "ece_q": dml_results["ece_q"][:, i_clipping_threshold],
                            "ece_u_5": dml_results["ece_u_5"][:, i_clipping_threshold],
                            "ece_q_5": dml_results["ece_q_5"][:, i_clipping_threshold],                                
                            "ece_l2": dml_results["ece_l2"][:, i_clipping_threshold],
                            "mce": dml_results["mce"][:, i_clipping_threshold],  
                            "learner_g": name_learner_g,
                            "learner_m": name_learner_m,
                            "n_obs": n_obs,
                            "clipping_threshold": clipping_threshold,
                            "repetition": i_rep,
                            "theta": theta,
                            "overlap": overlap
                        })
                            
                        df = pd.concat([df, df_small], ignore_index=True)


output_file = f'02_results_drug/ranks_{job_id}/output_rank_{rank:03d}.pkl'
df.to_pickle(output_file)

end_time = time.time()

if rank == 0:
    total_runtime = end_time - start_time
    print(f"Total Runtime: {total_runtime:.2f} seconds")


