################################# Calibration ########################################

import numpy as np #type:ignore
import pandas as pd #type:ignore
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split # type: ignore
from sklearn.isotonic import IsotonicRegression # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
from venn_abers import VennAbersCalibrator
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


def alg_1_uncalibrated(
        propensity_score,
        clipping_threshold):
    calibrated_prop_score = np.clip(propensity_score, 0.0 + clipping_threshold, 1.0 - clipping_threshold)
    return calibrated_prop_score

def alg_2_nested_cross_fitting_calib(
        covariates,
        treatment,
        learner_m,
        calib_method,
        clipping_threshold,
        smpls):
    
    if isinstance(covariates, pd.DataFrame):
        covariates = covariates.to_numpy()
    else:
        covariates = covariates
    
    if smpls is None:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        smpls = kf.split(X=np.zeros(len(covariates)), y=treatment)
    
    calibrated_prop_score = np.full_like(treatment, fill_value=np.nan,dtype=np.float64)

    for train_idx, test_idx in smpls:
        X_train, X_val, d_train, d_val = train_test_split(
            covariates[train_idx, :], treatment[train_idx],
            train_size=0.5, shuffle=False, random_state=42)   
        
        train_fit = learner_m.fit(X_train, d_train)
        prop_score_test = train_fit.predict_proba(covariates[test_idx, :])
        prop_score_val = train_fit.predict_proba(X_val)

        if calib_method == 'isotonic':
            isotonic_reg_model = IsotonicRegression(
                increasing=True,
                y_max=1.0 - clipping_threshold,
                y_min=0.0 + clipping_threshold,
                out_of_bounds='clip')
            iso_reg_val = isotonic_reg_model.fit(prop_score_val[:, 1], d_val)
            calibrated_prop_score[test_idx] = iso_reg_val.predict(prop_score_test[:, 1])
        
        elif calib_method == 'platt':
            platt_reg_model = CalibratedClassifierCV(
                base_estimator=train_fit,
                cv="prefit",
                method="sigmoid")
            platt_reg_val = platt_reg_model.fit(X_val, d_val)
            calibrated_prop_score[test_idx] = platt_reg_val.predict_proba(covariates[test_idx, :])[:, 1]

        elif calib_method == 'ivap':
            va = VennAbersCalibrator(precision=5)
            calibrated_prop_score[test_idx] = va.predict_proba(p_cal=prop_score_val, y_cal=d_val, p_test=prop_score_test, loss='Brier')[:, 1]

        else:
            raise ValueError("Unknown calibration method. Choose 'isotonic', 'platt', or 'ivap'.")

    return np.clip(calibrated_prop_score, clipping_threshold, 1 - clipping_threshold)

def alg_3_cross_fitted_calib(
        covariates,
        propensity_score,
        treatment,
        learner_m,
        calib_method,
        clipping_threshold,
        smpls):
    
    if isinstance(covariates, pd.DataFrame):
        covariates = covariates.to_numpy()
    else:
        covariates = covariates

    if smpls is None:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        smpls = list(kf.split(X=np.zeros(len(treatment)), y=treatment))
    
    covariates=covariates
    calibrated_prop_score = np.full(len(treatment), np.nan, dtype=np.float64)
    
    if calib_method == 'isotonic':
        isotonic_reg_model = IsotonicRegression(
            increasing=True,
            y_max=1.0 - clipping_threshold,
            y_min=0.0 + clipping_threshold,
            out_of_bounds='clip')
        
        test_smpls = [[test_idx, test_idx] for _, test_idx in smpls]
        calibrated_prop_score = cross_val_predict(
            estimator=isotonic_reg_model,
            X=propensity_score.reshape(-1, 1),
            y=treatment,
            cv=test_smpls,
            method='predict')

    elif calib_method == 'platt':
        test_smpls = [[test_idx, test_idx] for _, test_idx in smpls]
        platt_reg_model = CalibratedClassifierCV(
            estimator=learner_m,  # Changed from base_estimator to estimator
            method="sigmoid",
            ensemble=True  # Explicitly set ensemble behavior
        )
        
        calibrated_prop_score = cross_val_predict(
            estimator=platt_reg_model,
            X=propensity_score.reshape(-1, 1),
            y=treatment,
            cv=test_smpls,
            method='predict_proba',
            n_jobs=-1  # Consider adding parallelization
        )[:, 1]

    elif calib_method == 'ivap':
        for train_idx, test_idx in smpls:
            train_fit = learner_m.fit(covariates[train_idx, :], treatment[train_idx])
            
            kf_calib = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for cal_fold_idx, proper_test_idx in kf_calib.split(covariates[test_idx], treatment[test_idx]):
                cal_idx = test_idx[cal_fold_idx]  
                proper_test_idx = test_idx[proper_test_idx]  

                prop_score_cal = train_fit.predict_proba(covariates[cal_idx, :])
                prop_score_test = train_fit.predict_proba(covariates[proper_test_idx, :])

                va = VennAbersCalibrator(precision=5)
                calibrated_prop_score[proper_test_idx] = va.predict_proba(
                    p_cal=prop_score_cal,
                    y_cal=treatment[cal_idx],
                    p_test=prop_score_test
                )[:, 1]

    else:
        raise ValueError("Unknown calibration method. Choose 'isotonic', 'platt', or 'cvap'.")   
    
    return np.clip(calibrated_prop_score, clipping_threshold, 1 - clipping_threshold)

def alg_4_single_split_calib(
        covariates,
        treatment,
        learner_m,
        calib_method,
        clipping_threshold):

    if isinstance(covariates, pd.DataFrame):
        covariates = covariates.to_numpy()
    else:
        covariates = covariates
        
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    smpls = list(kf.split(X=covariates, y=treatment))

    if not smpls:
       raise ValueError("KFold-Generator has no folds")
    
    calibrated_prop_score = np.zeros_like(treatment, dtype=float)
    
    for train_idx, cal_idx in smpls:
        X_train, X_cal = covariates[train_idx], covariates[cal_idx]
        d_train, d_cal = treatment[train_idx], treatment[cal_idx]

        train_fit = learner_m.fit(X_train, d_train)
        prop_score_train = train_fit.predict_proba(X_train)
        prop_score_cal = train_fit.predict_proba(X_cal)
        
        if calib_method == 'isotonic':
            iso_reg = IsotonicRegression(
                increasing=True,
                y_min=clipping_threshold,
                y_max=1.0 - clipping_threshold,
                out_of_bounds='clip'
            )
            iso_reg.fit(prop_score_cal[:, 1], d_cal)

            calibrated_prop_score[train_idx] = iso_reg.predict(prop_score_train[:, 1])

        elif calib_method == 'platt':
            platt_reg_model = CalibratedClassifierCV(
                base_estimator=learner_m,
                method="sigmoid")
            platt_reg_val = platt_reg_model.fit(X_train, d_train)
            calibrated_prop_score[cal_idx] = platt_reg_val.predict_proba(X_cal)[:, 1]

        elif calib_method == 'ivap':
            va = VennAbersCalibrator(precision=5)
            calibrated_prop_score[train_idx] = va.predict_proba(p_cal=prop_score_cal, y_cal=d_cal, p_test=prop_score_train, loss='Brier')[:, 1]

        else:
            raise ValueError("Unknown calibration method. Choose 'isotonic', 'ivap'.")

    return calibrated_prop_score

def alg_5_full_sample_calib(
        covariates,
        propensity_score,
        treatment,
        learner_m,
        calib_method,
        clipping_threshold,
        smpls):

    if isinstance(covariates, pd.DataFrame):
        covariates = covariates.to_numpy()
    else:
        covariates = covariates

    if calib_method == 'isotonic':
        isotonic_reg_model = IsotonicRegression(
            increasing=True,
            y_max=1.0 - clipping_threshold,
            y_min=0.0 + clipping_threshold,
            out_of_bounds='clip')
        isotonic_reg_model.fit(propensity_score, treatment)
        calibrated_prop_score = isotonic_reg_model.predict(propensity_score)

    elif calib_method == 'ivap':

        if smpls is None:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            smpls = list(kf.split(X=np.zeros(len(treatment)), y=treatment))

        propensity_scores = cross_val_predict(
            estimator=learner_m,
            X=covariates,
            y=treatment,
            cv=smpls,
            method='predict_proba')
        calibrated_prop_score = np.zeros_like(treatment, dtype=float)

        kf_calib = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for cal_idx, test_idx in kf_calib.split(covariates, treatment):

            prop_scores_cal = propensity_scores[cal_idx]
            prop_scores_test = propensity_scores[test_idx]

            va = VennAbersCalibrator(precision=5)
            calibrated_prop_score[test_idx] = va.predict_proba(
                p_cal=prop_scores_cal,
                y_cal=treatment[cal_idx],
                p_test=prop_scores_test
            )[:, 1]

    elif calib_method == 'platt':

        if smpls is None:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            smpls = list(kf.split(X=np.zeros(len(treatment)), y=treatment))
    
        calibrated_prop_score = np.zeros_like(treatment, dtype=float)

        for train_idx, test_idx in smpls: 

            platt_reg_model = CalibratedClassifierCV(
                estimator=learner_m,  
                method="sigmoid",
                ensemble=True
            )
            
            platt_reg_model.fit(covariates[train_idx, :], treatment[train_idx])
            calibrated_prop_score[test_idx] = platt_reg_model.predict_proba(covariates[test_idx, :])[:, 1]

    else:
        raise ValueError("Unknown calibration method. Choose 'isotonic','platt','ivap'.")           
    
    return calibrated_prop_score


def calibrate_propensity_score(
                method,
                covariates,
                propensity_score, 
                treatment, 
                learner_m,
                calib_method,
                clipping_threshold,
                smpls,
                true_propensity_score):
    
    if method == 'alg-1-uncalibrated':
        calibrated_prop_score = alg_1_uncalibrated(propensity_score,clipping_threshold)

    elif method == 'alg-2-nested-cross-fitting-calib':
        calibrated_prop_score = alg_2_nested_cross_fitting_calib(covariates,treatment,learner_m,calib_method,clipping_threshold,
        smpls)

    elif method == 'alg-3-cross-fitted-calib':
        calibrated_prop_score = alg_3_cross_fitted_calib(covariates,propensity_score,treatment,learner_m,calib_method,clipping_threshold,
        smpls)

    elif method == 'alg-4-single-split-calib':
        calibrated_prop_score = alg_4_single_split_calib(covariates,treatment,learner_m,calib_method,clipping_threshold)

    elif method == 'alg-5-full-sample-calib':
        calibrated_prop_score = alg_5_full_sample_calib(covariates,propensity_score,treatment,learner_m,calib_method,clipping_threshold,smpls)

    elif method == 'oracle':
        assert true_propensity_score is not None
        calibrated_prop_score = np.clip(true_propensity_score, 0.0 + clipping_threshold, 1.0 - clipping_threshold)
    
    else:
        raise ValueError(f'Calibration method {method} not implemented.')
    return calibrated_prop_score


def compute_ipw_estimate(propensity_score, treatment, outcome, stabilized=False):
    weights_treated = treatment / propensity_score
    weights_control = (1 - treatment) / (1 - propensity_score)
    if not stabilized:
        y0 = np.mean(outcome * weights_control)
        y1 = np.mean(outcome * weights_treated)
    else:
        weights_treated = treatment / propensity_score
        weights_control = (1 - treatment) / (1 - propensity_score)
        y0 = np.sum(outcome * weights_control) / np.sum(weights_control)
        y1 = np.sum(outcome * weights_treated) / np.sum(weights_treated)

    ipw_estimate = y1 - y0
    return ipw_estimate

def compute_ipw_wls(propensity_score, treatment, outcome, true_ate=None, stabilized=False):
    df = pd.DataFrame({
        'd': treatment,
        'outcome': outcome,
        'propensity_score': np.clip(propensity_score, 1e-9, 1-1e-9)  # Prevent division by zero
    })
    
    if stabilized:
        m_hat = np.mean(df['d'])
        weights = (m_hat/df['pscore'])*df['d'] + \
                 ((1-m_hat)/(1-df['propensity_score']))*(1-df['d'])
    else:
        weights = 1 / (df['propensity_score'] * df['d'] + (1 - df['propensity_score']) * (1 - df['d']))
    
    # Fit WLS model
    model = smf.wls('outcome ~ d', weights=weights, data=df).fit()
    
    # Extract results
    results = {
        'IPW_coefs': model.params['d'],
        'IPW_ses': model.bse['d'],
        'IPW_ci_length': model.conf_int().loc['d', 1] - model.conf_int().loc['d', 0]
    }
    
    if true_ate is not None:
        ci_lower = model.conf_int().loc['d', 0]
        ci_upper = model.conf_int().loc['d', 1]
        results['IPW_cover'] = (ci_lower < true_ate) & (true_ate < ci_upper)
    
    return results


def calibration_errors(propensity_score, d, strategy = 'quantile',norm='l1',n_bins= float(10)):
    n_bins = int(n_bins)
    if(strategy == 'uniform'):
        quantiles_m = np.arange(0, 1, 1.0 / n_bins)
    elif(strategy == 'quantile'):
        quantiles_m = np.percentile(propensity_score, np.arange(0, 1, 1.0 / n_bins) * 100)
    sample_weight = np.ones(d.shape[0])
    avg_pred_true = np.zeros(n_bins)     
    bin_centroid = np.zeros(n_bins)
    delta_count = np.zeros(n_bins)
    threshold_indices = np.searchsorted(propensity_score, quantiles_m).tolist()
    threshold_indices.append(d.shape[0])

    loss = 0
    count = float(sample_weight.sum())
    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]

        if i_end == i_start:
            continue
        delta_count[i] = float(sample_weight[i_start:i_end].sum())
        avg_pred_true[i] = (np.dot(d[i_start:i_end],
                                   sample_weight[i_start:i_end])
                            / delta_count[i])
        bin_centroid[i] = (np.dot(propensity_score[i_start:i_end],
                                  sample_weight[i_start:i_end])
                           / delta_count[i])

    if norm == "max":
        loss = np.max(np.abs(avg_pred_true - bin_centroid))
    elif norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        loss = np.sum(delta_loss) / count
    elif norm == "l2":
        delta_loss = (avg_pred_true - bin_centroid)**2 * delta_count
        loss = np.sum(delta_loss) / count
        loss = np.sqrt(max(loss, 0.))
    return loss

def compute_ci_metrics(true_effect, lower, upper):
    """Helper function to compute coverage and CI length"""
    return {
        'ci_length': upper - lower,
        'cover': (lower < true_effect) & (true_effect < upper)
    }

def create_results_dict(base_shape, fill_data, metrics_dict):
    
    template = {
        key: np.full(base_shape, np.nan) 
        for key in metrics_dict
    }
    
    # Update with actual data, ensuring array shapes match
    for k, v in fill_data.items():
        template[k] = np.broadcast_to(v, base_shape)
    
    return template