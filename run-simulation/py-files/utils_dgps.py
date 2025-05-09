################################# DGPs ########################################

import numpy as np #type:ignore
import pandas as pd #type:ignore
from scipy.linalg import toeplitz
from scipy import stats
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

def make_high_dimensional_data(n_obs=500, dim_x=200, theta=0, R2_d=0.5, R2_y=0.5, return_type='DoubleMLData'):

    # inspired by https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12723, see suplement
    v = np.random.uniform(size=[n_obs, ])
    zeta = np.random.standard_normal(size=[n_obs, ])

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y/((1-R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3. * R2_d/((1-R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    propensity_score = xx/(1+xx)
    d = 1. * (propensity_score > v)

    y_0 = 0 * theta + 0 * np.dot(x, np.multiply(beta, c_y)) + zeta
    y_1 = 1 * theta + 1 * np.dot(x, np.multiply(beta, c_y)) + zeta 
    y   = d*y_1 + (1-d)*y_0
    
    theta = 1/n_obs* (sum(y_1) - sum(y_0))

    data_dict = {
        'outcome': y,
        'treatment': d,
        'covariates': x,
        'propensity_score': propensity_score,
        'treatment_effect': theta
    }
    return data_dict
    
def make_overlap_data(n_obs=500, overlap=0.1, return_type='DoubleMLData'):
    # inspired by https://arxiv.org/abs/2306.00382, see Appendix B

    # Simulate the second confounder dependend on gender
    # Assume if gender is 1 (male), age tends to be lower on average
    x_1 = np.random.binomial(1, 0.5, n_obs)
    mean_age = np.where(x_1 == 1, 49, 51)
    std_age = np.where(x_1 == 1, 7, 8)

    shape_age = (mean_age / std_age) ** 2
    scale_age = (std_age ** 2) / mean_age
    x_2 = np.random.gamma(shape_age, scale_age)

    # Simulate the third confounder, which depends on age
    mean_disease_severity = np.clip((x_2 - 20) / 20, 0.1, 5)
    x_3 = np.array([beta.rvs(ms, 2) for ms in mean_disease_severity])

    x = pd.DataFrame({'x1': x_1, 'x2': x_2, 'x3': x_3})
    
    # Coefficients for linear combination
    coefficients = [
        (-0.4, 0.2, 0.8),
        ((-0.4 +2*(1 - overlap)), 0.2, 0.8),
        ((-0.4 +2*(1 - overlap)), 0.3, 1),
        ((-0.4 +2*(1 - overlap)), 0.1, 1.2),
        ((-0.4 +2*(1 - overlap)), 0.1, 1.2)
    ]

    normalize = lambda arr: (arr - arr.min()) / (arr.max() - arr.min())
    linear_combs = [(b0 + b1 * normalize(x['x2']) + b2 * normalize(x['x3']) + np.random.normal(0, 0.5, x.shape[0])) for b0, b1, b2 in coefficients]

    # Apply logistic transformation and add small noise
    propensity_scores = [1 / (1 + np.exp(-lc)) for lc in linear_combs]

    # Initialize treatment assignment with the first set of propensity scores
    m_0 = np.copy(propensity_scores[0])

    conditions = [
        ((x['x1'] == 0) & (x['x2'] > 55) & (x['x3'] <= 0.55), propensity_scores[1]),
        ((x['x1'] == 1) & (x['x2'] > 55) & (x['x3'] <= 0.55), propensity_scores[2]),
        ((x['x1'] == 0) & (x['x3'] > 0.55), propensity_scores[3]),
        ((x['x1'] == 1) & (x['x3'] > 0.55), propensity_scores[4])
    ]

    # Simulate treatment assignment
    for condition, score in conditions:
        m_0[condition] = score[condition]

    # Treatment assignment
    d = np.random.binomial(1, m_0)

    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    else:
        x = x

    # Simulate outcome based on covariates and treatment
    y_1 = np.random.poisson(2 + 0.5*x_1 + 0.03*x_2 + 2*x_3 - 1)
    y_0 = np.random.poisson(2 + 0.5*x_1 + 0.03*x_2 + 2*x_3 - 0)
    y = d*y_1 + (1-d)*y_0
    theta = 1/n_obs* (sum(y_1) - sum(y_0))

    data_dict = {
        'outcome': y,
        'treatment': d,
        'covariates': x,
        'propensity_score': m_0,
        'treatment_effect': theta
    }

    return data_dict    

def make_prop_misspecification_data(n_obs=1000, return_type='DoubleMLData'):
    # Setting 1, from https://arxiv.org/abs/2302.14011 (Appendix D.1), low-dimensional + discrete y
    x = pd.DataFrame({
            'x1': np.random.uniform(-1, 1, n_obs),
            'x2': np.random.uniform(-1, 1, n_obs),
            'x3': np.random.uniform(-1, 1, n_obs),
            'x4': np.random.uniform(-1, 1, n_obs)
    })   
    xx = np.exp(-0.25 + x['x1'] + 0.5 * x['x2'] - x['x3'] + 0.5 * x['x4'])
    propensity_score = np.array(xx/(1+xx))
    d = np.random.binomial(1, propensity_score, n_obs)

    mu_0 = 1.5 + 1*0 + 2*0*np.abs(x['x1']) * np.abs(x['x2']) + 2.5*(1-0)*np.abs(x['x2'])*x['x3']+ 2.5*x['x3'] - 3*(1-0) * np.sqrt(np.abs(x['x4'])) - 1.5*0*(x['x2'] <
    .5)+1.5 * (1-0) * (x['x4'] < 0)
    mu_1 = 1.5 + 1*1 + 2*1*np.abs(x['x1']) * np.abs(x['x2']) + 2.5*(1-1)*np.abs(x['x2'])*x['x3']+ 2.5*x['x3'] - 3*(1-1) * np.sqrt(np.abs(x['x4'])) - 1.5*1*(x['x2'] <
    .5)+1.5 * (1-1) * (x['x4'] < 0)


    y_0 = np.random.normal(loc=mu_0,scale=1, size=n_obs)
    y_1 = np.random.normal(loc=mu_1,scale=1, size=n_obs)
    y   = d*y_1 + (1-d)*y_0

    theta = 1/n_obs* (sum(y_1) - sum(y_0))

    data_dict = {
        'outcome': y,
        'treatment': d,
        'covariates': x,
        'propensity_score': propensity_score,
        'treatment_effect': theta
    }
    return data_dict

def make_unbalanced_data(n_obs=500,dim_x=20,share_treated=0.05,simulation_type='A', sigma=1,return_type='DoubleMLData'):
    # inspired by https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12723, see supplement
    # & https://arxiv.org/abs/2302.14011, see Appendix D.1, Setting 2

    x = np.random.uniform(size=(n_obs, dim_x))
    # define propensity score as in Kuenzel et. al (2019) [https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116] and assign treatment
    if simulation_type == 'A':
        alpha = share_treated*21/31
        propensity_score = alpha*(1+stats.beta.cdf(np.min(x[:,:2], axis=1), 2, 4))
    elif simulation_type == 'B':
        propensity_score= np.full(n_obs,share_treated)
    d = np.random.binomial(1, propensity_score)
    # baseline effect is the scaled Friedmann (2019) function  [https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full]
    b = np.sin(np.pi*x[:,0]*x[:,1]) + 2*(x[:,2]-0.5)**2 + x[:,3] + 0.5*x[:,4]
    # individual treatment effects as in set-up A in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    tau = x[:,0]+x[:,1]

    # Simulate outcome based on covariates and treatment
    y_1 = b + (1-0.5) * tau + sigma * np.random.normal(size=x.shape[0]) 
    y_0  = b + (0-0.5) * tau + sigma * np.random.normal(size=x.shape[0]) 
    y = d*y_1 + (1-d)*y_0

    theta = 1/n_obs* (sum(y_1) - sum(y_0))

    data_dict = {
        'outcome': y,
        'treatment': d,
        'covariates': x,
        'propensity_score': propensity_score,
        'treatment_effect': theta
    }
    return data_dict


def dgp_wrapper(dgp_type, **kwargs):
    if dgp_type == 'sim_high_dimensional':
        dgp_dict = make_high_dimensional_data(**kwargs)
    elif dgp_type == 'sim_overlap':
        dgp_dict = make_overlap_data(**kwargs)
    elif dgp_type == 'sim_unbalanced':
        dgp_dict = make_unbalanced_data(**kwargs)   
    elif dgp_type == 'sim_prop_misspecification':
        dgp_dict = make_prop_misspecification_data(**kwargs)
    else:
        raise ValueError(f'DGP type {dgp_type} not implemented.')

    return dgp_dict