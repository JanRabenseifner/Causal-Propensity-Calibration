import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import plotnine as p9
import seaborn as sns


class BalanceTable:
    def __init__(self, df, cov_cols, treatment, weights):
        """
        Initialize the BalanceTable class and compute the standardized mean difference (SMD).

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            cov_cols (list): A list of columns as covariates.
            treatment (str): The column name of the treatment variable.
            weights (pd.Series or np.ndarray): A vector of weights
        """
        self.df = df
        self.cov_cols = cov_cols
        self.treatment = treatment
        self.weights = weights

        self.df_effective_sample_size, self.df_smd = self.compute_smd()

    @property
    def effective_sample_size(self):
        return self.df_effective_sample_size

    @property
    def smd_table(self):
        return self.df_smd

    def __str__(self):
        summary = f"Effective Sample Size:\n{self.df_effective_sample_size}\n\nSMD Table:\n{self.df_smd}"
        return summary

    def compute_smd(self):
        """
        Compute the standardized mean difference (SMD) for covariates in a dataframe.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            cov_cols (list): A list of columns as covariates.
            treatment (str): The column name of the treatment variable.
            weights (pd.Series or np.ndarray): A vector of weights

        Returns: Tuple[pd.DataFrame, pd.DataFrame]
            pd.DataFrame: A pandas.DataFrame containing the effective sample size with and without weighting.
            pd.DataFrame: A pandas.DataFrame containing the SMD for each covariate with weighting and without.
        """
        df_smd = pd.DataFrame()

        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0

        df_treated = self.df[treated]
        df_control = self.df[control]

        df_effective_sample_size = pd.DataFrame(
            {"Control": [control.sum(), np.sum(self.weights[control])**2 / np.sum(self.weights[control]**2)],
             "Treated": [treated.sum(), np.sum(self.weights[treated])**2 / np.sum(self.weights[treated]**2)],
             },
            index=['Unadjusted', 'Adjusted']).round(2)

        sum_weights = self.weights.sum()
        weights_factor = sum_weights / ((sum_weights)**2 - (self.weights**2).sum())

        for column in self.cov_cols:

            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                weighted_mean = np.average(self.df[column], weights=self.weights)
                sd2 = weights_factor * np.sum(self.weights * (self.df[column] - weighted_mean)**2)
                sd = np.sqrt(sd2)

            treated_mean_balanced = np.average(df_treated[column], weights=self.weights[df_treated.index])
            control_mean_balanced = np.average(df_control[column], weights=self.weights[df_control.index])

            treated_mean_unbalanced = np.mean(df_treated[column])
            control_mean_unbalanced = np.mean(df_control[column])

            smd_balanced = (treated_mean_balanced - control_mean_balanced) / sd
            smd_unbalanced = (treated_mean_unbalanced - control_mean_unbalanced) / sd

            new_row = pd.DataFrame({
                'covariate': [column],
                'type': ['Binary' if col_is_binary else 'Contin.'],
                'diff_unadj': [smd_unbalanced],
                'diff_adj': [smd_balanced],
            })
            df_smd = pd.concat([df_smd, new_row], ignore_index=True)

        return df_effective_sample_size, df_smd.round(4)

    def love_plot(self, thresholds=[0.1]):
        """
        Generate a love plot for the corresponding standardized mean difference (SMD) table.
        """

        df_renamed = self.df_smd.rename(columns={
            'diff_unadj': 'Unadjusted',
            'diff_adj': 'Adjusted'
        })

        cov_order = df_renamed.sort_values(
            by='Unadjusted', ascending=True
        )["covariate"].tolist()

        df_plot = df_renamed.melt(
            id_vars=['covariate', 'type'],
            value_vars=['Unadjusted', 'Adjusted'],
            var_name='Sample',
            value_name='SMD')

        p = (p9.ggplot(df_plot, p9.aes(x='covariate', y='SMD', fill='Sample', group='Sample')) +
             p9.geom_point(size=2) +
             p9.geom_line(p9.aes(color="Sample"), size=.5, show_legend=False) +
             p9.theme_minimal() +
             p9.theme_bw() +
             p9.scale_x_discrete(limits=cov_order) +
             p9.coord_flip() +
             p9.labs(title='Covariate Balance', x='Covariate', y='Standardized Mean Difference (SMD)'))

        for threshold in thresholds:
            p += p9.geom_hline(yintercept=abs(threshold), linetype='dashed', color='red')
            p += p9.geom_hline(yintercept=-abs(threshold), linetype='dashed', color='red')

        return p

    def balance_plot(self, covariate, n_bins=20):
        """
        Generate a balance plot for the specified covariate.

        Parameters:
            covariate (str): The covariate to plot.
            n_bins (int): The number of bins to use for the histogram.
        """

        cov_is_binary = set(self.df[covariate].unique()).issubset({0, 1})

        treated = self.df[self.treatment] == 1

        df_plot = pd.DataFrame({
            "covariate": self.df[covariate],
            "unadjusted": 1.0,
            "adjusted": self.weights,
            "treatment": np.where(treated, "Treated", "Control")
        }).melt(
            value_vars=["unadjusted", "adjusted"],
            id_vars=["treatment", "covariate"],
            var_name="Sample",
            value_name="weight")

        # Create subplots (2 subplots for 'unadjusted' and 'adjusted')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        palette = sns.color_palette()
        colors = {'Treated': palette[0], 'Control': palette[1]}

        for i, sample in enumerate(["unadjusted", "adjusted"]):
            ax = axes[i]
            ax.set_title(f"{sample.capitalize()} Distribution")
            ax.set_xlabel(covariate)
            ax.set_ylabel('Proportion')

            sample_data = df_plot[df_plot['Sample'] == sample]
            if cov_is_binary:
                unique_values = sorted(sample_data['covariate'].unique())
                ax.set_xticks(unique_values)
                ax.set_xticklabels(unique_values)
            else:
                bin_range = (sample_data['covariate'].min(), sample_data['covariate'].max())

            for treatment_group in ['Treated', 'Control']:
                treatment_data = sample_data[sample_data['treatment'] == treatment_group].copy()

                if cov_is_binary:
                    # adjust x to avoid overlapping bars
                    x_adjustment = 0.2 if treatment_group == 'Control' else -0.2
                    treatment_data["covariate"] += x_adjustment
                    sns.histplot(
                        treatment_data,
                        x="covariate",
                        weights="weight",
                        alpha=0.6,
                        stat="proportion",
                        color=colors[treatment_group],
                        label=treatment_group,
                        discrete=True,
                        shrink=0.4,
                        ax=ax
                    )
                else:
                    sns.histplot(
                        treatment_data,
                        x="covariate",
                        weights="weight",
                        alpha=0.6,
                        stat="proportion",
                        bins=n_bins,
                        color=colors[treatment_group],
                        label=treatment_group,
                        binrange=bin_range,
                        ax=ax
                    )

            ax.legend(title='Treatment')

        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Distributional Balance for {covariate}', fontsize=16)

        return fig

    def plot_prop_balance(self, propensity_score, covs=None, n_bins=10, confint=False, level=0.95):

        if covs is None:
            covs = self.cov_cols

        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0

        bin_edges = np.linspace(propensity_score.min(),
                                propensity_score.max(), n_bins + 1)
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        bin_size = [(bin_edges[i+1] - bin_edges[i]) for i in range(n_bins)]

        bin_obs_list = [None] * n_bins
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == 0:
                bin_obs_list[i] = (propensity_score >= bin_start) & (propensity_score <= bin_end)
            else:
                bin_obs_list[i] = (propensity_score > bin_start) & (propensity_score <= bin_end)

        df_plot = pd.DataFrame()
        smd = np.full((n_bins, len(covs)), np.nan)
        lower_bounds = np.full((n_bins, len(covs)), np.nan)
        upper_bounds = np.full((n_bins, len(covs)), np.nan)
        for j, column in enumerate(covs):
            # scaling on the whole sample
            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                s0_sample = self.df[column][control].std()
                s1_sample = self.df[column][treated].std()
                # using pooled variance from cobalt
                sd2 = (s0_sample**2 + s1_sample**2) / 2
                sd = np.sqrt(sd2)

            for i, bin_obs in enumerate(bin_obs_list):

                df_bin_treated = self.df[treated & bin_obs]
                df_bin_control = self.df[control & bin_obs]
                n0 = len(df_bin_control)
                n1 = len(df_bin_treated)

                if n0 == 0 and n1 == 0:
                    bin_smd = 0
                    bin_lower_bound = 0
                    bin_upper_bound = 0
                elif n1 == 0:
                    warnings.warn(f"No treated observations in bin {i}. Setting SMD to 5.")
                    bin_smd = 5
                    bin_lower_bound = 5
                    bin_upper_bound = 5
                elif n0 == 0:
                    warnings.warn(f"No control observations in bin {i}. Setting SMD to -5.")
                    bin_smd = -5
                    bin_lower_bound = -5
                    bin_upper_bound = -5
                else:
                    treated_mean = np.average(df_bin_treated[column])
                    control_mean = np.average(df_bin_control[column])
                    mean_diff = treated_mean - control_mean
                    bin_smd = mean_diff / sd

                    if n1 < 2 or n0 < 2:
                        warnings.warn(f"Bin {i} has less than 2 observations in one group. No CI computed.")
                        bin_lower_bound = np.nan
                        bin_upper_bound = np.nan
                    else:
                        s0 = df_bin_control[column].std()
                        s1 = df_bin_treated[column].std()

                        sd_diff = np.sqrt((s0**2 / n0) + (s1**2 / n1))

                        if n0 == 1 or n1 == 1:
                            df = 1
                        else:
                            df = ((s0**2 / n0) + (s1**2 / n1))**2 / (
                                ((s0**2 / n0)**2 / (n0 - 1)) + ((s1**2 / n1)**2 / (n1 - 1)))

                        t_critical = stats.t.ppf((1 + level) / 2, df)

                        bin_lower_bound = (mean_diff - t_critical * sd_diff) / sd
                        bin_upper_bound = (mean_diff + t_critical * sd_diff) / sd

                smd[i, j] = bin_smd
                lower_bounds[i, j] = bin_lower_bound
                upper_bounds[i, j] = bin_upper_bound

            df_plot_cov = pd.DataFrame({
                "bin_midpoints": bin_midpoints,
                "bin_size": bin_size,
                "smd": smd[:, j],
                "covariate": column,
                "lower_bound": lower_bounds[:, j],
                "upper_bound": upper_bounds[:, j]
            })

            df_plot = pd.concat([df_plot, df_plot_cov], ignore_index=True)

        p = (p9.ggplot(df_plot, p9.aes(x='bin_midpoints', y='smd', fill="covariate")) +
             p9.geom_point(size=2) +
             p9.geom_line(mapping=p9.aes(color="covariate"), size=.5) +
             p9.theme_minimal() +
             p9.theme_bw() +
             p9.coord_flip() +
             p9.labs(title='Covariate Balance', x='Propensity Score', y='Standardized Mean Difference (SMD)'))

        if confint:
            p += p9.geom_ribbon(p9.aes(ymin='lower_bound', ymax='upper_bound', fill="covariate"), alpha=0.2)

        return p


class BalanceTable_2:
    def __init__(self, df, cov_cols, treatment, weights, method=None):
        self.df = df
        self.cov_cols = cov_cols
        self.treatment = treatment
        self.weights = weights
        self.method = method
        self.df_effective_sample_size, self.df_smd = self.compute_smd()

    @property
    def effective_sample_size(self):
        return self.df_effective_sample_size

    @property
    def smd_table(self):
        return self.df_smd

    def __str__(self):
        summary = f"Effective Sample Size:\n{self.df_effective_sample_size}\n\nSMD Table:\n{self.df_smd}"
        return summary

    def compute_smd_2(self):
        df_smd = pd.DataFrame()

        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0

        df_treated = self.df[treated]
        df_control = self.df[control]

        df_effective_sample_size = pd.DataFrame(
            {"Control": [control.sum(), np.sum(self.weights[control])**2 / np.sum(self.weights[control]**2)],
             "Treated": [treated.sum(), np.sum(self.weights[treated])**2 / np.sum(self.weights[treated]**2)],
             },
            index=['Unadjusted', 'Adjusted']).round(2)

        sum_weights = self.weights.sum()
        weights_factor = sum_weights / ((sum_weights)**2 - (self.weights**2).sum())

        for column in self.cov_cols:
            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                weighted_mean = np.average(self.df[column], weights=self.weights)
                sd2 = weights_factor * np.sum(self.weights * (self.df[column] - weighted_mean)**2)
                sd = np.sqrt(sd2)

            treated_mean_balanced = np.average(df_treated[column], weights=self.weights[df_treated.index])
            control_mean_balanced = np.average(df_control[column], weights=self.weights[df_control.index])

            treated_mean_unbalanced = np.mean(df_treated[column])
            control_mean_unbalanced = np.mean(df_control[column])

            smd_balanced = (treated_mean_balanced - control_mean_balanced) / sd
            smd_unadjusted = (treated_mean_unbalanced - control_mean_unbalanced) / sd

            new_row = pd.DataFrame({
                'covariate': [column],
                'type': ['Binary' if col_is_binary else 'Contin.'],
                'diff_unadj': [smd_unadjusted],
                'diff_adj': [smd_balanced],
                'Method': [self.method]
            })
            df_smd = pd.concat([df_smd, new_row], ignore_index=True)

        return df_effective_sample_size, df_smd.round(4)
    

    def compute_smd(self):
        
        df_smd = pd.DataFrame()
        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0
        df_treated = self.df[treated]
        df_control = self.df[control]

        # Effective sample size calculation (unchanged)
        df_effective_sample_size = pd.DataFrame(
            {"Control": [control.sum(), np.sum(self.weights[control])**2 / np.sum(self.weights[control]**2)],
             "Treated": [treated.sum(), np.sum(self.weights[treated])**2 / np.sum(self.weights[treated]**2)]},
            index=['Unadjusted', 'Adjusted']
        ).round(2)

        # SMD calculation with proper pooling
        for column in self.cov_cols:
            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            
            # Adjusted (weighted) calculations
            treated_mean_balanced = np.average(df_treated[column], weights=self.weights[treated])
            control_mean_balanced = np.average(df_control[column], weights=self.weights[control])
            
            if col_is_binary:
                # Binary pooled proportion (weighted)
                sum_w_treated = np.sum(self.weights[treated])
                sum_w_control = np.sum(self.weights[control])
                pooled_p = (sum_w_treated*treated_mean_balanced + sum_w_control*control_mean_balanced) / (sum_w_treated + sum_w_control)
                sd_balanced = np.sqrt(pooled_p*(1-pooled_p))

  
            else:
                def weighted_variance(data, weights):
                    """Helper function to calculate weighted variance"""
                    mean = np.average(data, weights=weights)
                    variance = np.average((data - mean)**2, weights=weights)
                    return variance  
                
                # Continuous pooled variance (weighted)
                var_treated = weighted_variance(df_treated[column], self.weights[treated])
                var_control = weighted_variance(df_control[column], self.weights[control])
                sd_balanced = np.sqrt((var_treated + var_control)/2)

            # Unadjusted calculations
            treated_mean_unbalanced = df_treated[column].mean()
            control_mean_unbalanced = df_control[column].mean()
            
            if col_is_binary:
                # Binary pooled proportion (unweighted)
                n_treated = len(df_treated)
                n_control = len(df_control)
                pooled_p_unadj = (n_treated*treated_mean_unbalanced + n_control*control_mean_unbalanced)/(n_treated + n_control)
                sd_unadjusted = np.sqrt(pooled_p_unadj*(1-pooled_p_unadj))
            else:
                # Continuous pooled variance (unweighted)
                var_treated_unadj = df_treated[column].var(ddof=1)
                var_control_unadj = df_control[column].var(ddof=1)
                sd_unadjusted = np.sqrt((var_treated_unadj + var_control_unadj)/2)

            smd_balanced = (treated_mean_balanced - control_mean_balanced)/sd_balanced
            smd_unadjusted = (treated_mean_unbalanced - control_mean_unbalanced)/sd_unadjusted

            new_row = pd.DataFrame({
                'covariate': [column],
                'type': ['Binary' if col_is_binary else 'Contin.'],
                'diff_unadj': [smd_unadjusted],
                'diff_adj': [smd_balanced],
                'Method': [self.method]
            })
            df_smd = pd.concat([df_smd, new_row], ignore_index=True)

        return df_effective_sample_size, df_smd.round(4)
    


    def weighted_variance(data, weights):
        """Calculate weighted variance."""
        mean = np.average(data, weights=weights)
        variance = np.average((data - mean)**2, weights=weights)
        return variance



    def prepare_smd_data_2(self):
        df_renamed = self.df_smd.rename(columns={
            'diff_unadj': 'Unadjusted',
            'diff_adj': 'Adjusted'
        })

        # Replace inf values with NaN
        df_renamed.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Sort covariates by the value of unadjusted SMD
        sorted_covariates = df_renamed.sort_values(by='Unadjusted', ascending=False)['covariate'].unique()

        df_plot = df_renamed.melt(
            id_vars=['covariate', 'type', 'Method'],
            value_vars=['Unadjusted', 'Adjusted'],
            var_name='Sample',
            value_name='SMD')
        df_plot['covariate'] = pd.Categorical(df_plot['covariate'], categories=sorted_covariates, ordered=True)

        return df_plot
    

    def prepare_smd_data(self):
        df_renamed = self.df_smd.rename(columns={
            'diff_unadj': 'Unadjusted',
            'diff_adj': 'Adjusted'
        })

        # Replace inf values with NaN
        df_renamed.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Sort covariates by the value of unadjusted SMD
        sorted_covariates = df_renamed.sort_values(by='Unadjusted', ascending=False)['covariate'].unique()

        df_plot = df_renamed.melt(
            id_vars=['covariate', 'type', 'Method'],  # Include 'Method' if it's part of your DataFrame
            value_vars=['Unadjusted', 'Adjusted'],
            var_name='Sample',
            value_name='SMD')
        df_plot['covariate'] = pd.Categorical(df_plot['covariate'], categories=sorted_covariates, ordered=True)

        return df_plot


   
    def plot_prop_balance_data(self, propensity_score, covs=None, n_bins=10, confint=False, level=0.95):
        
        if covs is None:
            covs = self.cov_cols
        
        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0
        
        bin_edges = np.linspace(propensity_score.min(), propensity_score.max(), n_bins + 1)
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        bin_size = [(bin_edges[i+1] - bin_edges[i]) for i in range(n_bins)]
        
        bin_obs_list = [None] * n_bins
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == 0:
                bin_obs_list[i] = (propensity_score >= bin_start) & (propensity_score <= bin_end)
            else:
                bin_obs_list[i] = (propensity_score > bin_start) & (propensity_score <= bin_end)
        
        df_plot = pd.DataFrame()

        smd = np.full((n_bins, len(covs)), np.nan)
        lower_bounds = np.full((n_bins, len(covs)), np.nan)
        upper_bounds = np.full((n_bins, len(covs)), np.nan)
        for j, column in enumerate(covs):
            # scaling on the whole sample
            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                s0_sample = self.df[column][control].std()
                s1_sample = self.df[column][treated].std()
                # using pooled variance from cobalt
                sd2 = (s0_sample**2 + s1_sample**2) / 2
                sd = np.sqrt(sd2)

            for i, bin_obs in enumerate(bin_obs_list):

                df_bin_treated = self.df[treated & bin_obs]
                df_bin_control = self.df[control & bin_obs]
                n0 = len(df_bin_control)
                n1 = len(df_bin_treated)

                if n0 == 0 and n1 == 0:
                    bin_smd = 0
                    bin_lower_bound = 0
                    bin_upper_bound = 0
                elif n1 == 0:
                    warnings.warn(f"No treated observations in bin {i}. Setting SMD to 5.")
                    bin_smd = 5
                    bin_lower_bound = 5
                    bin_upper_bound = 5
                elif n0 == 0:
                    warnings.warn(f"No control observations in bin {i}. Setting SMD to -5.")
                    bin_smd = -5
                    bin_lower_bound = -5
                    bin_upper_bound = -5
                else:
                    treated_mean = np.average(df_bin_treated[column])
                    control_mean = np.average(df_bin_control[column])
                    mean_diff = treated_mean - control_mean
                    bin_smd = mean_diff / sd

                    if n1 < 2 or n0 < 2:
                        warnings.warn(f"Bin {i} has less than 2 observations in one group. No CI computed.")
                        bin_lower_bound = np.nan
                        bin_upper_bound = np.nan
                    else:
                        s0 = df_bin_control[column].std()
                        s1 = df_bin_treated[column].std()

                        sd_diff = np.sqrt((s0**2 / n0) + (s1**2 / n1))

                        if n0 == 1 or n1 == 1:
                            df = 1
                        else:
                            df = ((s0**2 / n0) + (s1**2 / n1))**2 / (
                                ((s0**2 / n0)**2 / (n0 - 1)) + ((s1**2 / n1)**2 / (n1 - 1)))

                        t_critical = stats.t.ppf((1 + level) / 2, df)

                        bin_lower_bound = (mean_diff - t_critical * sd_diff) / sd
                        bin_upper_bound = (mean_diff + t_critical * sd_diff) / sd

                smd[i, j] = bin_smd
                lower_bounds[i, j] = bin_lower_bound
                upper_bounds[i, j] = bin_upper_bound
                
            df_plot_cov = pd.DataFrame({
                "bin_midpoints": bin_midpoints,
                "bin_size": bin_size,
                "smd": smd[:, j],
                "covariate": column,
                "lower_bound": lower_bounds[:, j],
                "upper_bound": upper_bounds[:, j],
                "Method": propensity_score.name
            })
            
            df_plot = pd.concat([df_plot, df_plot_cov], ignore_index=True)
        
        return df_plot

