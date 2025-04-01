import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import os
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_col, labs, scale_y_continuous, scale_fill_manual
from plotnine import geom_segment, theme_light, theme, element_line, element_text

def plot_dist(
    df: pd.DataFrame,
    directory: str,
    varying_col: str,
    fixed_conditions: dict,
    varying_filter: list = None,
    theta: float = 0,
    window_size: float = 0.5,
    filename_zero_fmt: str = '03d',
    procedure: str = None,  # New parameter to select a single procedure
    **kwargs
) -> None:
    """
    Create distribution plots with dynamic filtering and custom formatting.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data to plot
    directory : str
        Output directory for saving the plot PDFs
    varying_col : str
        Column name to use for creating subplot columns
    fixed_conditions : dict
        Dictionary of {column: value} pairs for fixed filters
    varying_filter : list, optional
        Specific values to include from the varying column
    theta : float, default 0
        True parameter value for vertical reference line
    window_size : float, default 0.5
        Half-width of the x-axis window around theta
    filename_zero_fmt : str, default '03d'
        Format string for zero-padding numeric values in filenames
    procedure : str, optional
        Specific procedure to plot (default None plots all procedures)
    **kwargs
        Additional keyword arguments passed to sns.displot

    Returns:
    --------
    None
    """
    # Create directory and set theme
    os.makedirs(directory, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    # Apply filters
    filter_conds = [df[col] == val for col, val in fixed_conditions.items()]
    df_filtered = df[np.logical_and.reduce(filter_conds)]
    
    if varying_filter is not None:
        if isinstance(varying_filter, list):
            df_filtered = df_filtered[df_filtered[varying_col].isin(varying_filter)]
        else:
            df_filtered = df_filtered[df_filtered[varying_col] == varying_filter]

    # Filter to selected procedure if specified
    if procedure is not None:
        df_filtered = df_filtered[df_filtered['procedure'] == procedure]

    # Determine row parameter for displot
    row_param = "procedure" if procedure is None else None

    # Create plot
    g = sns.displot(
        data=df_filtered,
        x="estimate",
        hue="Method",
        row=row_param,  # Dynamic row parameter
        col=varying_col,
        kind="kde",
        height=3,
        aspect=1.5,
        palette=sns.color_palette("colorblind", 6),
        legend=False,
        **kwargs
    )

    # Remove default facet titles
    g.set_titles(col_template="", row_template="")

    # Create combined headers for each subplot
    if procedure is None:
        # Original behavior with both row and column headers
        for (row_val, col_val), ax in g.axes_dict.items():
            header_text = f"{row_val} | {varying_col} = {col_val}"
            ax.set_title(header_text, fontsize=16, fontweight='bold', pad=7)
    else:
        # Single procedure: include procedure name in column headers
        for key, ax in g.axes_dict.items():
            col_val = key[0] if isinstance(key, tuple) else key
            header_text = f"{procedure} | {varying_col} = {col_val}"
            ax.set_title(header_text, fontsize=16, fontweight='bold', pad=7)

    # Add panel labels (A, B, C...)
    for i, ax in enumerate(g.axes.flat):
        ax.text(
            0.05, 0.95,
            f"{chr(65+i)}",
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top'
        )

    # Add reference lines and limits
    for ax in g.axes.flat:
        ax.axvline(x=theta, color='r', linestyle=':', linewidth=1.5)
        ax.set_xlim(theta - window_size, theta + window_size)

    # Set axis labels
    g.set_axis_labels("estimate", "Density", fontsize=16)

    # Generate filename components
    fmt_map = {
        'n_obs': ('n_obs', '05d'),
        'dim_x': ('dim_x', '03d'),
        'learner_m': ('learner_m', ''),
        'learner_g': ('learner_g', ''),
        'clipping_threshold': ('clip', '.2f'),
        'R2_d': ('R2d', '.2f'),
        'overlap': ('overlap', '.2f'),
        'share_treated': ('share_treated', '.2f')
    }

    filename_parts = []
    for col in fmt_map:
        name, fmt = fmt_map[col]
        if col in fixed_conditions:
            val = fixed_conditions[col]
            if isinstance(val, (int, float)):
                filename_parts.append(f"{name}_{val:{fmt}}")
            else:
                filename_parts.append(f"{name}_{val}")
        elif col == varying_col:
            if fmt == '':
                filename_parts.append(f"{name}_0{filename_zero_fmt}")
            else:
                filename_parts.append(f"{name}_0{fmt}")

    # Add procedure to filename if specified
    if procedure is not None:
        filename_parts.append(f"procedure_{procedure}")

    # Save figure
    plt.savefig(
        f"{directory}{'_'.join(filename_parts)}.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_calibration_metrics(
    directory: str,
    data: pd.DataFrame,
    n_obs_list: list,
    dim_x: int,
    learner_g: str,
    clipping_thresholds: list,
    learner_dict_m: dict,
    R2_d: float,
    overlap: float,
    share_treated: float,
    metrics_config: list,
    palette_colors: list,
    panel_labels: bool = True,
    methods: list = None
) -> None:
    """
    Plot calibration metrics with dynamic subplot configuration and panel labels.

    Parameters:
    -----------
    directory : str
        Output directory for saving figures
    data : pd.DataFrame
        Input dataframe containing calibration metrics
    n_obs_list : list
        List of sample sizes to include in the plot
    methods : list
        List of algorithms to be included
    dim_x : int
        Dimension of covariates (X)
    learner_g : str
        Name of the outcome model learner
    clipping_thresholds : list
        List of clipping thresholds to process
    learner_dict_m : dict
        Dictionary of propensity score learners to evaluate
    R2_d : float
        R-squared value for treatment assignment model
    overlap : float
        Overlap parameter value
    share_treated : float
        Proportion of treated units
    metrics_config : list
        List of metric configuration dictionaries containing:
        - column: DataFrame column name for the metric
        - ylabel: Y-axis label for the plot
    palette_colors : list
        List of color codes for different methods
    panel_labels : bool, optional
        Whether to add (A), (B), etc. panel labels, default True

    Example:
    --------
    METRICS_CONFIG = [
        {'column': 'ECE_quantile_b10', 'ylabel': 'Quantile ECE (b=10)'}
    ]
    """
    os.makedirs(directory, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    for clipping_threshold in clipping_thresholds:
        for learner_m in learner_dict_m:
            # Data filtering
            df_filtered = data[
                (data["learner_m"] == learner_m) &
                (data["dim_x"] == dim_x) &
                (data["R2_d"] == R2_d) &
                (data["overlap"] == overlap) &
                (data["share_treated"] == share_treated)
            ]
            
            
            if methods:
                methods=methods
            else:
                methods = [f'Alg-1-Uncalib-Clipped_{clipping_threshold}', 'Alg-1-Uncalib-Unclipped',
                       'Alg-2-nested-cf-Iso-Unclipped', 'Alg-3-cf-Iso-Unclipped',
                       'Alg-4-single-split-Iso-Unclipped', 'Alg-5-full-sample-Iso-Unclipped']
            
            df_filtered = df_filtered[
                df_filtered["n_obs"].isin(n_obs_list) &
                df_filtered["Method_Clip"].isin(methods)
            ]

            # Dynamic subplot grid based on number of metrics
            n_metrics = len(metrics_config)
            n_cols = min(n_metrics, 2)  # Max 2 columns
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                   figsize=(4*n_cols, 3.5*n_rows),
                                   squeeze=False)
            axes = axes.flatten()

            # Plot each metric
            for idx, metric in enumerate(metrics_config):
                ax = axes[idx]
                sns.lineplot(
                    data=df_filtered,
                    x="n_obs",
                    y=metric['column'],
                    hue="Method_Clip",
                    style="Method_Clip",
                    ax=ax,
                    errorbar=None,
                    dashes=False,
                    markers=['X','o','d','^','v','>'],
                    markersize=8,
                    palette=sns.color_palette(palette_colors, 6),
                    legend=None
                )
                
                if panel_labels:
                    ax.text(0.1, 0.95, 
                           f'{chr(65+idx)}',
                           transform=ax.transAxes,
                           fontsize=14,
                           fontweight='bold',
                           va='top')
                
                ax.set_ylabel(metric['ylabel'])
                ax.set_xlabel('Sample Size' if idx >= n_metrics-n_cols else '')

            # Hide unused axes
            for idx in range(n_metrics, len(axes)):
                axes[idx].axis('off')

            # Add legend and adjust layout
            #if n_metrics > 0:
            #    handles, labels = axes[0].get_legend_handles_labels()
            #    fig.legend(handles, labels, 
            #             loc='lower center', 
            #             ncol=3,
            #             bbox_to_anchor=(0.5, -0.05 if n_rows > 1 else -0.15))
            
            plt.tight_layout()
            
            # Filename generation
            filename = (
                f"n_obs_00000_dim_x_{dim_x:03d}_learner_m_{learner_m}_"
                f"learner_g_{learner_g}_clip_{clipping_threshold:.2f}_"
                f"R2d_{R2_d:.2f}_overlap_{overlap:.2f}_"
                f"share_treated_{share_treated:.2f}.pdf"
            )
            
            plt.savefig(f"{directory}{filename}", bbox_inches='tight')
            plt.clf()
            plt.close()


def plot_ate_metrics(
    directory_base: str,
    data: pd.DataFrame,
    procedures: list,
    metrics: list,
    methods_config: dict,
    learner_m_list: list,
    clipping_threshold: float,
    n_obs_list: list,
    dim_x: int,
    learner_g: str,
    R2_d: float,
    overlap: float,
    share_treated: float,
    palette_colors: list,
    style_kwargs: dict = None,
    fig_size: tuple = None,
    add_headers_func: callable = None
):     
    """Plot ATE metrics grid with procedures as columns and metrics as rows.
        Parameters:
        directory_base (str): Base directory path for saving plots. Subdirectories will be created
            for each combination of procedures.
        data (pd.DataFrame): DataFrame containing simulation results data. Expected columns include
            'learner_g', 'learner_m', 'R2_d', 'dim_x', 'overlap', 'share_treated', 'n_obs',
            'Method_Clip', 'procedure', and metric columns.
        procedures (list[str]): List of procedure names to plot as columns (e.g., ['IPW', 'IRM']).
        metrics (list[str]): List of metric names to plot as rows (e.g., ['RMSE', 'Bias']).
        methods_config (dict): Configuration dictionary with:
            - 'methods': List of method names to include from data['Method_Clip']
            - 'labels': List of legend labels corresponding to each method
        learner_m_list (list[str]): List of mediator learner models to generate plots for
            (e.g., ['LGBM', 'Linear']).
        clipping_threshold (float): Threshold value used for clipping propensity scores
            (appears in output filename).
        n_obs_list (list[int]): Sample sizes to include in the x-axis (e.g., [100, 200, 500]).
        dim_x (int): Dimension of covariates used in the simulation (affects data filtering).
        learner_g (str): Name of outcome model learner used (e.g., 'LGBM').
        R2_d (float): R-squared value of the treatment assignment mechanism in the simulation.
        overlap (float): Overlap parameter value (0-1) controlling propensity score distribution.
        share_treated (float): Proportion of treated units in the simulation (0-1).
        palette_colors (list[str]): Color palette for methods (hex codes or named colors).
        style_kwargs (dict, optional): Custom style parameters for plots. Defaults to:
            {'title_size': 20, 'label_size': 16, 'tick_size': 15, 
             'panel_label_size': 20, 'marker_size': 12}
        fig_size (tuple[int], optional): Custom figure dimensions (width, height) in inches.
            Defaults to (5.5*num_columns, 5.5*num_rows).

    Returns:
    None: Saves plots to disk without returning any value."""

    sns.set_theme(style="whitegrid", context="paper")
    style = style_kwargs or {
        'title_size': 20,
        'label_size': 16,
        'tick_size': 15,
        'panel_label_size': 20,
        'marker_size': 12
    }

    markers = ['X', 'o', 'd', '^', 'v', '>']
    
    for learner_m in learner_m_list:
        directory = f"{directory_base}{'_'.join(procedures)}/"
        os.makedirs(directory, exist_ok=True)

        df_filtered = data[
            (data["learner_g"] == learner_g) &
            (data["learner_m"] == learner_m) &
            (data["R2_d"] == R2_d) &
            (data["dim_x"] == dim_x) &
            (data["overlap"] == overlap) &
            (data["share_treated"] == share_treated)
        ].copy()

        df_filtered = df_filtered[df_filtered["n_obs"].isin(n_obs_list)]
        df_filtered = df_filtered[df_filtered["Method_Clip"].isin(methods_config['methods'])]

        n_rows = len(metrics)
        n_cols = len(procedures)
        fig_size = fig_size or (5.5*n_cols, 5.5*n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=fig_size,
                                sharex=True,
                                squeeze=False)

        # Plot each metric-procedure combination
        for row_idx, metric in enumerate(metrics):
            for col_idx, procedure in enumerate(procedures):
                ax = axes[row_idx, col_idx]
                df_proc = df_filtered[df_filtered["procedure"] == procedure]

                if not df_proc.empty:
                    sns.lineplot(
                        data=df_proc,
                        x="n_obs",
                        y=metric,
                        hue="Method_Clip",
                        style="Method_Clip",
                        ax=ax,
                        errorbar=None,
                        dashes=False,
                        markers=markers[:len(methods_config['methods'])],
                        markersize=style['marker_size'],
                        palette=sns.color_palette(palette_colors),
                        legend=None, # bool(row_idx == 0 and col_idx == 0)  for Single legend
                    )

                # Panel labels and axis formatting
                panel_number = row_idx * n_cols + col_idx
                ax.text(0.07, 0.9, f'{chr(65 + panel_number)}',
                        transform=ax.transAxes,
                        fontsize=style['panel_label_size'],
                        fontweight='bold')

                if row_idx == n_rows - 1:
                    ax.set_xlabel('Sample Size', 
                                fontsize=style['label_size'],
                                fontweight='medium')
                ax.set_ylabel(metric, fontsize=style['label_size'], fontweight='medium')
                ax.tick_params(axis='both', labelsize=style['tick_size'])
        # Add headers if specified
        if add_headers_func:
            add_headers_func(fig, 
                            col_headers=[f'Model = {p}' for p in procedures])

        plt.tight_layout()
        filename = (
            f"n_obs_00000_dim_x_{dim_x:03d}_learner_m_{learner_m}_"
            f"learner_g_{learner_g}_clip_{clipping_threshold:.2f}_"
            f"R2d_{R2_d:.2f}_overlap_{overlap:.2f}_"
            f"share_treated_{share_treated:.2f}.pdf"
        )
        plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
        plt.close()
        

def plot_overlap_ratio(data, ps_col: str, treatment_col: str, metric: str, clipping_threshold: float=1e-12, n_bins=50, **kwargs):
    valid_metrics = ['ratio', 'count']
    assert metric in valid_metrics, "Invalid metric"

    y_col = "ratio" if metric == "ratio" else "count"
    y_label = "Ratio" if metric == "ratio" else "Count"

    df_propensity = data.assign(
        m_hat=lambda d: d[ps_col].clip(clipping_threshold, 1.0 - clipping_threshold),
    )
    treatment_indicator = df_propensity[treatment_col] == 1

    bin_edges = np.linspace(df_propensity["m_hat"].min(), df_propensity["m_hat"].max(), n_bins + 1)

    bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    bin_size = [(bin_edges[i + 1] - bin_edges[i]) for i in range(n_bins)]

    count_treated = [np.nan] * n_bins
    neg_count_control = [np.nan] * n_bins

    ratio_treated = [np.nan] * n_bins
    neg_ratio_control = [np.nan] * n_bins

    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        bin_obs = (df_propensity["m_hat"] >= bin_start) & (df_propensity["m_hat"] <= bin_end) if i == 0 else (df_propensity["m_hat"] > bin_start) & (df_propensity["m_hat"] <= bin_end)

        bin_treated = bin_obs & treatment_indicator
        bin_control = bin_obs & ~treatment_indicator

        bin_n_treated = bin_treated.sum()
        bin_n_control = bin_control.sum()
        bin_n_obs = bin_n_treated + bin_n_control

        count_treated[i] = bin_n_treated
        neg_count_control[i] = -1 * bin_n_control
        ratio_treated[i] = bin_n_treated / bin_n_obs
        neg_ratio_control[i] = -1 * bin_n_control / bin_n_obs

    df_plot = pd.DataFrame({
        "bin_midpoint": bin_midpoints * 2,
        "bin_size": bin_size * 2,
        "count": count_treated + neg_count_control,
        "ratio": ratio_treated + neg_ratio_control,
        "category": ["Treated"] * n_bins + ["Control"] * n_bins,
    })

    text_kwargs = kwargs.get('text_kwargs', {}) 
    text_kwargs.setdefault("fontweight", "bold")

    # Plot using Matplotlib
    plt.bar(df_plot['bin_midpoint'][df_plot['category'] == "Treated"], df_plot[y_col][df_plot['category'] == "Treated"], 
            width=df_plot['bin_size'][df_plot['category'] == "Treated"], 
            color='#87CEEB', edgecolor='black', alpha=0.7, label='Treated')
    plt.bar(df_plot['bin_midpoint'][df_plot['category'] == "Control"], df_plot[y_col][df_plot['category'] == "Control"], 
            width=df_plot['bin_size'][df_plot['category'] == "Control"], 
            color='#FA8072', edgecolor='black', alpha=0.7, label='Untreated')
    
    plt.axhline(y=0, color='black', linewidth=1)

    # Add the dashed lines 
    plt.plot([0, 1], [0, 1], linestyle='dashed', color='black', linewidth=0.5) 
    plt.plot([0, 1], [-1, 0], linestyle='dashed', color='black', linewidth=0.5)
    
    plt.xlabel('Propensity Score', **text_kwargs)
    plt.ylabel(y_label, **text_kwargs)
    plt.title(kwargs.get('title', ''), **text_kwargs)
    plt.yticks(ticks=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], labels=['1', '0.8', '0.6', '0.4', '0.2', '0', '0.2', '0.4', '0.6', '0.8', '1'])


def plot_calibration_comparison(
    directory: str,
    df: pd.DataFrame,
    methods: list,
    n_obs: int,
    dim_x: int,
    clipping_threshold: float,
    R2_d: float,
    overlap: float,
    share_treated: float,
    covariate: str = "x_1",
    scatter_color: str = '#023eff',
    ps_color: str = '#662506',
    true_score_color: str = '#018571',
    alpha_scatter: float = 0.4,
    alpha_line: float = 0.5,
    panel_label_fontsize: int = 14
):
    """
    Plot calibration comparisons across different learners with flexible covariate axis.
    
    Parameters:
    directory (str): Output directory for saving figures
    df (pd.DataFrame): Input DataFrame containing calibration data
    methods (list): List of methods to include from 'Method_Clip' column
    n_obs (int): Number of observations for filename
    dim_x (int): Dimension of covariates for filename
    clipping_threshold (float): Clipping threshold value for filename
    R2_d (float): R² value for filename
    overlap (float): Overlap parameter for filename
    share_treated (float): Share treated parameter for filename
    covariate (str): Covariate to plot on x-axis (default: 'x_1')
    scatter_color (str): Color for treatment indicator points
    ps_color (str): Color for predicted propensity scores
    true_score_color (str): Color for true propensity scores
    alpha_scatter (float): Transparency for scatter points
    alpha_line (float): Transparency for line plots
    panel_label_fontsize (int): Font size for panel labels (A, B, C...)
    """
    plt.rcParams.update({
        'font.weight': 'normal',
        "text.usetex": True,
        "font.family": plt.rcParamsDefault["font.family"]
    })
    
    os.makedirs(directory, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    # Filter and prepare data
    df_filtered = df[df["Method_Clip"].isin(methods)].copy()
    df_filtered['Learner'] = df_filtered['Learner'].str.replace('Logistic', 'Logit')

    # Create FacetGrid
    g = sns.FacetGrid(data=df_filtered, col='Learner', sharex=False, margin_titles=True)
    
    # Plot components using specified covariate
    g.map(sns.scatterplot, covariate, "treatment", 
          color=scatter_color, alpha=alpha_scatter)
    g.map(sns.lineplot, covariate, "ps", 
          color=ps_color, alpha=1)
    g.map(sns.lineplot, covariate, "m_0", 
          color=true_score_color, alpha=alpha_line)
    
    # Add panel labels (A, B, C, ...)
    for i, ax in enumerate(g.axes.flat):
        ax.text(0.05, 0.93, 
                rf'\textbf{{{chr(65 + i)}}}',
                transform=ax.transAxes,
                fontsize=panel_label_fontsize,
                fontweight='bold',
                va='top')

    # Set titles and labels
    g.set_titles(col_template=r'\textbf{{m = {col_name}}}', 
                fontweight='bold', size=12)
    g.set_axis_labels(covariate.upper(), "m")  # Use covariate name for x-axis

    # Generate filename
    filename = (
        f'n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_'
        f'covariate_{covariate}_clip_{clipping_threshold:.2f}_'
        f'R2d_{R2_d:.2f}_overlap_{overlap:.2f}_'
        f'share_treated_{share_treated:.2f}.pdf'
    )
    
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
    plt.close()


def plot_mirrored_propensity_histogram(
    treatment: np.ndarray,
    m_0: np.ndarray,
    directory: str,
    n_obs: int,
    dim_x: int,
    clipping_threshold: float,
    R2_d: float,
    overlap: float,
    share_treated: float,
    panel_label: str = None,
    bins: int = 50,
    figsize: tuple = (6, 6)
):
    """
    Create mirrored histogram of propensity scores by treatment status
    
    Parameters:
    treatment -- Array of treatment indicators (1/0)
    m_0 -- Array of propensity scores
    directory -- Output directory for saving
    n_obs -- Number of observations for filename
    dim_x -- Covariate dimension for filename
    clipping_threshold -- Clipping threshold for filename
    R2_d -- R² value for filename
    overlap -- Overlap parameter for filename
    share_treated -- Share treated parameter for filename
    panel_label -- Panel label text (e.g., "(A)") 
    bins -- Number of histogram bins
    figsize -- Figure size
    """
    
    os.makedirs(directory, exist_ok=True)
    
    # Separate propensity scores
    treated = m_0[treatment == 1]
    untreated = m_0[treatment == 0]
    bin_edges = np.linspace(0, 1, bins + 1)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot treated (positive)
    ax.hist(treated, bins=bin_edges, alpha=0.7, density=True,
            color='#018571', edgecolor='black')
    
    # Plot untreated (negative)
    hist_untreated, _ = np.histogram(untreated, bins=bin_edges, density=True)
    ax.bar(bin_edges[:-1], -hist_untreated, width=np.diff(bin_edges),
           edgecolor='black', alpha=0.7, color='#662506')

    # Add reference line and labels
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distribution by Treatment Status')
    
    # Format y-axis with absolute values
    y_max = max(np.max(np.histogram(treated, bins=bin_edges, density=True)[0]),
                np.max(hist_untreated))
    y_ticks = np.linspace(-y_max, y_max, 7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{abs(y):.2f}" for y in y_ticks])

    # Add panel label if specified
    if panel_label:
        ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Save and close
    filename = (f'n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_'
                f'learner_m_000_learner_g_000_clip_{clipping_threshold:.2f}_'
                f'R2d_{R2_d:.2f}_overlap_{overlap:.2f}_'
                f'share_treated_{share_treated:.2f}.pdf')
    
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
    plt.close()

def plot_ps_treatment_comparison(
    directory,
    df,
    methods,
    n_obs,
    dim_x,
    clipping_threshold,
    R2_d,
    overlap,
    share_treated,
    plot_fn,
    ps_col='ps',
    treatment_col='treatment',
    metric='ratio',
    panel_label_fontsize=14,
    row='Method_Clip',
    col='Learner'
):
    """
    Plot propensity score treatment comparisons with panel labels.
    
    Parameters:
    directory (str): Output directory for saving figures
    df (DataFrame): Input data
    methods (list): List of methods to include
    n_obs (int): Number of observations
    dim_x (int): Dimension of X
    clipping_threshold (float): Clipping threshold value
    R2_d (float): R-squared value
    overlap (float): Overlap parameter
    share_treated (float): Share treated parameter
    plot_fn (function): Custom plotting function
    ps_col (str): Propensity score column name
    treatment_col (str): Treatment column name
    metric (str): Metric name
    panel_label_fontsize (int): Font size for panel labels
    row (str): Row facet column name,
    col (str): Column facet column name
    """
    os.makedirs(directory, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    df_filtered = df[df["Method_Clip"].isin(methods)]

    g = sns.FacetGrid(
        data=df_filtered,
        row=row,
        col=col,
        sharex=False,
        sharey='row',
        margin_titles=True
    )
    
    g.map_dataframe(
        plot_fn,
        ps_col=ps_col,
        treatment_col=treatment_col,
        metric=metric
    )

    # Add panel labels (A, B, C, ...)
    for i, ax in enumerate(g.axes.flat):
        ax.text(
            0.05, 0.95,
            f'{chr(65 + i)}',  # 65 = 'A' in ASCII
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            fontweight='bold',
            va='top'
        )

    g.set_titles(
        col_template='{col_name}',
        row_template='{row_name}',
        fontweight='bold',
        size=12
    )

    filename = (
        f'n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_'
        f'learner_m_000_learner_g_000_clip_{clipping_threshold:.2f}_'
        f'R2d_{R2_d:.2f}_overlap_{overlap:.2f}_'
        f'share_treated_{share_treated:.2f}.pdf'
    )

    plt.savefig(
        os.path.join(directory, filename),
        bbox_inches='tight'
    )
    plt.close()    

def plot_overlap_comparison(
    df: pd.DataFrame,
    directory: str,
    ps_col: str,
    treatment_col: str,
    metric: str,
    subgroups: pd.DataFrame = None,
    n_obs: int = None,
    dim_x: int = None,
    clipping_threshold: float = 1e-12,
    R2_d: float = None,
    overlap: float = None,
    share_treated: float = None,
    n_bins: int = 50,
    panel_labels: bool = True,
    style_kwargs: dict = None
):
    """
    Create standardized overlap comparison plots with panel labels
    
    Parameters:
    df -- DataFrame containing treatment and propensity score data
    directory -- Output directory for saving figures
    ps_col -- Name of propensity score column
    treatment_col -- Name of treatment indicator column
    metric -- Metric to plot ('ratio' or 'count')
    subgroups -- DataFrame of boolean masks for subgroup analysis
    n_obs -- Number of observations (for filename)
    dim_x -- Covariate dimension (for filename)
    clipping_threshold -- PS clipping threshold (for filename)
    R2_d -- R² value (for filename)
    overlap -- Overlap parameter (for filename)
    share_treated -- Share treated parameter (for filename)
    n_bins -- Number of bins for PS stratification
    panel_labels -- Whether to add panel labels (A, B, C...)
    style_kwargs -- Dictionary of plot styling parameters
    """
    
    # Set default style parameters
    default_style = {
        'colors': {'Treated': 'skyblue', 'Control': 'salmon'},
        'dash_color': 'black',
        'title_size': 12,
        'panel_label_fontsize': 14,
        'panel_label_pos': (0.05, 0.95)
    }
    style = {**default_style, **(style_kwargs or {})}

    os.makedirs(directory, exist_ok=True)

    def create_plot(data, title=""):
        # Data processing from original plot_overlap_ratio
        df_propensity = data.assign(
            m_hat=lambda d: d[ps_col].clip(clipping_threshold, 1.0 - clipping_threshold)
        )
        treatment_indicator = df_propensity[treatment_col] == 1

        # Define y-axis label based on metric
        y_label = "Ratio" if metric == "ratio" else "Count"

        # Bin calculation
        bin_edges = np.linspace(df_propensity["m_hat"].min(), 
                            df_propensity["m_hat"].max(), 
                            n_bins + 1)

        bin_midpoints = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(n_bins)]
        bin_size = [bin_edges[i+1] - bin_edges[i] for i in range(n_bins)]

        count_treated = [np.nan] * n_bins
        neg_count_control = [np.nan] * n_bins
        ratio_treated = [np.nan] * n_bins
        neg_ratio_control = [np.nan] * n_bins

        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            bin_obs = (df_propensity["m_hat"] >= bin_start) & (df_propensity["m_hat"] <= bin_end) if i == 0 else \
                    (df_propensity["m_hat"] > bin_start) & (df_propensity["m_hat"] <= bin_end)

            bin_treated = bin_obs & treatment_indicator
            bin_control = bin_obs & ~treatment_indicator

            bin_n_treated = bin_treated.sum()
            bin_n_control = bin_control.sum()
            bin_n_obs = bin_n_treated + bin_n_control

            count_treated[i] = bin_n_treated
            neg_count_control[i] = -bin_n_control
            ratio_treated[i] = bin_n_treated/bin_n_obs if bin_n_obs > 0 else np.nan
            neg_ratio_control[i] = -bin_n_control/bin_n_obs if bin_n_obs > 0 else np.nan

        y_col = "ratio" if metric == "ratio" else "count"
        values = ratio_treated + neg_ratio_control if metric == "ratio" else count_treated + neg_count_control

        df_plot = pd.DataFrame({
            "bin_midpoint": bin_midpoints * 2,
            "bin_size": bin_size * 2,
            y_col: values,
            "category": ["Treated"] * n_bins + ["Control"] * n_bins,
        })

        plot = (
            ggplot(df_plot, aes(x="bin_midpoint", y=y_col, fill="category")) +
            geom_col(aes(width="bin_size"), color="black", show_legend=False) +
            labs(x="Propensity Score", y=y_label, title=title) +
            scale_y_continuous(labels=lambda l: [abs(i) for i in l]) +
            scale_fill_manual(values=style['colors']) +
            geom_segment(aes(x=0, xend=1, y=0, yend=1), 
                        linetype="dashed", 
                        color=style['dash_color'],
                        size=0.5) +
            geom_segment(aes(x=0, xend=1, y=-1, yend=0), 
                        linetype="dashed", 
                        color=style['dash_color'],
                        size=0.5) +
            theme_light() +
            theme(
                panel_grid_major=element_line(color="#DCDCDC"),
                panel_grid_minor=element_line(color="#E6E6E6"),
                plot_title=element_text(size=style['title_size'], 
                                    face="bold")
            )
        )
        
        if panel_labels:
            # Add panel label using plotnine's annotation system
            label_df = pd.DataFrame({
                'x': [0.05],  # Normalized coordinates
                'y': [0.95],  # Normalized coordinates
                'label': [f'{chr(65 + panel_counter[0])}']
            })
            
            plot += geom_text(
                data=label_df,
                mapping=aes(x='x', y='y', label='label'),
                inherit_aes=False,
                format_string='',
                size=style['panel_label_fontsize'],
                fontweight='bold',
                ha='left',
                va='top',
                # Convert to normalized plot coordinates
                nudge_x=0.05,
                nudge_y=-0.05
            )
            panel_counter[0] += 1
            
        return plot

    # Handle subgroups or full sample
    panel_counter = [0]  # Mutable counter for panel labels
    if subgroups is not None:
        for col in subgroups.columns:
            subgroup = subgroups[col]
            share = subgroup.mean()
            plot = create_plot(
                df[subgroup], 
                title=f"Median Subgroup: {col} (Top {share*100:.1f}%)" 
            )
            filename = (
            f"n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_"
            f"clip_{clipping_threshold:.2f}_"
            f"R2d_{R2_d:.2f}_overlap_{overlap:.2f}_"
            f"share_treated_{share_treated:.2f}_{col}.pdf"
            )
            plot.save(os.path.join(directory, filename), dpi=300)
    else:
        plot = create_plot(df, title=None)
        filename = (
        f"n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_"
        f"clip_{clipping_threshold:.2f}_"
        f"R2d_{R2_d:.2f}_overlap_{overlap:.2f}_"
        f"share_treated_{share_treated:.2f}_full_sample.pdf"
        )
        plot.save(os.path.join(directory, filename), dpi=300)
    

def plot_overlap_ratio_detailed(
    data: pd.DataFrame,
    ps_col: str,
    treatment_col: str,
    metric: str,
    directory: str,
    n_obs: int,
    dim_x: int,
    learner_m: str,
    learner_g: str,
    clipping_threshold: float,
    R2_d: float,
    overlap: float,
    share_treated: float,
    panel_labels: bool = True,
    n_bins: int = 50,
    **kwargs
):
    """
    Function to plot the propensity score ratio over different ranges to assess overlap.
    
    Args:
    - data: DataFrame containing propensity score and treatment column
    - ps_col: Column name for propensity scores
    - treatment_col: Column name for treatment indicator
    - metric: Metric to plot (only 'ratio')
    - directory: Output directory for saving plots
    - n_obs: Number of observations 
    - dim_x: Covariate dimension 
    - learner_m: Learner for m 
    - learner_g: Learner for g 
    - clipping_threshold: Clipping threshold for propensity scores
    - R2_d: R² value for filename
    - overlap: Overlap parameter 
    - share_treated: Share treated parameter 
    - panel_labels: Whether to add panel labels (A), (B)
    - n_bins: Number of bins in the histogram
    """
    
    os.makedirs(directory, exist_ok=True)
    
    valid_metrics = ['ratio']
    assert metric in valid_metrics, "Invalid metric"

    y_col = "ratio"
    y_label = "Ratio"
    treatment_indicator = data[treatment_col] == 1
    n_treated = treatment_indicator.sum()
    n_control = len(treatment_indicator) - n_treated

    col_names = {
        "treated": f"Treated (n={n_treated})",
        "control": f"Control (n={n_control})",
    }

    data = data.assign(
        _m_hat=lambda d: d[ps_col].clip(clipping_threshold, 1.0 - clipping_threshold),
    )

    bin_edges = np.linspace(data["_m_hat"].min(),
                            data["_m_hat"].max(), n_bins + 1)

    bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    bin_size = [(bin_edges[i + 1] - bin_edges[i]) for i in range(n_bins)]

    count_treated = [np.nan] * n_bins
    count_control = [np.nan] * n_bins
    ratio_treated = [np.nan] * n_bins
    ratio_control = [np.nan] * n_bins

    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == 0:
            bin_obs = (data["_m_hat"] >= bin_start) & (data["_m_hat"] <= bin_end)
        else:
            bin_obs = (data["_m_hat"] > bin_start) & (data["_m_hat"] <= bin_end)

        bin_treated = bin_obs & treatment_indicator
        bin_control = bin_obs & ~treatment_indicator

        bin_n_treated = bin_treated.sum()
        bin_n_control = bin_control.sum()
        bin_n_obs = bin_n_treated + bin_n_control

        count_treated[i] = bin_n_treated
        count_control[i] = bin_n_control
        ratio_treated[i] = bin_n_treated / bin_n_obs
        ratio_control[i] = bin_n_control / bin_n_obs

    df_plot = pd.DataFrame({
        "bin_midpoint": bin_midpoints * 2,
        "bin_size": bin_size * 2,
        "count": count_treated + count_control,
        "ratio": ratio_treated + ratio_control,
        "category": [col_names["treated"]] * n_bins + [col_names["control"]] * n_bins,
    })

    df_plot["ratio"] = -1.0 * df_plot["ratio"]
    df_plot["count"] = -1.0 * df_plot["count"]

    df_segments = pd.DataFrame({
        "x_start": [0, 0],
        "x_end": [1, 1],
        "y_start": [-1, 0],
        "y_end": [0, -1],
        "category": [col_names["control"], col_names["treated"]]
    })

    df_density = pd.DataFrame({
        "m_hat": data["_m_hat"],
        "category": [col_names["treated"] if treated else col_names["control"] for treated in treatment_indicator],
        "treatment_indicator": treatment_indicator,
    })

    chart = (
    ggplot(df_plot, aes(x="bin_midpoint", y=y_col, fill="category")) +
    geom_col(aes(width="bin_size"), color="black") +
    geom_density(data=df_density,
                    mapping=aes(x="m_hat", y="..scaled..")) +
    facet_wrap("~category") +
    labs(title="Overlap", x="Propensity Score", y=y_label) +
    scale_y_continuous(labels=lambda l: [abs(i) for i in l]) +
    scale_fill_manual(values={col_names["treated"]: "skyblue", col_names["control"]: "salmon"}) +
    geom_segment(data=df_segments,
                    mapping=aes(x="x_start", xend="x_end", y="y_start", yend="y_end"),
                    linetype="dashed", color="black", size=0.5, inherit_aes=False) +
    theme_bw()
    )
    
    # Add panel labels
    if panel_labels:
        label_df = pd.DataFrame({
            'category': [col_names["control"], col_names["treated"]],
            'label': ['A', 'B'],
            'x': [0.01, 0.01],  # X position in normalized coordinates
            'y': [0.99, 0.99]   # Y position in normalized coordinates
        })

        chart += geom_text(
            data=label_df,
            mapping=aes(x='x', y='y', label='label'),
            inherit_aes=False,
            format_string='',
            size=14,
            fontweight='bold',
            ha='left',
            va='top',
            nudge_x=0,
            nudge_y=0
        )

    # Generate filename
    filename = (
        f"n_obs_{n_obs:05d}_dim_x_{dim_x:03d}_"
        f"learner_m_{learner_m}_learner_g_{learner_g}_"
        f"clip_{clipping_threshold:.2f}_R2d_{R2_d:.2f}_"
        f"overlap_{overlap:.2f}_share_treated_{share_treated:.2f}.pdf"
    )
    
    # Save plot
    chart.save(os.path.join(directory, filename), dpi=300)
    
    return chart


def evaluate_estimation(ate: np.ndarray, theta: float, level: float = 0.9) -> dict:
    """Calculate estimation metrics for ATE results.
    
    Args:
        ate: Array of ATE estimates
        theta: True treatment effect value
        level: Confidence level for variance calculation
        
    Returns:
        Dictionary of evaluation metrics
    """
    ate_true = np.full_like(ate, theta)  # Correct: Match array shape
    bias = ate - ate_true
    
    return {
        'Mean Bias': np.nanmean(bias),
        'RMSE': np.sqrt(np.nanmean(bias ** 2)),
        'Std. dev.': np.nanstd(ate),
        'MAE': np.nanmean(np.abs(bias)),
    }

def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Ensure fontweight is set to bold by default
    text_kwargs.setdefault("fontweight", "bold")
    text_kwargs.setdefault("fontsize", "20")


    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

