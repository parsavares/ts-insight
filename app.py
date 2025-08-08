# Author: Parsa Vares
# Affiliation: LIST, 2025
#----------------------------
# TS-Insight:
# visualization tool for understanding, debugging, and explaining Thompson Sampling algorithms
#----------------------------
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches    # For legend
from matplotlib.lines import Line2D       # For legend
import matplotlib.gridspec as gridspec
import os
import torch                             # For loading .pt file
import json                              # For loading config_run.json
from pathlib import Path                # For easier path manipulation
import logging                           # For messages from the plotting function
from datetime import datetime           # For unique filenames if saving (used for download)
import tempfile                          # For handling uploaded files
from io import BytesIO                   # For plot download
# meaning used: query = sampling
# --- Import for Beta distribution quantiles and CDF/PPF ---
from scipy.stats import beta as beta_distribution

# --- MODIFIED: Use matplotlib.colormaps for modern Matplotlib ---
from matplotlib import colormaps

# --- Configure Matplotlib (from vis.py) ---
plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 14,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5,
    "figure.titlesize": 16, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "lines.linewidth": 1.5,
    "lines.markersize": 5, "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "grid.linestyle": ":",
})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PLOT - %(levelname)s - %(message)s')

# --- Color Palettes (from vis.py) ---
viridis_cmap = colormaps.get_cmap('viridis')
greys_cmap   = colormaps.get_cmap('Greys')
PLOT_COLORS = {
    'success_reward': "#0072B2", 'failure_reward': "#D55E00",
    'alpha_line': "#0072B2", 'beta_line': "#D55E00",
    'posterior_draw_marker': "#009E73",
    'explore': "#E0E0E0",
    'vline_color_tuple': greys_cmap(0.99),
    'alpha': "#0072B2", 'beta': "#D55E00",
    'other_sample': "#009E73",
}
PLOT_COLORS['vline'] = PLOT_COLORS['vline_color_tuple'][:3]

# --- Helper: compute_hdr_bounds (from vis.py, with minor robustness additions) ---
def compute_hdr_bounds(alpha_arr, beta_arr, rho=0.50, tol=1e-6):
    N = len(alpha_arr)
    a_vals = np.zeros(N)
    b_vals = np.zeros(N)
    for i in range(N):
        α, β = alpha_arr[i], beta_arr[i]
        if α <= 0 or β <= 0 : # Beta distribution parameters must be > 0
             # Handle cases like prior (0,0) or (1,0) etc.
            if α <= 0 and β <= 0: # e.g. (0,0) prior
                mu = 0.5
            elif α > 0 and β <=0 : # effectively point mass at 1
                mu = 1.0
            elif α <=0 and β > 0: # effectively point mass at 0
                mu = 0.0
            else: # should not happen if one is positive
                mu = 0.5

            a_vals[i] = mu # Or np.nan / specific handling
            b_vals[i] = mu # Or np.nan
            # logging.debug(f"HDR bounds for ill-defined Beta({α},{β}) set to ({a_vals[i]},{b_vals[i]})")
            continue

        mu = α / (α + β)
        lo, hi = 0.0, min(mu, 1.0 - mu)

        if hi <= lo + tol : # Covers mu=0, mu=1, or very sharp distributions where delta is tiny
            try:
                lower_bound = beta_distribution.ppf((1.0 - rho) / 2.0, α, β)
                upper_bound = beta_distribution.ppf(1.0 - (1.0 - rho) / 2.0, α, β)
                a_vals[i] = mu
                b_vals[i] = mu
            except ValueError: 
                a_vals[i] = mu
                b_vals[i] = mu
            continue

        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            try:
                lower_cdf = beta_distribution.cdf(mu - mid, α, β)
                upper_cdf = beta_distribution.cdf(mu + mid, α, β)
            except ValueError: 
                a_vals[i], b_vals[i] = mu, mu 
                break
            if (upper_cdf - lower_cdf) > rho:
                hi = mid
            else:
                lo = mid
        else: 
            δ = 0.5 * (lo + hi)
            a_vals[i] = max(0.0, mu - δ)
            b_vals[i] = min(1.0, mu + δ)
    return a_vals, b_vals

# --- NEW A+++++ Pre-computation Function ---
@st.cache_data
def precompute_values(_df):
    """
    Pre-computes HDR bounds and posterior means for the entire dataset.
    Uses @st.cache_data to run only once for a given input DataFrame.
    """
    df_processed = _df.copy()
    
    # Vectorized computation of posterior mean
    epsilon = 1e-9
    alphas = np.maximum(df_processed['alpha_before_update'].values, epsilon)
    betas = np.maximum(df_processed['beta_before_update'].values, epsilon)
    df_processed['posterior_mean'] = alphas / (alphas + betas)
    
    # For HDR, we still iterate per arm as it's a complex calculation
    # but this will only run ONCE thanks to the cache.
    df_processed['hdr_lower'] = 0.0
    df_processed['hdr_upper'] = 0.0
    
    unique_arms = df_processed['arm_name_of_state'].unique()
    for arm_name in unique_arms:
        arm_mask = df_processed['arm_name_of_state'] == arm_name
        
        arm_alphas = df_processed.loc[arm_mask, 'alpha_before_update'].values
        arm_betas = df_processed.loc[arm_mask, 'beta_before_update'].values
        
        a_vals, b_vals = compute_hdr_bounds(arm_alphas, arm_betas, rho=0.50, tol=1e-6)
        
        df_processed.loc[arm_mask, 'hdr_lower'] = a_vals
        df_processed.loc[arm_mask, 'hdr_upper'] = b_vals
        
    return df_processed
    
# --- MODIFIED Main Plotting Function for Streamlit ---
def plot_ts_arm_evolution_streamlit(
    df_expanded_input: pd.DataFrame,    # Data for SELECTED arms & T-range
    df_log_input: pd.DataFrame,         # Log data for ALL arms & T-range (for barcode context)
    arms_to_plot: list,                 # Specific ORIGINAL arms TO PLOT
    all_query_nums_in_T_range: list, # All query numbers in the selected T-range (for x-axis consistency)
    dataset_name_for_plot: str,
    title_suffix: str = "",
    figsize=(20, 10), 
    show_main_plot: bool = True,
    show_alphabeta_plot: bool = True,
    show_barcode_plot: bool = True,
    mask_arm_names: bool = False, # NEW: Parameter to control arm name masking
    # Existing parameters for relative heights, used if components are shown
    alpha_beta_height_ratio: float = 0.40,
    barcode_height_ratio: float = 0.10
):
    if not arms_to_plot:
        logging.info("plot_ts_arm_evolution_streamlit: No arms selected for plotting.")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No arms selected for visualization.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    # --- Determine active plot components and their layout ---
    active_components_ordered = [] # Tracks which components are active, in their intended plot order
    base_proportions = {}          # Stores the target proportion for each component type

    # Define target proportions if all three were to be shown, using input ratios
    # These are relative weights for active plots.
    eff_ab_r = alpha_beta_height_ratio
    eff_bc_r = barcode_height_ratio
    eff_main_r = 1.0 - eff_ab_r - eff_bc_r

    # Ensure main plot has a minimum proportion if others are very large
    if eff_main_r < 0.1 and (eff_ab_r > 0 or eff_bc_r > 0) : # Check if others are present
        eff_main_r, eff_ab_r, eff_bc_r = 0.6, 0.2, 0.2 # Fallback to default balanced proportions

    base_proportions['main'] = eff_main_r
    base_proportions['alphabeta'] = eff_ab_r
    base_proportions['barcode'] = eff_bc_r
    
    current_total_proportion_sum = 0
    height_ratios_list_final = []

    if show_main_plot:
        active_components_ordered.append('main')
        height_ratios_list_final.append(base_proportions['main'])
        current_total_proportion_sum += base_proportions['main']
    if show_alphabeta_plot:
        active_components_ordered.append('alphabeta')
        height_ratios_list_final.append(base_proportions['alphabeta'])
        current_total_proportion_sum += base_proportions['alphabeta']
    if show_barcode_plot:
        active_components_ordered.append('barcode')
        height_ratios_list_final.append(base_proportions['barcode'])
        current_total_proportion_sum += base_proportions['barcode']

    if not active_components_ordered:
        logging.info("plot_ts_arm_evolution_streamlit: No plot components selected for display.")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No plot components selected to display.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    num_rows_per_arm = len(active_components_ordered)

    # Normalize height_ratios_list_final based on active components
    if current_total_proportion_sum > 0:
        height_ratios_list_final = [r / current_total_proportion_sum for r in height_ratios_list_final]
    else: # Should not happen if active_components_ordered is not empty
        height_ratios_list_final = [1.0 / num_rows_per_arm] * num_rows_per_arm if num_rows_per_arm > 0 else []


    n_arms_to_plot = len(arms_to_plot)
    if not all_query_nums_in_T_range:
        min_query, max_query = 0, 1
        logging.warning("plot_ts_arm_evolution_streamlit: No query numbers in all_query_nums_in_T_range.")
    else:
        min_query = min(all_query_nums_in_T_range)
        max_query = max(all_query_nums_in_T_range)

    epsilon_beta_params = 1e-9 
    fig = plt.figure(figsize=figsize)
    gs_outer = gridspec.GridSpec(n_arms_to_plot, 1, figure=fig, hspace=0.00)

    barcode_reward_colors = {1.0: PLOT_COLORS['success_reward'], 0.0: PLOT_COLORS['failure_reward']}
    posterior_draw_marker_style = {
        'marker': '.', 'color': PLOT_COLORS['posterior_draw_marker'],
        's': 60, 'edgecolors': greys_cmap(0.6), 'linewidths': 0.3
    }
    legend_handles_main = {} # Collect handles for the figure-level legend

    global_max_ab_val = 1.0 # Initialize
    if show_alphabeta_plot and not df_expanded_input.empty:
        if 'alpha_before_update' in df_expanded_input.columns:
            max_alpha_overall = df_expanded_input['alpha_before_update'].max(skipna=True)
            if pd.notna(max_alpha_overall): global_max_ab_val = max(global_max_ab_val, max_alpha_overall)
        if 'beta_before_update' in df_expanded_input.columns:
            max_beta_overall  = df_expanded_input['beta_before_update'].max(skipna=True)
            if pd.notna(max_beta_overall):  global_max_ab_val = max(global_max_ab_val, max_beta_overall)
    if global_max_ab_val < 1.0: global_max_ab_val = 1.0


    # --- NEW: Determine tick locations for vlines and x-axis based on all_query_nums_in_T_range ---
    query_nums_for_vlines_and_ticks = []
    if all_query_nums_in_T_range:
        if len(all_query_nums_in_T_range) <= 20: # If 20 or fewer, use all of them
            query_nums_for_vlines_and_ticks = all_query_nums_in_T_range
        else: # If more than 20, select a subset (e.g., 10)
            num_ticks_desired = 10 
            # Ensure num_ticks_desired is not more than available unique query numbers
            num_ticks_actual = min(num_ticks_desired, len(all_query_nums_in_T_range))
            if num_ticks_actual > 0 : # Ensure we have at least one tick to compute
                tick_indices = np.linspace(0, len(all_query_nums_in_T_range) - 1, num=num_ticks_actual, dtype=int)
                query_nums_for_vlines_and_ticks = [all_query_nums_in_T_range[j] for j in tick_indices]
            else: # Fallback if all_query_nums_in_T_range was empty or resulted in 0 ticks
                query_nums_for_vlines_and_ticks = [] 
    # --- END NEW ---


    # --- This is the new, corrected block ---
    # --- NEW: Prepare display labels for arms based on mask_arm_names setting ---
    arm_name_map_full = st.session_state.get('arm_name_map', {})
    display_labels_for_arms = []
    if mask_arm_names:
        # Use the pre-built global map to get masked names for the arms being plotted
        display_labels_for_arms = [arm_name_map_full.get(name, name) for name in arms_to_plot]
    else:
        display_labels_for_arms = arms_to_plot[:] # Use original names
    # --- END NEW ---


    for i, arm_of_interest in enumerate(arms_to_plot):
        display_name_current_arm = display_labels_for_arms[i] # Get the display name (original or masked)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            num_rows_per_arm, 1, subplot_spec=gs_outer[i],
            height_ratios=height_ratios_list_final, hspace=0.00
        )
        
        ax_main = None
        ax_alphabeta = None
        ax_barcode = None
        
        # Determine which axes will be created and their sharex relationship
        created_axes_this_arm = []
        shared_x_target = None
        current_subplot_idx = 0

        if 'main' in active_components_ordered:
            ax_main = fig.add_subplot(gs_inner[current_subplot_idx])
            if shared_x_target is None: shared_x_target = ax_main
            created_axes_this_arm.append(ax_main)
            current_subplot_idx +=1
        if 'alphabeta' in active_components_ordered:
            ax_alphabeta = fig.add_subplot(gs_inner[current_subplot_idx], sharex=shared_x_target)
            if shared_x_target is None: shared_x_target = ax_alphabeta # Should be set by ax_main if main is active
            created_axes_this_arm.append(ax_alphabeta)
            current_subplot_idx +=1
        if 'barcode' in active_components_ordered:
            ax_barcode = fig.add_subplot(gs_inner[current_subplot_idx], sharex=shared_x_target)
            # if shared_x_target is None: shared_x_target = ax_barcode # Should be set already
            created_axes_this_arm.append(ax_barcode)
            current_subplot_idx +=1
        
        # Assign arm ylabel to the topmost visible plot for this arm
        display_name_current_arm = display_labels_for_arms[i] # Get the display name (original or masked)
        if created_axes_this_arm:
            # Use display_name_current_arm for the y-label
            created_axes_this_arm[0].set_ylabel(display_name_current_arm, rotation=0, labelpad=50, va='center', ha='right', fontsize=11, fontweight='bold')

        arm_state_data = df_expanded_input[df_expanded_input['arm_name_of_state'] == arm_of_interest].sort_values('query_num_total')
        
        current_xlim_padding = max(1, (max_query - min_query) * 0.02 if max_query > min_query and all_query_nums_in_T_range else 1)
        current_xlim = (min_query - current_xlim_padding, max_query + current_xlim_padding)
        
        for ax_to_configure in created_axes_this_arm:
            ax_to_configure.set_xlim(current_xlim)
            # MODIFIED: Use pre-calculated query_nums_for_vlines_and_ticks
            if query_nums_for_vlines_and_ticks: 
                 for q_num_val in query_nums_for_vlines_and_ticks: 
                    ax_to_configure.axvline(q_num_val, color=PLOT_COLORS['vline_color_tuple'], linestyle=':', linewidth=0.5, zorder=1)

        if arm_state_data.empty:
            # Display "No data" message on the first available plot for this arm
            if created_axes_this_arm:
                 created_axes_this_arm[0].text(0.5, 0.5, 'No data for this arm in selected T-range', ha='center', va='center', transform=created_axes_this_arm[0].transAxes, fontsize=9, color='grey')
            
            is_last_arm_overall = (i == n_arms_to_plot - 1)
            bottom_ax_for_this_empty_arm = created_axes_this_arm[-1] if created_axes_this_arm else None

            for ax_clean in created_axes_this_arm:
                ax_clean.set_yticks([])
                is_bottom_plot_for_this_arm_stack = (ax_clean == bottom_ax_for_this_empty_arm)

                if is_last_arm_overall and is_bottom_plot_for_this_arm_stack and all_query_nums_in_T_range:
                     plt.setp(ax_clean.get_xticklabels(), visible=True, rotation=30, ha='right', fontsize=10)
                     ax_clean.set_xlabel("Sampling Step t", fontsize=12, labelpad=15)
                     ax_clean.spines['bottom'].set_linewidth(0.8); ax_clean.spines['bottom'].set_visible(True)
                else:
                     plt.setp(ax_clean.get_xticklabels(), visible=False)
                     ax_clean.spines['bottom'].set_visible(False)
                for spine_dir in ['left', 'top', 'right']: ax_clean.spines[spine_dir].set_visible(False)
            continue # Next arm

        x_axis_data_state = arm_state_data['query_num_total'].values

        # --- Main Plot (HDR, Mean, Draws) ---
        # --- THIS IS THE NEW, OPTIMIZED BLOCK ---
        if ax_main:
            # **CHANGE: Use precomputed values directly from the dataframe**
            a_vals = arm_state_data['hdr_lower'].values
            b_vals = arm_state_data['hdr_upper'].values
            mu_vals = arm_state_data['posterior_mean'].values
        
            # Plot HDR regions (no computation needed)
            for idx, T_val in enumerate(x_axis_data_state):
                a_T, b_T = a_vals[idx], b_vals[idx]
                ax_main.fill_between([T_val, T_val], [0.0, 0.0], [a_T, a_T], color=PLOT_COLORS['explore'], zorder=2)
                ax_main.fill_between([T_val, T_val], [a_T, a_T], [b_T, b_T], color=PLOT_COLORS['posterior_draw_marker'], zorder=3)
                ax_main.fill_between([T_val, T_val], [b_T, b_T], [1.0, 1.0], color=PLOT_COLORS['explore'], zorder=2)
        
            # Plot posterior mean (precomputed)
            h_mean_line, = ax_main.plot(x_axis_data_state, mu_vals, color=PLOT_COLORS['posterior_draw_marker'], linestyle='-', linewidth=1, label='Posterior Mean (μ)', zorder=4)
            if 'posterior_mean' not in legend_handles_main: legend_handles_main['posterior_mean'] = h_mean_line

            chosen_mask = (arm_state_data['arm'] == arm_of_interest)
            if chosen_mask.any():
                h_chosen_draws = ax_main.scatter(
                    arm_state_data.loc[chosen_mask, 'query_num_total'], arm_state_data.loc[chosen_mask, 'posterior_sample'],
                    **posterior_draw_marker_style, label='Posterior Draw (Max in t)', zorder=6)
                if 'chosen_draw' not in legend_handles_main: legend_handles_main['chosen_draw'] = h_chosen_draws

            not_chosen_mask = (arm_state_data['arm'] != arm_of_interest) & pd.notna(arm_state_data['posterior_sample'])
            if not_chosen_mask.any():
                h_not_chosen_draws = ax_main.scatter(
                    arm_state_data.loc[not_chosen_mask, 'query_num_total'], arm_state_data.loc[not_chosen_mask, 'posterior_sample'],
                    **posterior_draw_marker_style, label='Posterior Draw (Not Max in t)', zorder=5) # zorder 5 to be behind chosen
                if 'not_chosen_draw' not in legend_handles_main: legend_handles_main['not_chosen_draw'] = h_not_chosen_draws
            
            ax_main.set_ylim(-0.05, 1.05); ax_main.tick_params(axis='y', labelsize=9)
            ax_main.grid(False); ax_main.spines['top'].set_visible(False); ax_main.spines['right'].set_visible(False)

        # --- Alpha/Beta Plot ---
        if ax_alphabeta:
            alpha_plot_vals = arm_state_data['alpha_before_update']
            beta_plot_vals  = arm_state_data['beta_before_update']

            h_alpha, = ax_alphabeta.plot(x_axis_data_state, alpha_plot_vals, color=PLOT_COLORS['alpha_line'], linestyle='-', linewidth=1, label='α (Success Counts)')
            h_beta, = ax_alphabeta.plot(x_axis_data_state, beta_plot_vals, color=PLOT_COLORS['beta_line'], linestyle='--', linewidth=1, label='β (Failure Counts)')
            if 'alpha_counts' not in legend_handles_main: legend_handles_main['alpha_counts'] = h_alpha
            if 'beta_counts' not in legend_handles_main: legend_handles_main['beta_counts'] = h_beta

            ax_alphabeta.set_ylabel("Counts", fontsize=9, color="dimgray"); ax_alphabeta.tick_params(axis='y', labelsize=8, labelcolor="dimgray")
            ax_alphabeta.set_ylim(bottom=-0.05 * global_max_ab_val, top=max(1.1, global_max_ab_val * 1.1)) # Use calculated global_max_ab_val
            ax_alphabeta.grid(True, axis='y', linestyle=':', linewidth=0.5, color=PLOT_COLORS['vline_color_tuple'])
            ax_alphabeta.spines['top'].set_visible(False); ax_alphabeta.spines['right'].set_visible(False)
            ax_alphabeta.spines['left'].set_color('dimgray') # bottom spine handled later

        # --- Barcode Plot ---
        if ax_barcode:
            required_barcode_cols = {'query_num_total', 'arm', 'reward'}
            if not df_log_input.empty and required_barcode_cols.issubset(df_log_input.columns):
                arm_pull_data = df_log_input[
                    (df_log_input['arm'] == arm_of_interest) &
                    (df_log_input['query_num_total'] >= min_query) &
                    (df_log_input['query_num_total'] <= max_query)
                ].sort_values('query_num_total')

                if not arm_pull_data.empty:
                    for _, pull_row in arm_pull_data.iterrows():
                        q_num, reward = pull_row['query_num_total'], pull_row['reward']
                        barcode_color = barcode_reward_colors.get(reward, PLOT_COLORS['posterior_draw_marker']) # Fallback color
                        ax_barcode.plot([q_num, q_num], [0.05, 0.95], color=barcode_color, linestyle='-', linewidth=2, solid_capstyle='butt', zorder=3)
            
            ax_barcode.set_yticks([]); ax_barcode.set_yticklabels([])
            ax_barcode.spines['left'].set_visible(False); ax_barcode.spines['top'].set_visible(False); ax_barcode.spines['right'].set_visible(False)
            ax_barcode.set_ylim(-0.1, 1.1); ax_barcode.grid(False)

        # --- X-axis tick and label visibility management for the current arm's stack ---
        is_last_arm_being_plotted = (i == n_arms_to_plot - 1)
        bottom_most_visible_ax_for_this_arm = created_axes_this_arm[-1] if created_axes_this_arm else None

        for ax_k in created_axes_this_arm:
            is_bottom_plot_of_stack = (ax_k == bottom_most_visible_ax_for_this_arm)
            if is_last_arm_being_plotted and is_bottom_plot_of_stack:
                # This is the very last plot of the entire figure
                plt.setp(ax_k.get_xticklabels(), visible=True, rotation=30, ha='right', fontsize=10)
                ax_k.set_xlabel("Sampling Step t", fontsize=12, labelpad=15)
                ax_k.spines['bottom'].set_linewidth(0.8); ax_k.spines['bottom'].set_visible(True)
                # MODIFIED: Use pre-calculated query_nums_for_vlines_and_ticks for setting x-ticks
                if query_nums_for_vlines_and_ticks:
                     ax_k.set_xticks(query_nums_for_vlines_and_ticks)
                elif all_query_nums_in_T_range: # Fallback to all if subsetting failed or wasn't needed but still want ticks
                    ax_k.set_xticks(all_query_nums_in_T_range)
            else:
                # Not the last plot of the figure, or not the bottom of its stack if it's not the last arm
                plt.setp(ax_k.get_xticklabels(), visible=False)
                ax_k.spines['bottom'].set_visible(False)


    # ─────────────────────────────────────────────────────────────────────
    # (Legend building code - unchanged as it's already robust)
    # ─────────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = []
    if show_main_plot: # Only add main plot related legend items if main plot is shown
        patch_exploit = Patch(facecolor=PLOT_COLORS['posterior_draw_marker'], edgecolor='none', label='HDR (50%)')
        legend_handles.append(patch_exploit)
        patch_explore = Patch(facecolor=PLOT_COLORS['explore'], edgecolor='none', label='Explore Tails')
        legend_handles.append(patch_explore)
        if 'posterior_mean' in legend_handles_main: legend_handles.append(legend_handles_main['posterior_mean'])
        else: legend_handles.append(Line2D([], [], color=PLOT_COLORS['posterior_draw_marker'], linewidth=1.5, label='Posterior Mean (μ)'))
        if 'chosen_draw' in legend_handles_main: legend_handles.append(legend_handles_main['chosen_draw'])
        else: legend_handles.append(Line2D([], [], marker='o', color=PLOT_COLORS['posterior_draw_marker'], linestyle='None', markersize=10, label='Posterior Draw (Chosen)'))
        if 'not_chosen_draw' in legend_handles_main: legend_handles.append(legend_handles_main['not_chosen_draw'])
        else: legend_handles.append(Line2D([], [], marker='o', color=PLOT_COLORS['posterior_draw_marker'], linestyle='None', markersize=10, label='Posterior Draw (Not Chosen)')) # Should be different marker style ideally if main not shown

    vline_handle = Line2D([], [], marker='|', linestyle=':', color=PLOT_COLORS['vline'], linewidth=1.5, label='Sampling Step Indicator')
    legend_handles.append(vline_handle)

    if show_alphabeta_plot: # Only add alpha/beta legend items if shown
        if 'alpha_counts' in legend_handles_main: legend_handles.append(legend_handles_main['alpha_counts'])
        else: legend_handles.append(Line2D([], [], color=PLOT_COLORS['alpha_line'], linewidth=1.5, label='α (Success Counts)'))
        if 'beta_counts' in legend_handles_main: legend_handles.append(legend_handles_main['beta_counts'])
        else: legend_handles.append(Line2D([], [], color=PLOT_COLORS['beta_line'], linestyle='--', linewidth=1.5, label='β (Failure Counts)'))

    if show_barcode_plot: # Only add barcode legend items if shown
        rel_color = barcode_reward_colors.get(1.0, PLOT_COLORS['posterior_draw_marker'])
        irr_color = barcode_reward_colors.get(0.0, PLOT_COLORS['posterior_draw_marker'])
        barcode_rel = Line2D([], [], color=rel_color, marker='|', markersize=12, linewidth=0, label='Sample Pulled: Relevant')
        barcode_irr = Line2D([], [], color=irr_color, marker='|', markersize=12, linewidth=0, label='Sample Pulled: Irrelevant')
        legend_handles.extend([barcode_rel, barcode_irr])

    if legend_handles: # Only show legend if there's something to show
        fig.legend(
            handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995),
            ncol=min(5, len(legend_handles)), frameon=False, fontsize=10
        )
    # ─────────────────────────────────────────────────────────────────────

    fig.suptitle(
        f"DATASET: {dataset_name_for_plot}{title_suffix}",
        fontsize=17, y=1.015, fontweight='bold'
    )
    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.92]) # Adjusted rect slightly for legend space if needed
    return fig

# --- NEW A+++++ Minimal Function for XAI Snapshot View ---
def plot_xai_snapshot_minimal(
    df_snapshot: pd.DataFrame, 
    pulled_arm: str, 
    arms_to_plot: list,
    mask_arm_names: bool,
    figsize=(10, 7)
):
    """
    Creates a minimal, focused bar chart comparing selected arms at a single sampling step 't'.
    - Horizontal bars for posterior mean.
    - Dots for posterior samples.
    - Log scale on the x-axis for value comparison.
    - Minimal color scheme: gray and black.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if df_snapshot.empty or arms_to_plot is None or len(arms_to_plot) == 0:
        ax.text(0.5, 0.5, "No data available for the selected arms at this step.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        return fig

    # --- Data Preparation ---
    # Filter for only the arms selected for visualization
    df_plot = df_snapshot[df_snapshot['arm_name_of_state'].isin(arms_to_plot)].copy()
    
    # --- This is the new, corrected block ---
    # Create display names using the global map for consistency
    arm_name_map_full = st.session_state.get('arm_name_map', {})
    if mask_arm_names:
        df_plot['display_name'] = df_plot['arm_name_of_state'].map(arm_name_map_full)
    else:
        df_plot['display_name'] = df_plot['arm_name_of_state']
        
    # Order the dataframe according to the selection order in the sidebar
    df_plot['arm_name_of_state'] = pd.Categorical(df_plot['arm_name_of_state'], categories=arms_to_plot, ordered=True)
    df_plot = df_plot.sort_values('arm_name_of_state').reset_index(drop=True)
    
    # --- THIS IS THE NEW, OPTIMIZED BLOCK ---
    # **CHANGE: Use precomputed posterior_mean. Add a fallback for safety.**
    if 'posterior_mean' not in df_plot.columns:
        st.warning("Precomputed 'posterior_mean' not found. Calculating on-the-fly.")
        epsilon = 1e-9
        alphas = np.maximum(df_plot['alpha_before_update'].values, epsilon)
        betas = np.maximum(df_plot['beta_before_update'].values, epsilon)
        df_plot['posterior_mean'] = alphas / (alphas + betas)
    
    # --- Plotting ---
    arm_labels = df_plot['display_name']
    y_pos = np.arange(len(arm_labels))

    # --- This is the new, corrected block with highlighting ---
    # Bars for posterior mean
    bars = ax.barh(y_pos, df_plot['posterior_mean'], align='center', color='darkgray', zorder=2, height=0.6)
    
    # Dots for posterior sample
    ax.scatter(df_plot['posterior_sample'], y_pos, color='black', marker='o', s=100, zorder=3, ec='white', lw=1)
    
    # --- NEW: Highlight the winning arm with a bold border ---
    if pulled_arm is not None:
        # Find the index of the pulled arm in our plotted dataframe
        pulled_arm_indices = df_plot.index[df_plot['arm_name_of_state'] == pulled_arm].tolist()
        # --- This is the new, corrected block with a darker gray highlight ---
        if pulled_arm_indices:
            idx_pulled = pulled_arm_indices[0]
            # This single line changes the bar's color to be darker and fully opaque
            bars[idx_pulled].set_facecolor('dimgray')
    
    # --- Formatting ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(arm_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Value (Log Scale)', fontsize=11)
    ax.set_ylabel('Arm', fontsize=11)
    
    # Use log scale and handle potential zero values
    # We add a small constant to prevent log(0) errors.
    ax.set_xscale('log')
    xmin = df_plot[['posterior_mean', 'posterior_sample']].min().min()
    xmax = df_plot[['posterior_mean', 'posterior_sample']].max().max()
    ax.set_xlim(left=max(xmin * 0.5, 1e-12), right=xmax * 1.5) # Dynamic limits
    
    ax.grid(True, which="both", ls=":", linewidth=0.5, axis='x', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    query_t = df_snapshot['query_num_total'].iloc[0]
    fig.suptitle(f'XAI Snapshot: Arm Comparison at Sampling Step t = {query_t}', fontsize=16, y=0.98)

    # --- This is the new, corrected legend block ---
    # Minimalist Legend with Darker Bar Highlight
    legend_elements = [
        mpatches.Patch(color='darkgray', label='Posterior Mean ($\\mu$)'),
        mpatches.Patch(color='dimgray', label='Pulled Arm Mean'), # 'dimgray' is darker
        Line2D([0], [0], marker='o', color='w', label='Posterior Sample ($\\tilde{\\theta}$)',
               markerfacecolor='k', markersize=9, markeredgecolor='w'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=9.5)

    # --- Manual Layout Adjustment to avoid tight_layout() crash with logit scale ---
    plt.subplots_adjust(
        left=0.25,
        right=0.95,
        bottom=0.15,
        top=0.85 
    )
    return fig

# --- Streamlit App Logic ---
st.set_page_config(layout="wide", page_title="TS-Insight")
st.title("TS-Insight")
st.markdown("""
Thompson Sampling Arm Evolution Visualizer: Upload your experiment results (`.pt` file) and an optional configuration (`.json` file).
Select arms, adjust the sampling range (T), and choose which plot sections to display, then press "Run Visualization".
""")

# --- Initialize Session State ---
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'df_expanded_full' not in st.session_state: st.session_state.df_expanded_full = pd.DataFrame()
if 'df_log_full' not in st.session_state: st.session_state.df_log_full = pd.DataFrame()
if 'arms' not in st.session_state: st.session_state.arms = []
if 'all_query_nums_full' not in st.session_state: st.session_state.all_query_nums_full = []
if 'dataset_name' not in st.session_state: st.session_state.dataset_name = "Uploaded_Data"
if 'temp_pt_path' not in st.session_state: st.session_state.temp_pt_path = None
if 'temp_config_path' not in st.session_state: st.session_state.temp_config_path = None
if 'selected_arms_for_plot' not in st.session_state: st.session_state.selected_arms_for_plot = []
if 'plot_fig' not in st.session_state: st.session_state.plot_fig = None
if 'numeric_df' not in st.session_state: st.session_state.numeric_df = pd.DataFrame()
if 'selected_t_range_for_display' not in st.session_state: st.session_state.selected_t_range_for_display = (0,0)


# --- File Uploaders ---
st.sidebar.header("1. Upload Files")
uploaded_pt_file = st.sidebar.file_uploader("Upload .pt results file", type=["pt"], key="pt_uploader")
uploaded_config_file = st.sidebar.file_uploader("Upload .json config file (optional)", type=["json"], key="json_uploader")

# --- Data Loading and Processing Button ---
if st.sidebar.button("Load Data", key="load_data_button"):
    if uploaded_pt_file is not None:
        st.session_state.data_loaded = False 
        st.session_state.plot_fig = None
        st.session_state.numeric_df = pd.DataFrame()
        st.session_state.arms = []
        st.session_state.selected_arms_for_plot = []

        if st.session_state.temp_pt_path and os.path.exists(st.session_state.temp_pt_path):
            try: os.unlink(st.session_state.temp_pt_path)
            except: pass
        if st.session_state.temp_config_path and os.path.exists(st.session_state.temp_config_path):
            try: os.unlink(st.session_state.temp_config_path)
            except: pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_pt:
            tmp_pt.write(uploaded_pt_file.getvalue())
            st.session_state.temp_pt_path = tmp_pt.name
        if uploaded_config_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
                tmp_json.write(uploaded_config_file.getvalue())
                st.session_state.temp_config_path = tmp_json.name
        else: st.session_state.temp_config_path = None

        with st.spinner("Loading and processing data..."):
            try:
                al_results_data = torch.load(st.session_state.temp_pt_path, map_location='cpu', weights_only=False) # Assuming weights_only=False is intended.
                dataset_name = "Unknown_Dataset"
                if st.session_state.temp_config_path and os.path.exists(st.session_state.temp_config_path):
                    with open(st.session_state.temp_config_path, 'r') as f: config_run = json.load(f)
                    dataset_name = config_run.get("dataset_name", dataset_name)
                elif isinstance(al_results_data, dict):
                    if 'baseline_params' in al_results_data and isinstance(al_results_data['baseline_params'], dict) and 'dataset_name' in al_results_data['baseline_params']:
                        dataset_name = al_results_data['baseline_params']['dataset_name']
                    elif 'dataset_name' in al_results_data: dataset_name = al_results_data['dataset_name']
                st.session_state.dataset_name = dataset_name
                res_proc = al_results_data
                detailed_log_proc = res_proc.get('detailed_log')
                if not detailed_log_proc:
                    st.error("No 'detailed_log' found in the .pt file."); st.stop()

                df_log_full = pd.DataFrame(detailed_log_proc)
                if df_log_full.empty or 'arm_states' not in df_log_full.columns:
                    st.error("Detailed log is empty or missing 'arm_states'."); st.stop()

                original_cols_proc = df_log_full.columns.drop('arm_states', errors='ignore').tolist()
                temp_df_list_proc = []
                for _, row_log_proc in df_log_full.iterrows():
                    common_data_proc = {col: row_log_proc[col] for col in original_cols_proc}
                    if row_log_proc['arm_states'] is None or not isinstance(row_log_proc['arm_states'], dict): continue
                    for arm_name_state_proc, state_dict_proc in row_log_proc['arm_states'].items():
                        record_proc = {**common_data_proc, 'arm_name_of_state': arm_name_state_proc, **state_dict_proc}
                        temp_df_list_proc.append(record_proc)
                
                df_expanded_full = pd.DataFrame(temp_df_list_proc)
                if df_expanded_full.empty:
                    st.error("Failed to expand arm states."); st.stop()

                required_cols = ['query_num_total', 'alpha_before_update', 'beta_before_update', 'posterior_sample', 'reward', 'arm_name_of_state', 'arm']
                for rcol in required_cols:
                    if rcol not in df_expanded_full.columns:
                        logging.warning(f"Column '{rcol}' missing in expanded data. Filling with default.")
                        if rcol in ['arm_name_of_state', 'arm']: df_expanded_full[rcol] = "Unknown"
                        elif rcol == 'query_num_total': df_expanded_full[rcol] = 0
                        else: df_expanded_full[rcol] = 0.0 
                
                df_expanded_full = df_expanded_full.astype({
                    'query_num_total': int, 'alpha_before_update': float, 'beta_before_update': float,
                    'posterior_sample': float, 'reward': float, 'arm_name_of_state': str, 'arm': str
                })

                # --- THIS IS THE NEW, OPTIMIZED BLOCK ---
                # **NEW: PRECOMPUTE ALL VALUES AND CACHE THE RESULT**
                with st.spinner("Pre-computing derived values for visualization..."):
                    df_expanded_precomputed = precompute_values(df_expanded_full)

                st.session_state.df_expanded_full = df_expanded_precomputed
                st.session_state.df_log_full = df_log_full

                ts_param_order_proc = list(res_proc.get('ts_final_params', {}).keys())
                unique_arms_from_states = df_expanded_full['arm_name_of_state'].unique()
                # --- This is the new, corrected block ---
                st.session_state.arms = [a for a in ts_param_order_proc if a in unique_arms_from_states] or sorted(list(unique_arms_from_states))
                st.session_state.all_query_nums_full = sorted(df_expanded_full['query_num_total'].unique())
                st.session_state.selected_arms_for_plot = st.session_state.arms[:]
                
                # --- NEW: Create a stable, global map for masked names ---
                st.session_state.arm_name_map = {name: f"Arm {i+1}" for i, name in enumerate(st.session_state.arms)}

                if not st.session_state.arms or not st.session_state.all_query_nums_full:
                    st.error("No arms or query numbers found after processing."); st.stop()
                
                st.session_state.data_loaded = True
                min_q_init = min(st.session_state.all_query_nums_full) if st.session_state.all_query_nums_full else 0
                max_q_init = max(st.session_state.all_query_nums_full) if st.session_state.all_query_nums_full else 1
                st.session_state.selected_t_range_for_display = (min_q_init, max_q_init)

                st.sidebar.success(f"Data loaded: {st.session_state.dataset_name}")
                st.rerun() 
            except Exception as e:
                st.error(f"Error loading/processing data: {e}"); import traceback; st.error(traceback.format_exc())
            finally:
                if st.session_state.temp_pt_path and os.path.exists(st.session_state.temp_pt_path):
                    try: os.unlink(st.session_state.temp_pt_path); st.session_state.temp_pt_path = None
                    except: pass
                if st.session_state.temp_config_path and os.path.exists(st.session_state.temp_config_path):
                    try: os.unlink(st.session_state.temp_config_path); st.session_state.temp_config_path = None
                    except: pass
    else: st.sidebar.warning("Please upload a .pt file.")

# --- Display Info and Controls if Data Loaded ---
if st.session_state.data_loaded:
    st.sidebar.header("2. Plot Controls")
    st.sidebar.write(f"**Dataset:** {st.session_state.dataset_name}")
    
    # --- This is the new, corrected multiselect block ---
    # Use the format_func to display numbered arm names in the dropdown
    selected_arms_val = st.sidebar.multiselect(
        "Select Arms to Visualize:",
        options=st.session_state.arms,  # The actual values remain the original arm names
        default=st.session_state.selected_arms_for_plot,
        key="arm_selector_multiselect",
        format_func=lambda arm_name: f"{st.session_state.arm_name_map.get(arm_name, '')}: {arm_name}"
    )
    if selected_arms_val != st.session_state.selected_arms_for_plot : 
        st.session_state.selected_arms_for_plot = selected_arms_val
    st.sidebar.caption(f"{len(st.session_state.selected_arms_for_plot)} of {len(st.session_state.arms)} arms selected.")

    min_q_overall = min(st.session_state.all_query_nums_full) if st.session_state.all_query_nums_full else 0
    max_q_overall = max(st.session_state.all_query_nums_full) if st.session_state.all_query_nums_full else 1
    st.sidebar.write(f"**Overall Sampling Range (T):** {min_q_overall}–{max_q_overall}")

    default_t_min = st.session_state.selected_t_range_for_display[0]
    default_t_max = st.session_state.selected_t_range_for_display[1]
    if not (min_q_overall <= default_t_min <= max_q_overall and min_q_overall <= default_t_max <= max_q_overall and default_t_min <= default_t_max) :
        default_t_min, default_t_max = min_q_overall, max_q_overall

    if min_q_overall >= max_q_overall : 
        selected_t_range_current = (min_q_overall, max_q_overall)
        st.sidebar.info(f"Sampling range fixed: T={min_q_overall}" if min_q_overall == max_q_overall else "Sampling range not available.")
    else:
        selected_t_range_current = st.sidebar.slider(
            "Select T Range (Sampling steps):",
            min_value=min_q_overall, max_value=max_q_overall,
            value=(default_t_min, default_t_max),
            key="t_range_slider_control"
        )
    if selected_t_range_current != st.session_state.selected_t_range_for_display:
        st.session_state.selected_t_range_for_display = selected_t_range_current

    # NEW: Plot component visibility controls
    st.sidebar.subheader("Plot Component Visibility")
    cfg_show_main_plot = st.sidebar.checkbox("Show Main Plot (HDR/Mean/Draws)", value=True, key="cfg_show_main_plot")
    cfg_show_alphabeta_plot = st.sidebar.checkbox("Show Alpha/Beta Lines", value=True, key="cfg_show_alphabeta_plot") 
    cfg_show_barcode_plot = st.sidebar.checkbox("Show Barcode Plot", value=True, key="cfg_show_barcode_plot")
    # --- NEW: Checkbox for masking arm names ---
    cfg_mask_arm_names = st.sidebar.checkbox("Mask Arm Names (e.g., Arm 1, Arm 2,...)", value=False, key="cfg_mask_arm_names")


    if st.sidebar.button("Run Visualization", key="run_viz_button"):
        st.session_state.plot_fig = None 
        st.session_state.numeric_df = pd.DataFrame()

        if not st.session_state.selected_arms_for_plot:
            st.warning("Please select at least one arm to visualize.")
        elif not cfg_show_main_plot and not cfg_show_alphabeta_plot and not cfg_show_barcode_plot:
            st.warning("Please select at least one plot component to display (Main, Alpha/Beta, or Barcode).")
        else:
            with st.spinner("Processing & Generating Visuals..."):
                current_t_min, current_t_max = st.session_state.selected_t_range_for_display
                
                df_expanded_T_filtered = st.session_state.df_expanded_full[
                    (st.session_state.df_expanded_full['query_num_total'] >= current_t_min) &
                    (st.session_state.df_expanded_full['query_num_total'] <= current_t_max)
                ].copy()
                df_log_T_filtered = st.session_state.df_log_full[
                    (st.session_state.df_log_full['query_num_total'] >= current_t_min) &
                    (st.session_state.df_log_full['query_num_total'] <= current_t_max)
                ].copy()

                if df_expanded_T_filtered.empty and (cfg_show_main_plot or cfg_show_alphabeta_plot) : # Barcode might still show if log data exists
                    st.warning("No arm state data available for the selected T range to display Main or Alpha/Beta plots.")
                # Barcode plot can proceed even if df_expanded_T_filtered is empty, as it uses df_log_T_filtered
                # Check if any data is available for any selected plot type
                can_plot_something = False
                if cfg_show_main_plot and not df_expanded_T_filtered.empty : can_plot_something = True
                if cfg_show_alphabeta_plot and not df_expanded_T_filtered.empty : can_plot_something = True
                if cfg_show_barcode_plot and not df_log_T_filtered.empty : can_plot_something = True
                
                if not can_plot_something and not (df_expanded_T_filtered.empty and df_log_T_filtered.empty) : # if some data but not for selected plots
                     if not df_expanded_T_filtered.empty: # only log empty
                        pass # allow barcode to proceed if selected
                     elif not df_log_T_filtered.empty and (cfg_show_main_plot or cfg_show_alphabeta_plot): # only expanded empty
                        st.warning("No arm state data (for Main/AlphaBeta plots) in selected T range. Barcode might still appear if selected and data exists.")
                     # If both empty, a general "No data" will be handled by plotting function or later checks

                # Proceed if at least one component is selected AND ( (main or ab selected AND expanded data exists) OR (barcode selected AND log data exists) )
                # Simplified: plotting function will handle empty inputs for its respective parts.

                all_query_nums_for_current_T_range = sorted(
                    pd.concat([df_expanded_T_filtered['query_num_total'], df_log_T_filtered['query_num_total']]).unique()
                )

                if not all_query_nums_for_current_T_range and (cfg_show_main_plot or cfg_show_alphabeta_plot or cfg_show_barcode_plot):
                    st.warning("No Sampling steps in selected T range for any data.")
                else:
                    df_expanded_for_plot = df_expanded_T_filtered[
                        df_expanded_T_filtered['arm_name_of_state'].isin(st.session_state.selected_arms_for_plot)
                    ].copy()
                    
                    num_selected_arms = len(st.session_state.selected_arms_for_plot)
                    plot_height = (2 + 3.8 * num_selected_arms) if num_selected_arms > 0 else 6 
                    plot_width = 18

                    st.session_state.plot_fig = plot_ts_arm_evolution_streamlit(
                        df_expanded_input=df_expanded_for_plot, # Used for main and alpha/beta
                        df_log_input=df_log_T_filtered,         # Used for barcode
                        arms_to_plot=st.session_state.selected_arms_for_plot, # Pass original selected arm names
                        all_query_nums_in_T_range=all_query_nums_for_current_T_range,
                        dataset_name_for_plot=st.session_state.dataset_name,
                        title_suffix=f" (t: {current_t_min}–{current_t_max})", # t instead of T
                        show_main_plot=cfg_show_main_plot,
                        show_alphabeta_plot=cfg_show_alphabeta_plot,
                        show_barcode_plot=cfg_show_barcode_plot,
                        mask_arm_names=cfg_mask_arm_names, # --- NEW: Pass mask state ---
                        figsize=(plot_width, plot_height)
                        # alpha_beta_height_ratio and barcode_height_ratio use their defaults from function signature
                    )

                    # Numeric details generation (relies on df_expanded_T_filtered)
                    if not df_expanded_T_filtered.empty:
                        df_pull_info = df_log_T_filtered.set_index('query_num_total')[['arm', 'reward']].copy()
                        df_pull_info.rename(columns={'arm': 'pulled_arm_at_T', 'reward': 'reward_at_T'}, inplace=True)
                        
                        df_numeric_display = df_expanded_T_filtered.merge(df_pull_info, on='query_num_total', how='left')

                        def get_outcome(row):
                            if pd.isna(row['pulled_arm_at_T']): return "N/A (No pull recorded this T)"
                            if row['arm_name_of_state'] == row['pulled_arm_at_T']:
                                reward_val = row['reward_at_T']
                                if reward_val == 1.0: return "Pulled: Success (1.0)"
                                if reward_val == 0.0: return "Pulled: Failure (0.0)"
                                return f"Pulled: Reward ({reward_val})"
                            return "Not Pulled"
                        df_numeric_display['Outcome'] = df_numeric_display.apply(get_outcome, axis=1)
                        
                        st.session_state.numeric_df = df_numeric_display[[
                            'query_num_total', 'arm_name_of_state', 'alpha_before_update', 
                            'beta_before_update', 'posterior_sample', 'Outcome'
                        ]].rename(columns={
                            'query_num_total': 'T', 'arm_name_of_state': 'Arm',
                            'alpha_before_update': 'Alpha', 'beta_before_update': 'Beta',
                            'posterior_sample': 'Beta Draw', 'Outcome': 'Arm Status @ T'
                        }).sort_values(by=['T', 'Arm']).reset_index(drop=True)
                    else:
                        st.session_state.numeric_df = pd.DataFrame() # Clear if no expanded data for numeric details

# --- Main Area with Tabs ---
tab_vis, tab_num = st.tabs(["Visualization", "XAI"])

# --- THIS IS THE NEW, CORRECTED BLOCK ---
with tab_vis:
    # This tab's only job is to DISPLAY the plot if it exists.
    if st.session_state.get('plot_fig'):
        fig = st.session_state.plot_fig
        
        # Display the persistent figure from the session state
        st.pyplot(fig) # Display first

        # --- NEW: Download Logic with Format Selection ---
        col1_dl, col2_dl = st.columns([1, 4])
        with col1_dl:
            file_format = st.selectbox(
                "Format:",
                options=['SVG', 'PDF'], # SVG is the default
                key="main_plot_format_select"
            )

        format_options = {
            'SVG': {'ext': 'svg', 'mime': 'image/svg+xml'},
            'PDF': {'ext': 'pdf', 'mime': 'application/pdf'}
        }
        selected_format = format_options[file_format]
        buf = BytesIO()
        fig.savefig(buf, format=selected_format['ext'], dpi=300, bbox_inches="tight")
        buf.seek(0)
        
        with col2_dl:
            st.download_button(
                label=f"Download Plot as {file_format} (Camera-Ready)",
                data=buf,
                file_name=(
                    f"ts_evo_{st.session_state.dataset_name.replace(' ', '_')}"
                    f"_T{st.session_state.selected_t_range_for_display[0]}"
                    f"-{st.session_state.selected_t_range_for_display[1]}.{selected_format['ext']}"
                ),
                mime=selected_format['mime'],
                key="download_plot_main"
            )
    elif st.session_state.data_loaded:
        st.info("Adjust controls in the sidebar and click 'Run Visualization' to generate the main plot.")
    else:
        st.info("<- Upload a .pt file and click 'Load Data' to begin.")


# --- This is the new, corrected block ---
with tab_num:
    st.header("XAI Snapshot: Why was an arm chosen at a specific time?")
    st.markdown("""
    This view provides a snapshot of the algorithm's state at a single sampling step `t` to help answer: **"Why was a particular arm chosen?"**
    - **Gray Bar**: The arm's estimated average performance (posterior mean).
    - **Black Dot**: The actual random value sampled from the arm's belief. The highest dot wins.
    """)

    if st.session_state.data_loaded:
        # --- NEW: UI controls within the tab ---
        col1, col2 = st.columns([1, 2])
        
        # Get the list of available sampling steps within the selected T-range
        t_min_selected, t_max_selected = st.session_state.selected_t_range_for_display
        available_ts_in_range = sorted(
            st.session_state.df_log_full[
                (st.session_state.df_log_full['query_num_total'] >= t_min_selected) &
                (st.session_state.df_log_full['query_num_total'] <= t_max_selected)
            ]['query_num_total'].unique()
        )

        if not available_ts_in_range:
            st.warning("No sampling steps with recorded pulls found in the selected T-Range.")
        else:
            with col1:
                selected_t = st.selectbox(
                    f"Select Sampling Step (t):",
                    options=available_ts_in_range,
                    index=0,
                    format_func=lambda x: f"t = {x}",
                    help="Select a specific time `t` to see why the winning arm was chosen over all others."
                )
            with col2:
                # This checkbox determines the behavior
                respect_filter = st.checkbox(
                    "Show only arms selected in sidebar", 
                    value=False, 
                    help="Uncheck this to see all arms for a complete comparison, which is recommended for explanation."
                )
            
            # --- Plotting Logic ---
            if selected_t is not None:
                df_snapshot_t = st.session_state.df_expanded_full[
                    st.session_state.df_expanded_full['query_num_total'] == selected_t
                ].copy()

                pulled_arm_at_t_series = st.session_state.df_log_full[
                    st.session_state.df_log_full['query_num_total'] == selected_t
                ]['arm']
                pulled_arm_at_t = pulled_arm_at_t_series.iloc[0] if not pulled_arm_at_t_series.empty else None

                if df_snapshot_t.empty:
                    st.warning(f"No detailed arm state data available for t = {selected_t}.")
                else:
                    # Determine which list of arms to plot
                    if respect_filter:
                        arms_to_visualize = st.session_state.get('selected_arms_for_plot', [])
                        if not arms_to_visualize:
                            st.info("No arms selected in the sidebar. Showing all arms.")
                            arms_to_visualize = st.session_state.arms
                    else:
                        arms_to_visualize = st.session_state.arms # Default: show all arms

                    # Get the mask setting from the sidebar
                    mask_names_setting = st.session_state.get('cfg_mask_arm_names', False)
                    
                    # --- THIS IS THE NEW, CORRECTED BLOCK ---
                    # --- THIS IS THE NEW, CORRECTED BLOCK ---
                    with st.spinner("Generating XAI Snapshot..."):
                        xai_fig = plot_xai_snapshot_minimal(
                            df_snapshot=df_snapshot_t,
                            pulled_arm=pulled_arm_at_t,
                            arms_to_plot=arms_to_visualize,
                            mask_arm_names=mask_names_setting
                        )
                        
                        # --- STEP 1: PREPARE ALL DOWNLOADS BEFORE DISPLAYING ---
                        
                        # Create an in-memory buffer for the SVG format
                        svg_buf_xai = BytesIO()
                        xai_fig.savefig(svg_buf_xai, format="svg", dpi=300, bbox_inches="tight")
                        svg_buf_xai.seek(0)
                        
                        # Create an in-memory buffer for the PDF format
                        pdf_buf_xai = BytesIO()
                        xai_fig.savefig(pdf_buf_xai, format="pdf", dpi=300, bbox_inches="tight")
                        pdf_buf_xai.seek(0)
                        
                        # --- STEP 2: DISPLAY THE PLOT (AND CLEAR IT) ---
                        st.pyplot(xai_fig, clear_figure=True)
                    
                        # --- STEP 3: CREATE THE DOWNLOAD UI ---
                        col1_dl, col2_dl = st.columns([1, 4])
                        with col1_dl:
                            file_format_xai = st.selectbox(
                                "Format:",
                                options=['SVG', 'PDF'],
                                key=f"xai_format_select_{selected_t}" # Unique key
                            )
                        
                        with col2_dl:
                            if file_format_xai == 'SVG':
                                st.download_button(
                                    label=f"Download Snapshot as SVG",
                                    data=svg_buf_xai,
                                    file_name=f"xai_snapshot_{st.session_state.dataset_name.replace(' ', '_')}_t{selected_t}.svg",
                                    mime="image/svg+xml",
                                    key=f"download_xai_svg_{selected_t}"
                                )
                            else: # PDF
                                st.download_button(
                                    label=f"Download Snapshot as PDF",
                                    data=pdf_buf_xai,
                                    file_name=f"xai_snapshot_{st.session_state.dataset_name.replace(' ', '_')}_t{selected_t}.pdf",
                                    mime="application/pdf",
                                    key=f"download_xai_pdf_{selected_t}"
                                )
    else:
        st.info("<- Upload a .pt file and click 'Load Data' to begin.")

st.markdown("---")
st.caption("Use the 'XAI' tab for specific values in a sampling step.")
