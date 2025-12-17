"""
Causal Inference Visual Analysis Dashboard
==========================================
Pure visualization layer for causal inference results.

This module contains ALL plotting logic - it consumes CausalAnalyzer's outputs.
NO computation happens here, only visualization.

Responsibilities:
- Create all plots and charts
- Format visualizations for presentations
- Combine multiple plots into reports
- Takes CausalAnalyzer instance and calls its methods to get data

Architecture:
- CausalAnalyzer = Calculator (returns data)
- CausalDashboard = Visualizer (plots data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CausalDashboard:
    """
    Visual dashboard for comprehensive causal inference analysis.
    
    Provides visualization for:
    - Main DID results with confidence intervals
    - Event study (relative time effects)
    - Cohort effects
    - Heterogeneous treatment effects by subgroups
    - Parallel trends validation
    
    Usage:
    ------
    dashboard = CausalDashboard(analyzer)
    dashboard.full_report()  # Generate complete visual report
    """
    
    def __init__(self, analyzer, figsize_base: Tuple[int, int] = (14, 5)):
        """
        Initialize dashboard with a CausalAnalyzer instance.
        
        Parameters
        ----------
        analyzer : CausalAnalyzer
            Fitted CausalAnalyzer object with matched data
        figsize_base : tuple
            Base figure size for plots
        """
        self.analyzer = analyzer
        self.figsize_base = figsize_base
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style for professional output."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10.5
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['axes.labelsize'] = 11.5
        plt.rcParams['figure.autolayout'] = False
    
    # =========================================================================
    # MAIN DID RESULT
    # =========================================================================
    
    def plot_main_result(self, ax=None) -> None:
        """Plot main ATT result with confidence interval."""
        result = self.analyzer.estimate_did()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Bar for ATT
        color = 'tab:green' if result.att > 0 else 'tab:red'
        ax.barh(['ATT'], [result.att], color=color, alpha=0.65, height=0.35)
        
        # Error bar for CI
        ax.errorbar(
            result.att, 0, 
            xerr=[[result.att - result.ci_lower], [result.ci_upper - result.att]],
            fmt='o', color='black', capsize=8, capthick=2, markersize=10
        )
        
        # Zero line
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Annotations
        sig = '***' if result.p_value < 0.01 else '**' if result.p_value < 0.05 else '*' if result.p_value < 0.1 else ''
        ax.set_title(f'Main Effect: ATT = {result.att:,.0f} {sig}', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Causal Effect on {self.analyzer.outcome_col}')
        
        # Add text annotation
        ax.text(
            0.02, 0.93, 
            f'95% CI: [{result.ci_lower:,.0f}, {result.ci_upper:,.0f}]\np-value: {result.p_value:.4f}',
            transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35, pad=0.3)
        )
        
        plt.tight_layout()
    
    # =========================================================================
    # EVENT STUDY (Relative Time Effects)
    # =========================================================================
    
    def plot_event_study(self, pre_periods: int = 6, post_periods: int = 6, 
                         ax=None, true_effect: Optional[float] = None) -> None:
        """
        Plot event study with pre/post treatment effects.
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        ax : matplotlib axis
            Optional axis to plot on
        true_effect : float, optional
            If provided, draws horizontal line at true effect for validation
        """
        es = self.analyzer.estimate_event_study(pre_periods, post_periods)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)
        
        # Plot pre-treatment (gray)
        pre = es[es['relative_time'] < 0]
        post = es[es['relative_time'] >= 0]
        
        # Pre-treatment coefficients
        ax.errorbar(
            pre['relative_time'], pre['att'],
            yerr=[pre['att'] - pre['ci_lower'], pre['ci_upper'] - pre['att']],
            fmt='o-', capsize=4, markersize=7, color='gray', alpha=0.65,
            label='Pre-Treatment (should ≈ 0)'
        )
        
        # Post-treatment coefficients
        ax.errorbar(
            post['relative_time'], post['att'],
            yerr=[post['att'] - post['ci_lower'], post['ci_upper'] - post['att']],
            fmt='o-', capsize=4, markersize=7, color='tab:blue',
            label='Post-Treatment (causal effect)'
        )
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Treatment')
        
        # True effect line if provided
        if true_effect is not None:
            ax.axhline(y=true_effect, color='green', linestyle=':', alpha=0.7, 
                      linewidth=2, label=f'True Effect ({true_effect:,.0f})')
        
        # Shade regions
        ax.axvspan(-pre_periods - 0.5, -0.5, alpha=0.05, color='gray')
        ax.axvspan(-0.5, post_periods + 0.5, alpha=0.05, color='green')
        
        ax.set_xlabel('Months Relative to Treatment')
        ax.set_ylabel(f'Effect on {self.analyzer.outcome_col}')
        ax.set_title('Event Study: Dynamic Treatment Effects', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add validation annotation
        pre_mean = pre['att'].mean()
        pre_max = pre['att'].abs().max()
        ax.text(
            0.98, 0.02,
            f'Pre-trend mean: {pre_mean:,.0f}\nPre-trend max: {pre_max:,.0f}',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
        )
    
    # =========================================================================
    # COHORT EFFECTS
    # =========================================================================
    
    def plot_cohort_effects(self, ax=None) -> None:
        """Plot treatment effects by treatment cohort."""
        cohort_df = self.analyzer.effect_by_cohort()
        
        if len(cohort_df) == 0:
            print("No cohort data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)
        
        x = range(len(cohort_df))
        
        # Color by significance
        colors = ['tab:blue' if p < 0.05 else 'gray' for p in cohort_df['p_value']]
        
        ax.bar(x, cohort_df['att'], color=colors, alpha=0.7, edgecolor='black')
        ax.errorbar(
            x, cohort_df['att'],
            yerr=[cohort_df['att'] - cohort_df['ci_lower'], 
                  cohort_df['ci_upper'] - cohort_df['att']],
            fmt='none', capsize=5, capthick=2, color='black'
        )
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Overall mean line
        overall_att = cohort_df['att'].mean()
        ax.axhline(y=overall_att, color='green', linestyle=':', alpha=0.7, 
                  linewidth=2, label=f'Mean ATT: {overall_att:,.0f}')
        
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_df['cohort'], rotation=45, ha='right')
        ax.set_xlabel('Treatment Cohort')
        ax.set_ylabel(f'Effect on {self.analyzer.outcome_col}')
        ax.set_title('Effects by Cohort', fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
        
        # Add summary stats
        std_att = cohort_df['att'].std()
        ax.text(
            0.97, 0.03,
            f'Mean: {overall_att:,.0f}\nStd: {std_att:,.0f}\nN cohorts: {len(cohort_df)}',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.25)
        )
    
    # =========================================================================
    # EFFECT MATRIX HEATMAP
    # =========================================================================
    
    def plot_effect_matrix(
        self, 
        pre_periods: int = 6, 
        post_periods: int = 6,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'RdYlGn',
        annot: bool = True,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """
        Plot cohort × relative time effect matrix as a heatmap.
        
        Shows how treatment effects evolve for each cohort over time (M-x to M+y).
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        figsize : tuple, optional
            Figure size (auto-calculated if None)
        cmap : str
            Colormap name (default: RdYlGn for red-yellow-green)
        annot : bool
            Whether to annotate cells with values
        save_path : str, optional
            Path to save the figure
        """
        matrix = self.analyzer.effect_matrix(pre_periods, post_periods, metric='att')
        
        if len(matrix) == 0:
            print("No effect matrix data available")
            return
        
        # Auto-size based on matrix dimensions
        created_fig = False
        if ax is None:
            if figsize is None:
                figsize = (max(12, len(matrix.columns) * 1.2), max(6, len(matrix) * 0.5))
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        
        # Create heatmap
        # Center colormap at 0
        vmax = max(abs(matrix.max().max()), abs(matrix.min().min()))
        vmin = -vmax
        
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=',.0f',
            annot_kws={'fontsize': 7},
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': f'Effect on {self.analyzer.outcome_col}', 'shrink': 0.75}
        )
        
        # Add vertical line at M+0
        m0_idx = list(matrix.columns).index('M+0') if 'M+0' in matrix.columns else None
        if m0_idx is not None:
            ax.axvline(x=m0_idx, color='black', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Relative Time', fontsize=11)
        ax.set_ylabel('Treatment Cohort', fontsize=11)
        ax.set_title('Effect Matrix: Cohort × Time', fontsize=12, fontweight='bold')
        
        # Add summary annotation
        post_cols = [c for c in matrix.columns if c.startswith('M+')]
        if post_cols:
            post_mean = matrix[post_cols].mean().mean()
            ax.text(
                1.02, 0.5,
                f'Post-treatment\nmean: {post_mean:,.0f}',
                transform=ax.transAxes, fontsize=10, va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Effect matrix saved to: {save_path}")
        
        if created_fig:
            plt.show()
    
    def get_effect_matrix_table(
        self,
        pre_periods: int = 3,
        post_periods: int = 6
    ) -> pd.DataFrame:
        """
        Return the effect matrix as a styled DataFrame for display.
        
        Returns
        -------
        pd.DataFrame
            Cohort × time matrix with formatted values
        """
        matrix = self.analyzer.effect_matrix(pre_periods, post_periods, metric='att')
        return matrix.round(0).astype(int)
    
    # =========================================================================
    # HETEROGENEOUS EFFECTS
    # =========================================================================
    
    def plot_heterogeneous_effects(self, features: List[str], ax=None) -> None:
        """
        Plot treatment effects by subgroups (heterogeneous treatment effects).
        
        Parameters
        ----------
        features : list
            List of column names to analyze (e.g., ['segmento', 'region'])
        """
        all_results = []
        
        for feature in features:
            try:
                res = self.analyzer.effect_by_subgroup(filter_col=feature)
                res['feature'] = feature
                all_results.append(res)
            except Exception:
                continue
        
        if not all_results:
            print("No heterogeneity data available")
            return
        
        het_df = pd.concat(all_results, ignore_index=True)
        het_df = het_df.sort_values('att', ascending=True)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(6, len(het_df) * 0.4)))
        
        y = range(len(het_df))
        
        # Color by significance
        colors = ['tab:green' if p < 0.05 and att > 0 
                  else 'tab:red' if p < 0.05 and att < 0 
                  else 'gray' 
                  for p, att in zip(het_df['p_value'], het_df['att'])]
        
        ax.barh(y, het_df['att'], color=colors, alpha=0.7, edgecolor='black')
        ax.errorbar(
            het_df['att'], y,
            xerr=[het_df['att'] - het_df['ci_lower'], 
                  het_df['ci_upper'] - het_df['att']],
            fmt='none', capsize=4, capthick=1.5, color='black'
        )
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        ax.set_yticks(y)
        ax.set_yticklabels(het_df['subgroup'])
        ax.set_xlabel(f'Effect on {self.analyzer.outcome_col}')
        ax.set_title('Heterogeneous Treatment Effects by Subgroup', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()

    # =========================================================================
    # LOVE PLOT (SMD)
    # =========================================================================
    def _compute_smd(self, df: pd.DataFrame, feature: str) -> float:
        """Compute standardized mean difference for a feature (treated vs control)."""
        treated = df[df['is_treated'] == 1][feature].dropna()
        control = df[df['is_treated'] == 0][feature].dropna()
        if len(treated) < 2 or len(control) < 2:
            return np.nan
        m1, m0 = treated.mean(), control.mean()
        s1, s0 = treated.std(), control.std()
        pooled = np.sqrt((s1**2 + s0**2) / 2)
        if pooled == 0:
            return np.nan
        return (m1 - m0) / pooled
    
    def plot_love_plot(self, features: Optional[List[str]] = None, pre_only: bool = True, ax=None) -> None:
        """
        Plot standardized mean differences (SMD) for selected numeric features.
        
        Parameters
        ----------
        features : list, optional
            List of numeric feature names. If None, auto-select numeric columns.
        pre_only : bool
            If True, compute SMD on pre-treatment observations (relative_time < 0).
        ax : matplotlib axis, optional
        """
        panel = self.analyzer.get_panel()
        data = panel[panel['relative_time'] < 0] if pre_only else panel
        
        # Auto-select numeric columns if not provided
        if features is None:
            drop_cols = {
                self.analyzer.id_col,
                self.analyzer.time_col,
                self.analyzer.outcome_col,
                'matching_cohort',
                'relative_time',
                'post',
                'treated_x_post',
                'is_treated',
            }
            features = [
                c for c in data.columns
                if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])
            ]
        
        smd_rows = []
        for feat in features:
            smd = self._compute_smd(data, feat)
            smd_rows.append({'feature': feat, 'smd': smd})
        
        smd_df = pd.DataFrame(smd_rows).dropna().sort_values('smd')
        if smd_df.empty:
            if ax is None:
                print("No features available for SMD plot.")
            else:
                ax.text(0.5, 0.5, 'No features available', ha='center', va='center')
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, max(4, len(smd_df) * 0.3)))
        
        y = range(len(smd_df))
        ax.barh(y, smd_df['smd'], color='tab:blue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        # Positive thresholds
        ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='SMD=±0.1')
        ax.axvline(0.25, color='orange', linestyle=':', alpha=0.7, label='SMD=±0.25')
        # Negative thresholds
        ax.axvline(-0.1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(-0.25, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_yticks(y)
        ax.set_yticklabels(smd_df['feature'])
        ax.set_xlabel('Standardized Mean Difference (SMD)')
        ax.set_title('Balance Check', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8.5)
        ax.grid(alpha=0.3, axis='x')
    
    # =========================================================================
    # PARALLEL TRENDS
    # =========================================================================
    
    def plot_parallel_trends(self, ax=None) -> None:
        """Plot raw means for treated and control over time."""
        data = self.analyzer.effect_over_time()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)
        
        for group, color in [('Treated', 'tab:orange'), ('Control', 'tab:blue')]:
            subset = data[data['group'] == group].sort_values('relative_time')
            ax.plot(subset['relative_time'], subset['mean'], 
                   marker='o', label=group, color=color, linewidth=2, markersize=6)
            ax.fill_between(subset['relative_time'], 
                          subset['ci_lower'], subset['ci_upper'],
                          alpha=0.15, color=color)
        
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Treatment')
        ax.set_xlabel('Months Relative to Treatment')
        ax.set_ylabel(f'Mean {self.analyzer.outcome_col}')
        ax.set_title('Parallel Trends Check', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Shade pre-treatment
        xmin = data['relative_time'].min()
        ax.axvspan(xmin - 0.5, -0.5, alpha=0.05, color='gray')
    
    # =========================================================================
    # VALIDATION SUMMARY
    # =========================================================================
    
    def plot_validation_summary(self, ax=None) -> None:
        """Plot validation checks as a visual summary."""
        checks = self.analyzer.validate(verbose=False)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        check_names = list(checks.keys())
        check_vals = [1 if v else 0 if v is not None else 0.5 for v in checks.values()]
        colors = ['tab:green' if v else 'tab:red' if v is not None else 'tab:gray' 
                  for v in checks.values()]
        
        ax.barh(check_names, check_vals, color=colors, alpha=0.7)
        
        for i, (name, val) in enumerate(checks.items()):
            status = '✓' if val else '✗' if val is not None else '?'
            ax.text(0.5, i, status, ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title('Validation Checks', fontweight='bold')
        ax.set_xlabel('')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tab:green', alpha=0.7, label='Pass'),
            Patch(facecolor='tab:red', alpha=0.7, label='Fail'),
            Patch(facecolor='tab:gray', alpha=0.7, label='N/A')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # =========================================================================
    # FULL REPORT
    # =========================================================================
    
    def full_report(self, 
                    heterogeneity_features: Optional[List[str]] = None,
                    true_effect: Optional[float] = None,
                    save_path: Optional[str] = None) -> None:
        """
        Generate comprehensive executive-ready visual report with all analyses.
        
        Parameters
        ----------
        heterogeneity_features : list, optional
            Features to use for heterogeneity analysis
        true_effect : float, optional
            True effect value for validation (if known)
        save_path : str, optional
            Path to save the figure
        """
        # Executive-friendly styling
        plt.style.use('seaborn-v0_8-white')
        
        # Large canvas with ample whitespace
        fig = plt.figure(figsize=(20, 24), facecolor='white')
        
        # Layout: Header + 4 rows × 2 columns
        # Header: Title and key metrics (spanning full width)
        # Row 1: Main ATT + Validation Checks
        # Row 2: Event Study (full width)
        # Row 3: Parallel Trends + Love Plot
        # Row 4: Cohort Effects + Effect Matrix
        # Row 5: Heterogeneity (if provided) + placeholder
        
        gs = fig.add_gridspec(
            6, 2,
            height_ratios=[0.35, 1.0, 1.15, 1.0, 1.0, 0.95],
            hspace=0.55, wspace=0.35,
            top=0.96, bottom=0.04, left=0.06, right=0.96
        )
        
        # ===== HEADER: Key Metrics =====
        result = self.analyzer.estimate_did()
        sig = '***' if result.p_value < 0.01 else '**' if result.p_value < 0.05 else '*' if result.p_value < 0.1 else ''
        
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Title
        ax_header.text(
            0.5, 0.75, 
            'Causal Inference Analysis Report',
            ha='center', va='center', fontsize=22, fontweight='bold',
            transform=ax_header.transAxes
        )
        
        # Key metrics boxes
        metrics_text = (
            f"Average Treatment Effect: {result.att:,.0f} {sig}     "
            f"95% CI: [{result.ci_lower:,.0f}, {result.ci_upper:,.0f}]     "
            f"p-value: {result.p_value:.4f}     "
            f"N Treated: {result.n_treated:,}     "
            f"N Control: {result.n_control:,}"
        )
        
        ax_header.text(
            0.5, 0.15,
            metrics_text,
            ha='center', va='center', fontsize=13,
            transform=ax_header.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', 
                     edgecolor='#888888', linewidth=1.5, alpha=0.9)
        )
        
        # ===== ROW 1: Main ATT + Validation =====
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        
        # Main ATT (simplified, cleaner)
        self._plot_main_result_executive(ax1, result)
        
        # Validation checks
        self.plot_validation_summary(ax=ax2)
        ax2.set_title('Validation Checks', fontsize=14, fontweight='bold', pad=12)
        
        # ===== ROW 2: Event Study (full width) =====
        ax3 = fig.add_subplot(gs[2, :])
        self.plot_event_study(ax=ax3, true_effect=true_effect)
        ax3.set_title('Event Study: Dynamic Treatment Effects', fontsize=14, fontweight='bold', pad=12)
        # Move legend
        ax3.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=9.5, 
                  framealpha=0.95, edgecolor='gray')
        
        # ===== ROW 3: Parallel Trends + Love Plot =====
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[3, 1])
        
        self.plot_parallel_trends(ax=ax4)
        ax4.set_title('Parallel Trends Check', fontsize=14, fontweight='bold', pad=12)
        ax4.legend(loc='upper left', fontsize=9.5, framealpha=0.95, edgecolor='gray')
        
        self.plot_love_plot(ax=ax5)
        ax5.set_title('Balance Check', fontsize=14, fontweight='bold', pad=12)
        
        # ===== ROW 4: Cohort Effects + Effect Matrix =====
        ax6 = fig.add_subplot(gs[4, 0])
        ax7 = fig.add_subplot(gs[4, 1])
        
        self.plot_cohort_effects(ax=ax6)
        ax6.set_title('Effects by Treatment Cohort', fontsize=14, fontweight='bold', pad=12)
        
        # Effect matrix (M+0 to M+6, cleaner annotations)
        self._plot_effect_matrix_executive(ax7)
        
        # ===== ROW 5: Heterogeneity =====
        ax8 = fig.add_subplot(gs[5, 0])
        ax9 = fig.add_subplot(gs[5, 1])
        ax9.axis('off')  # Right side empty for balance
        
        if heterogeneity_features:
            self.plot_heterogeneous_effects(heterogeneity_features, ax=ax8)
            ax8.set_title('Heterogeneous Treatment Effects', fontsize=14, fontweight='bold', pad=12)
        else:
            ax8.text(0.5, 0.5, 'Heterogeneous effects analysis\n(specify features to enable)', 
                    ha='center', va='center', fontsize=12, color='#888888', style='italic')
            ax8.set_title('Heterogeneous Treatment Effects', fontsize=14, fontweight='bold', pad=12)
            ax8.axis('off')
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', pad_inches=0.3)
            print(f"✓ Executive report saved to: {save_path}")
        
        plt.show()
    
    def _plot_main_result_executive(self, ax, result):
        """Simplified ATT plot for executive report."""
        # Large bar with CI
        color = '#2E7D32' if result.att > 0 else '#C62828'  # Professional green/red
        
        ax.barh([''], [result.att], color=color, alpha=0.75, height=0.5, 
               edgecolor='white', linewidth=2)
        
        # Error bars
        ax.errorbar(
            result.att, 0,
            xerr=[[result.att - result.ci_lower], [result.ci_upper - result.att]],
            fmt='o', color='#424242', capsize=10, capthick=2.5, markersize=11,
            elinewidth=2.5
        )
        
        # Zero reference line
        ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.6, linewidth=2)
        
        # Clean styling
        ax.set_xlabel(f'{self.analyzer.outcome_col}', fontsize=12, fontweight='bold')
        ax.set_title('Average Treatment Effect', fontsize=14, fontweight='bold', pad=12)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
        
        # Value annotation
        sig = '***' if result.p_value < 0.01 else '**' if result.p_value < 0.05 else '*' if result.p_value < 0.1 else ''
        ax.text(
            result.att, 0.3,
            f'{result.att:,.0f}{sig}',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=color, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.95)
        )
    
    def _plot_effect_matrix_executive(self, ax):
        """Cleaner effect matrix for executive report (M+0 to M+6)."""
        try:
            # effect_matrix() already returns a pivoted DataFrame (cohorts × time)
            heatmap_data = self.analyzer.effect_matrix(pre_periods=0, post_periods=6)
            
            if heatmap_data.empty:
                raise ValueError("No cohort event study data available")
            
            # Sort cohorts descending for better visualization
            heatmap_data = heatmap_data.sort_index(ascending=False)
            
            # Calculate symmetric vmin/vmax for better color distribution
            max_abs_val = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
            vmin, vmax = -max_abs_val, max_abs_val
            
            # Professional diverging color scheme (red-white-green)
            # Using RdYlGn_r for red=negative, green=positive
            cmap = 'RdYlGn'
            
            # Heatmap with cleaner annotations
            sns.heatmap(
                heatmap_data,
                annot=True, fmt='.0f',
                cmap=cmap, 
                center=0,
                vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Effect Size', 'shrink': 0.75, 'aspect': 18},
                linewidths=1.5, linecolor='white',
                ax=ax,
                annot_kws={'fontsize': 8.5, 'fontweight': 'normal'}
            )
            
            # Clean labels
            ax.set_title('Effect Matrix: Cohort × Time', fontsize=14, fontweight='bold', pad=12)
            ax.set_xlabel('Months Relative to Treatment', fontsize=11, fontweight='bold')
            ax.set_ylabel('Treatment Cohort', fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', labelsize=9.5)
            
            # Rotate x labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Effect matrix unavailable\n({str(e)})', 
                   ha='center', va='center', fontsize=10, color='gray')
            ax.set_title('Effect Matrix: Cohort × Time', fontsize=14, fontweight='bold', pad=12)
            ax.axis('off')
    
    # =========================================================================
    # QUICK SUMMARY
    # =========================================================================
    
    def summary_table(self) -> pd.DataFrame:
        """Return DataFrame with summary statistics."""
        result = self.analyzer.estimate_did()
        cohorts = self.analyzer.effect_by_cohort()
        
        return pd.DataFrame([
            {'Metric': 'ATT (Main Effect)', 'Value': f'{result.att:,.0f}'},
            {'Metric': '95% CI Lower', 'Value': f'{result.ci_lower:,.0f}'},
            {'Metric': '95% CI Upper', 'Value': f'{result.ci_upper:,.0f}'},
            {'Metric': 'p-value', 'Value': f'{result.p_value:.4f}'},
            {'Metric': 'Effect as % of Baseline', 'Value': f'{result.att_pct:.2%}' if result.att_pct else 'N/A'},
            {'Metric': 'N Treated', 'Value': f'{result.n_treated:,}'},
            {'Metric': 'N Control', 'Value': f'{result.n_control:,}'},
            {'Metric': 'N Cohorts', 'Value': f'{len(cohorts)}'},
            {'Metric': 'ATT Std Across Cohorts', 'Value': f'{cohorts["att"].std():,.0f}' if len(cohorts) > 0 else 'N/A'},
        ])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

if __name__ == "__main__":
    # Demo with sample data
    import sys
    import os
    sys.path.append(os.path.abspath("."))
    sys.path.append(os.path.abspath("src"))
    
    from sample_data import generate_sample_data
    from src.matching import PropensityMatcher
    from src.causal_analyzer import CausalAnalyzer
    from src.propensity_model import CausalPropensityScore
    
    print("Generating sample data...")
    data = generate_sample_data(
        n_accounts=3000,
        n_treatment_accounts=300,
        treatment_effect=15000,
        noise_scale=3000,
        selection_mode='balanced'
    )
    
    df = data['feature_store']
    df_treat = data['treatment_table']
    
    # Prepare data
    units = df.groupby('cod_conta').first().reset_index()
    df_treat['matching_cohort'] = df_treat['treatment_date'].dt.strftime('%Y%m').astype(int)
    units = units.merge(df_treat[['cod_conta', 'matching_cohort']], on='cod_conta', how='left')
    units['ever_treated'] = units['matching_cohort'].notnull().astype(int)
    
    # Propensity Score
    print("Fitting propensity model...")
    X = units[['cod_conta', 'val_cap_liq', 'potencial', 'n_acessos_hub']].copy()
    y = units['ever_treated']
    ps = CausalPropensityScore(id_col='cod_conta')
    scores = ps.fit_predict_honest(X, y)
    units = units.merge(scores, on='cod_conta')
    
    # Matching
    print("Matching...")
    matcher = PropensityMatcher(use_caliper=True)
    matched = matcher.match(units, 'ever_treated', 'cod_conta', 'propensity_score', 'matching_cohort')
    
    # Create dashboard
    print("Creating dashboard...")
    analyzer = CausalAnalyzer(matched, df, 'val_cap_liq', 'num_ano_mes', 'cod_conta')
    dashboard = CausalDashboard(analyzer)
    
    # Run validation
    analyzer.validate()
    
    # Generate full report
    print("\nGenerating full visual report...")
    dashboard.full_report(
        heterogeneity_features=['segmento'],
        true_effect=15000,
        save_path='causal_analysis_report.png'
    )
    
    # Print summary
    print("\nSummary Table:")
    print(dashboard.summary_table().to_string(index=False))
