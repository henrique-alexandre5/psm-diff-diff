"""
Causal Analyzer Module
======================
Pure computation engine for staggered Difference-in-Differences estimation.

This module uses a Callaway & Sant'Anna (2021) style approach:
- Computes ATT(g,t) for each cohort and relative time
- Aggregates to overall ATT with proper weighting
- Provides staggered-safe event studies
- Includes pre-trend diagnostics

For visualization, use CausalDashboard which consumes CausalAnalyzer's outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .staggered import StaggeredDiD, StackedDiD, StaggeredATTResult


# ============================================================================
# CONSTANTS
# ============================================================================

class ValidationThresholds:
    """Thresholds for validation checks."""
    MIN_SAMPLE_SIZE = 30
    MIN_COHORT_OBS = 20
    MIN_COHORT_ES_OBS = 50
    MIN_PARALLEL_TRENDS_OBS = 10
    MAX_OUTLIER_PCT = 0.01
    PARALLEL_TRENDS_ALPHA = 0.05
    OUTLIER_IQR_MULTIPLIER = 3


# ============================================================================
# MAIN CLASS
# ============================================================================

class CausalAnalyzer:
    """
    Causal effect estimation using staggered Difference-in-Differences.
    
    Uses Callaway & Sant'Anna (2021) style ATT(g,t) estimation with
    matched pairs from PropensityMatcher.
    
    Parameters
    ----------
    matched_df : pd.DataFrame
        Output from PropensityMatcher.match() containing:
        treated_id, control_id, ps_treated, ps_control, distance, matching_cohort
    outcome_df : pd.DataFrame
        Panel data with outcomes, containing:
        id_col, time_col, outcome_col, and optionally other features
    outcome_col : str
        Name of the outcome column
    time_col : str
        Name of the time column (YYYYMM format)
    id_col : str
        Name of the account/entity ID column
    """
    
    def __init__(
        self,
        matched_df: pd.DataFrame,
        outcome_df: pd.DataFrame,
        outcome_col: str = 'val_cap_liq',
        time_col: str = 'num_ano_mes',
        id_col: str = 'cod_conta',
    ):
        self.matched_df = matched_df.copy()
        self.outcome_df = outcome_df.copy()
        self.outcome_col = outcome_col
        self.time_col = time_col
        self.id_col = id_col
        
        # Processed data cache
        self._panel: Optional[pd.DataFrame] = None
        self._diagnostics: Dict = {}
        self._staggered_result: Optional[StaggeredATTResult] = None
        
        # Initialize staggered estimator
        self._staggered_estimator = StaggeredDiD(
            matched_df=self.matched_df,
            outcome_df=self.outcome_df,
            outcome_col=self.outcome_col,
            time_col=self.time_col,
            id_col=self.id_col,
        )
        
        # Build panel on initialization (for compatibility with dashboard)
        self._build_panel()
    
    # ========================================================================
    # PANEL CONSTRUCTION (for visualization compatibility)
    # ========================================================================
    
    def _build_panel(self) -> None:
        """Build the analysis panel from matched pairs and outcome data."""
        # Extract treated and control units with their cohorts
        treated = self.matched_df[['treated_id', 'matching_cohort']].copy()
        treated.columns = [self.id_col, 'matching_cohort']
        treated['is_treated'] = 1
        
        control = self.matched_df[['control_id', 'matching_cohort']].copy()
        control.columns = [self.id_col, 'matching_cohort']
        control['is_treated'] = 0
        
        # Combine and merge with outcomes
        panel = pd.concat([treated, control], ignore_index=True)
        panel = panel.merge(self.outcome_df, on=self.id_col, how='inner')
        
        # Create time variables
        panel['relative_time'] = self._calculate_month_diff(
            panel['matching_cohort'], panel[self.time_col]
        )
        panel['post'] = (panel['relative_time'] >= 0).astype(int)
        panel['treated_x_post'] = panel['is_treated'] * panel['post']
        
        self._panel = panel
    
    @staticmethod
    def _calculate_month_diff(start: pd.Series, end: pd.Series) -> pd.Series:
        """Calculate month difference between two YYYYMM series."""
        start, end = start.astype(int), end.astype(int)
        start_y, start_m = start // 100, start % 100
        end_y, end_m = end // 100, end % 100
        return (end_y - start_y) * 12 + (end_m - start_m)
    
    # ========================================================================
    # VALIDATION & DIAGNOSTICS
    # ========================================================================
    
    def validate(self, verbose: bool = True) -> Dict[str, Optional[bool]]:
        """
        Run diagnostic checks to validate data quality and assumptions.
        
        Checks:
        - Sufficient sample size
        - Balanced time coverage
        - No extreme outliers
        - Pre-trend test (joint test on leads)
        
        Returns
        -------
        Dict[str, Optional[bool]]
            Check names and pass/fail status (None = N/A)
        """
        panel = self._panel
        checks = {}
        warnings = []
        
        # Sample size check
        n_treated, n_control = self._count_units(panel)
        checks['sufficient_sample'], warn = self._check_sample_size(n_treated, n_control)
        if warn:
            warnings.append(warn)
        
        # Time coverage check
        time_counts = self._count_time_coverage(panel)
        checks['balanced_time'], warn = self._check_time_balance(time_counts)
        if warn:
            warnings.append(warn)
        
        # Outliers check
        pct_outliers = self._calculate_outlier_percentage(panel[self.outcome_col])
        checks['no_extreme_outliers'], warn = self._check_outliers(pct_outliers)
        if warn:
            warnings.append(warn)
        
        # Pre-trend joint test (staggered-safe)
        pretrend = self._test_pretrend_joint()
        pt_pvalue = pretrend.get('pvalue') if pretrend else None
        checks['pretrend_test'], warn = self._check_parallel_trends(pt_pvalue)
        if warn:
            warnings.append(warn)
        
        # Store diagnostics
        self._diagnostics = {
            'n_treated': n_treated,
            'n_control': n_control,
            **time_counts,
            'pct_extreme_outliers': pct_outliers,
            'pretrend_pvalue': pt_pvalue,
            'pretrend_test': pretrend,
            'checks': checks,
            'warnings': warnings,
        }
        
        if verbose:
            self._print_validation_report(checks, warnings, n_treated, n_control, time_counts)
        
        return checks
    
    def _test_pretrend_joint(self) -> Optional[Dict]:
        """Run joint pre-trend test using staggered estimator."""
        try:
            # Compute ATT(g,t) if not already computed
            att_gt = self._staggered_estimator.compute_att_gt(pre_periods=6, post_periods=1)
            return self._staggered_estimator.pretrend_joint_test(att_gt)
        except Exception:
            return None
    
    def _count_units(self, panel: pd.DataFrame) -> Tuple[int, int]:
        """Count treated and control units."""
        n_treated = panel[panel['is_treated'] == 1][self.id_col].nunique()
        n_control = panel[panel['is_treated'] == 0][self.id_col].nunique()
        return n_treated, n_control
    
    def _count_time_coverage(self, panel: pd.DataFrame) -> Dict[str, int]:
        """Count observations by treatment status and time period."""
        return {
            'pre_treated_obs': len(panel[(panel['is_treated'] == 1) & (panel['post'] == 0)]),
            'post_treated_obs': len(panel[(panel['is_treated'] == 1) & (panel['post'] == 1)]),
            'pre_control_obs': len(panel[(panel['is_treated'] == 0) & (panel['post'] == 0)]),
            'post_control_obs': len(panel[(panel['is_treated'] == 0) & (panel['post'] == 1)]),
        }
    
    @staticmethod
    def _check_sample_size(n_treated: int, n_control: int) -> Tuple[bool, Optional[str]]:
        """Check if sample sizes are sufficient."""
        threshold = ValidationThresholds.MIN_SAMPLE_SIZE
        is_sufficient = n_treated >= threshold and n_control >= threshold
        
        warning = None
        if not is_sufficient:
            warning = f"⚠️ Low sample: {n_treated} treated, {n_control} control. Recommend {threshold}+ each."
        
        return is_sufficient, warning
    
    @staticmethod
    def _check_time_balance(time_counts: Dict[str, int]) -> Tuple[bool, Optional[str]]:
        """Check if all groups have pre and post observations."""
        is_balanced = all(count > 0 for count in time_counts.values())
        
        warning = None
        if not is_balanced:
            warning = "⚠️ Unbalanced time: Missing pre or post observations for some groups."
        
        return is_balanced, warning
    
    @staticmethod
    def _calculate_outlier_percentage(values: pd.Series) -> float:
        """Calculate percentage of extreme outliers."""
        values = values.dropna()
        if len(values) == 0:
            return 0.0
        q1, q99 = values.quantile([0.01, 0.99])
        iqr = q99 - q1
        multiplier = ValidationThresholds.OUTLIER_IQR_MULTIPLIER
        
        extreme_low = (values < q1 - multiplier * iqr).sum()
        extreme_high = (values > q99 + multiplier * iqr).sum()
        
        return (extreme_low + extreme_high) / len(values) if len(values) > 0 else 0
    
    @staticmethod
    def _check_outliers(pct_outliers: float) -> Tuple[bool, Optional[str]]:
        """Check if outlier percentage is acceptable."""
        threshold = ValidationThresholds.MAX_OUTLIER_PCT
        is_acceptable = pct_outliers < threshold
        
        warning = None
        if not is_acceptable:
            warning = f"⚠️ Extreme outliers detected: {pct_outliers:.1%} of observations."
        
        return is_acceptable, warning
    
    @staticmethod
    def _check_parallel_trends(pvalue: Optional[float]) -> Tuple[Optional[bool], Optional[str]]:
        """Check pre-trend test result."""
        if pvalue is None:
            return None, "⚠️ Could not run pre-trend test (insufficient pre-treatment data)"
        
        alpha = ValidationThresholds.PARALLEL_TRENDS_ALPHA
        passed = pvalue > alpha
        
        warning = None
        if not passed:
            warning = f"⚠️ Pre-trend test failed: p-value = {pvalue:.3f} (should be > {alpha})"
        
        return passed, warning
    
    def _print_validation_report(
        self,
        checks: Dict,
        warnings: List[str],
        n_treated: int,
        n_control: int,
        time_counts: Dict[str, int]
    ) -> None:
        """Print formatted validation report."""
        print("=" * 50)
        print("CAUSAL ANALYZER VALIDATION")
        print("=" * 50)
        print(f"  Treated units:     {n_treated}")
        print(f"  Control units:     {n_control}")
        print(f"  Pre-treatment obs: {time_counts['pre_treated_obs'] + time_counts['pre_control_obs']}")
        print(f"  Post-treatment obs: {time_counts['post_treated_obs'] + time_counts['post_control_obs']}")
        print()
        
        for check_name, passed in checks.items():
            if passed is None:
                status = "⚠️ N/A"
            elif passed:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  {check_name:25} {status}")
        
        if warnings:
            print("\nWarnings:")
            for warn in warnings:
                print(f"  {warn}")
        else:
            print("\n✓ All checks passed - Ready for analysis!")
        
        print("=" * 50)
    
    def diagnostic_summary(self) -> pd.DataFrame:
        """Return DataFrame with key diagnostic metrics."""
        if not self._diagnostics:
            self.validate(verbose=False)
        
        d = self._diagnostics
        return pd.DataFrame([
            {'metric': 'N Treated Units', 'value': d['n_treated']},
            {'metric': 'N Control Units', 'value': d['n_control']},
            {'metric': 'Pre-Treatment Obs (Treated)', 'value': d['pre_treated_obs']},
            {'metric': 'Post-Treatment Obs (Treated)', 'value': d['post_treated_obs']},
            {'metric': 'Pre-Treatment Obs (Control)', 'value': d['pre_control_obs']},
            {'metric': 'Post-Treatment Obs (Control)', 'value': d['post_control_obs']},
            {'metric': 'Pre-trend Test p-value', 'value': d.get('pretrend_pvalue')},
            {'metric': '% Extreme Outliers', 'value': f"{d['pct_extreme_outliers']:.2%}"},
        ])
    
    # ========================================================================
    # MAIN ESTIMATION METHODS (Staggered DiD)
    # ========================================================================
    
    def estimate(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
        robustness: bool = False,
    ) -> StaggeredATTResult:
        """
        Estimate staggered ATT using Callaway & Sant'Anna style approach.
        
        Computes:
        - ATT(g,t) for each cohort and relative time
        - Overall ATT (weighted aggregation of post-treatment cells)
        - Event study (aggregated across cohorts)
        - Pre-trend diagnostic tests
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods to include
        post_periods : int
            Number of post-treatment periods to include
        robustness : bool
            If True, also run stacked DiD as robustness check
        
        Returns
        -------
        StaggeredATTResult
            Complete estimation results
        """
        result = self._staggered_estimator.estimate(
            pre_periods=pre_periods,
            post_periods=post_periods,
        )
        self._staggered_result = result
        
        if robustness:
            # Run stacked DiD robustness check
            stacked = StackedDiD(
                matched_df=self.matched_df,
                outcome_df=self.outcome_df,
                outcome_col=self.outcome_col,
                time_col=self.time_col,
                id_col=self.id_col,
            )
            robustness_result = stacked.estimate(
                pre_periods=pre_periods,
                post_periods=post_periods,
            )
            # Store robustness result in the main result
            result.robustness = robustness_result
        
        return result
    
    def estimate_event_study(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> pd.DataFrame:
        """
        Estimate staggered-safe event study.
        
        Aggregates ATT(g,t) across cohorts for each relative time,
        avoiding the TWFE biases under heterogeneous effects.
        
        Parameters
        ----------
        pre_periods : int
            Number of periods before treatment
        post_periods : int
            Number of periods after treatment
        
        Returns
        -------
        pd.DataFrame
            Columns: relative_time, att, se, ci_lower, ci_upper, pvalue, n_pairs, n_cohorts
        """
        att_gt = self._staggered_estimator.compute_att_gt(pre_periods, post_periods)
        return self._staggered_estimator.compute_event_study(att_gt)
    
    def effect_by_cohort(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> pd.DataFrame:
        """
        Get treatment effect by cohort (aggregated across post-treatment periods).
        
        Returns
        -------
        pd.DataFrame
            Columns: cohort, att, se, ci_lower, ci_upper, pvalue, n_pairs
        """
        att_gt = self._staggered_estimator.compute_att_gt(pre_periods, post_periods)
        
        # Aggregate by cohort (post-treatment only)
        post = att_gt[att_gt['relative_time'] >= 0].copy()
        
        def agg_cohort(group):
            weights = group['n_pairs'].values
            total_w = weights.sum()
            if total_w == 0:
                return pd.Series({'att': np.nan, 'se': np.nan, 'n_pairs': 0})
            
            att_mean = np.average(group['att'].values, weights=weights)
            w_norm = weights / total_w
            se_pooled = np.sqrt(np.sum((w_norm ** 2) * (group['se'].values ** 2)))
            
            return pd.Series({
                'att': att_mean,
                'se': se_pooled,
                'n_pairs': int(total_w / len(group))  # Average pairs per period
            })
        
        cohort_effects = post.groupby('cohort').apply(agg_cohort).reset_index()
        cohort_effects['ci_lower'] = cohort_effects['att'] - 1.96 * cohort_effects['se']
        cohort_effects['ci_upper'] = cohort_effects['att'] + 1.96 * cohort_effects['se']
        from scipy import stats
        cohort_effects['pvalue'] = 2 * (1 - stats.norm.cdf(
            np.abs(cohort_effects['att'] / cohort_effects['se'].replace(0, np.nan))
        ))
        
        return cohort_effects
    
    def effect_matrix(
        self,
        pre_periods: int = 6,
        post_periods: int = 6,
        metric: str = 'att'
    ) -> pd.DataFrame:
        """
        Create cohort × relative time matrix of treatment effects.
        
        This is the core ATT(g,t) table that reconciles with overall ATT.
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        metric : str
            Metric to display: 'att', 'se', or 'pvalue'
        
        Returns
        -------
        pd.DataFrame
            Matrix with cohorts as rows, relative time as columns
        """
        att_gt = self._staggered_estimator.compute_att_gt(pre_periods, post_periods)
        
        if att_gt.empty:
            return pd.DataFrame()
        
        # Map metric name
        metric_map = {'att': 'att', 'se': 'se', 'p_value': 'pvalue', 'pvalue': 'pvalue'}
        metric_col = metric_map.get(metric, 'att')
        
        # Pivot to matrix
        matrix = att_gt.pivot(
            index='cohort',
            columns='relative_time',
            values=metric_col
        )
        
        # Format column names (e=-3, e=-2, e=-1, e=0, e=+1, ...)
        matrix.columns = [f"e={int(c):+d}" if c != 0 else "e=0" for c in matrix.columns]
        matrix = matrix.sort_index()
        
        # Sort columns by event time
        def sort_key(x):
            return int(x.replace('e=', '').replace('+', ''))
        col_order = sorted(matrix.columns, key=sort_key)
        
        return matrix[col_order]
    
    def effect_matrix_summary(
        self,
        pre_periods: int = 3,
        post_periods: int = 6
    ) -> Dict[str, pd.DataFrame]:
        """Return multiple effect matrices (ATT, SE, p-value)."""
        return {
            'att': self.effect_matrix(pre_periods, post_periods, 'att'),
            'se': self.effect_matrix(pre_periods, post_periods, 'se'),
            'pvalue': self.effect_matrix(pre_periods, post_periods, 'pvalue'),
        }
    
    # ========================================================================
    # HETEROGENEITY ANALYSIS
    # ========================================================================
    
    def effect_by_subgroup(
        self,
        filter_col: str,
        filter_values: Optional[List] = None,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> pd.DataFrame:
        """
        Estimate treatment effect for subpopulations.
        
        Uses pre-treatment values (at matching month) to avoid post-treatment bias.
        
        Parameters
        ----------
        filter_col : str
            Column to segment by
        filter_values : list, optional
            Specific values to analyze (default: all unique values)
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        
        Returns
        -------
        pd.DataFrame
            Columns: subgroup, att, se, ci_lower, ci_upper, pvalue, n_pairs
        """
        # Get pre-treatment feature values
        if filter_col not in self.outcome_df.columns:
            raise ValueError(f"Column '{filter_col}' not found in outcome data")
        
        # Map IDs to their pre-treatment feature values
        feature_lookup = self._get_pretreatment_features(filter_col)
        
        if filter_values is None:
            filter_values = feature_lookup[filter_col].dropna().unique()
        
        results = []
        for value in filter_values:
            # Get IDs with this feature value
            ids_with_value = set(feature_lookup[feature_lookup[filter_col] == value][self.id_col])
            
            # Filter matched pairs
            subgroup_matched = self.matched_df[
                self.matched_df['treated_id'].isin(ids_with_value)
            ].copy()
            
            if len(subgroup_matched) < ValidationThresholds.MIN_COHORT_OBS:
                continue
            
            try:
                # Run staggered estimation on subgroup
                subgroup_estimator = StaggeredDiD(
                    matched_df=subgroup_matched,
                    outcome_df=self.outcome_df,
                    outcome_col=self.outcome_col,
                    time_col=self.time_col,
                    id_col=self.id_col,
                )
                sub_result = subgroup_estimator.estimate(pre_periods, post_periods)
                
                results.append({
                    'subgroup': f'{filter_col}={value}',
                    'att': sub_result.overall_att,
                    'se': sub_result.overall_se,
                    'ci_lower': sub_result.overall_ci_lower,
                    'ci_upper': sub_result.overall_ci_upper,
                    'pvalue': sub_result.overall_pvalue,
                    'n_pairs': sub_result.n_pairs,
                })
            except Exception as e:
                print(f"Warning: Could not estimate for {filter_col}={value}: {e}")
        
        return pd.DataFrame(results)
    
    def _get_pretreatment_features(self, feature_col: str) -> pd.DataFrame:
        """Get feature values at pre-treatment (matching cohort) time."""
        # Get (id, cohort) pairs from matched data
        treated_cohorts = self.matched_df[['treated_id', 'matching_cohort']].copy()
        treated_cohorts.columns = [self.id_col, 'cohort']
        
        # Join with outcome data to get feature at cohort time
        feature_data = self.outcome_df[[self.id_col, self.time_col, feature_col]].copy()
        
        merged = treated_cohorts.merge(
            feature_data,
            left_on=[self.id_col, 'cohort'],
            right_on=[self.id_col, self.time_col],
            how='left'
        )
        
        return merged[[self.id_col, feature_col]].drop_duplicates()
    
    def explore_subgroups(self, features: List[str]) -> pd.DataFrame:
        """
        Automated heterogeneity analysis across multiple features.
        
        Parameters
        ----------
        features : List[str]
            List of columns to explore
        
        Returns
        -------
        pd.DataFrame
            Subgroup effects sorted by magnitude
        """
        all_results = []
        
        for feature in features:
            if feature not in self.outcome_df.columns:
                print(f"Skipping {feature}: not in outcome data columns")
                continue
            
            print(f"Analyzing heterogeneity by: {feature}")
            try:
                res = self.effect_by_subgroup(filter_col=feature)
                res['feature'] = feature
                all_results.append(res)
            except Exception as e:
                print(f"Error analyzing {feature}: {e}")
        
        if not all_results:
            return pd.DataFrame()
        
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.sort_values('att', key=abs, ascending=False)
        
        return final_df
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def effect_over_time(self) -> pd.DataFrame:
        """
        Calculate mean outcome by treatment status and relative time.
        
        Useful for parallel trends visualization.
        
        Returns
        -------
        pd.DataFrame
            Columns: relative_time, is_treated, mean, std, count, se, ci_lower, ci_upper, group
        """
        panel = self._panel.copy()
        
        agg = panel.groupby(['relative_time', 'is_treated'])[self.outcome_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        agg['se'] = agg['std'] / np.sqrt(agg['count'])
        agg['ci_lower'] = agg['mean'] - 1.96 * agg['se']
        agg['ci_upper'] = agg['mean'] + 1.96 * agg['se']
        agg['group'] = agg['is_treated'].map({1: 'Treated', 0: 'Control'})
        
        return agg
    
    def summary_report(self) -> Dict:
        """
        Generate comprehensive summary report.
        
        Returns
        -------
        dict
            Complete analysis results
        """
        result = self.estimate()
        cohort_effects = self.effect_by_cohort()
        time_effects = self.effect_over_time()
        
        return {
            'main_result': result,
            'cohort_effects': cohort_effects,
            'time_evolution': time_effects,
            'att_table': result.att_table,
            'event_study': result.event_study,
            'pretrend_test': result.pretrend_test,
            'panel_summary': {
                'n_treated': result.n_treated,
                'n_control': result.n_control,
                'n_pairs': result.n_pairs,
                'n_observations': len(self._panel),
                'time_range': (self._panel['relative_time'].min(), self._panel['relative_time'].max()),
                'n_cohorts': self._panel['matching_cohort'].nunique(),
            }
        }
    
    def get_panel(self) -> pd.DataFrame:
        """Return copy of analysis panel for custom analysis."""
        return self._panel.copy()
    
    def get_att_table(self, pre_periods: int = 6, post_periods: int = 12) -> pd.DataFrame:
        """
        Get the raw ATT(g,t) table (not pivoted).
        
        Returns
        -------
        pd.DataFrame
            Columns: cohort, relative_time, att, se, ci_lower, ci_upper, pvalue, n_pairs
        """
        return self._staggered_estimator.compute_att_gt(pre_periods, post_periods)
