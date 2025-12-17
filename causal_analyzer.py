"""
Causal Analyzer Module
======================
Pure computation engine for Difference-in-Differences estimation and causal effect analysis.

This module contains NO plotting logic - it only computes and returns data.
For visualization, use CausalDashboard which consumes CausalAnalyzer's outputs.

Responsibilities:
- Statistical estimation (DID, Event Study, Cohort Effects)
- Data validation and diagnostics
- Panel construction and feature engineering
- Returns: DataFrames, numbers, DIDResult objects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResults


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
# DATA CLASSES
# ============================================================================

@dataclass
class DIDResult:
    """Container for Difference-in-Differences estimation results."""
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    n_observations: int
    att_pct: Optional[float] = None
    pre_parallel_trends_pvalue: Optional[float] = None
    
    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        if self.p_value < 0.01:
            return "***"
        elif self.p_value < 0.05:
            return "**"
        elif self.p_value < 0.1:
            return "*"
        return ""
    
    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant at 5% level."""
        return self.p_value < 0.05
    
    def __repr__(self):
        report = [
            "DID Result:",
            f"  ATT:       {self.att:,.4f} {self.significance_stars}",
        ]
        
        if self.att_pct is not None:
            report.append(f"  ATT (%):   {self.att_pct:.2%} (vs baseline)")
        
        report.extend([
            f"  SE:        {self.se:,.4f}",
            f"  95% CI:    [{self.ci_lower:,.4f}, {self.ci_upper:,.4f}]",
            f"  p-value:   {self.p_value:.4f}",
            f"  N treated: {self.n_treated:,}",
            f"  N control: {self.n_control:,}",
            f"  N obs:     {self.n_observations:,}",
        ])
        
        return "\n".join(report)


# ============================================================================
# MAIN CLASS
# ============================================================================

class CausalAnalyzer:
    """
    Causal effect estimation using Difference-in-Differences.
    
    Works with matched data from PropensityMatcher to estimate ATT.
    
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
        
        # Build panel on initialization
        self._build_panel()
    
    # ========================================================================
    # PANEL CONSTRUCTION
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
    # FORMULA BUILDERS (DRY principle)
    # ========================================================================
    
    def _build_did_formula(self, panel: pd.DataFrame, include_cohort_fe: bool = True) -> str:
        """
        Build DID regression formula with fixed effects.
        
        Formula: Y ~ Treated + Post + Treated×Post + Time_FE + [Cohort_FE]
        """
        parts = [f'{self.outcome_col} ~ is_treated + post + treated_x_post']
        
        # Time fixed effects
        if self.time_col in panel.columns:
            parts.append(f'C({self.time_col})')
        
        # Cohort fixed effects (only if multiple cohorts and requested)
        if include_cohort_fe and 'matching_cohort' in panel.columns:
            n_cohorts = panel['matching_cohort'].dropna().nunique()
            if n_cohorts > 1:
                parts.append('C(matching_cohort)')
        
        return ' + '.join(parts)
    
    def _build_event_study_formula(self, panel: pd.DataFrame, ref_period: int = -1) -> str:
        """
        Build event study regression formula with fixed effects.
        
        Formula: Y ~ Treated + RelTime + Treated×RelTime + Time_FE + Cohort_FE
        """
        parts = [
            f"{self.outcome_col} ~ is_treated",
            f"C(relative_time, Treatment(reference={ref_period}))",
            f"is_treated:C(relative_time, Treatment(reference={ref_period}))",
        ]
        
        # Fixed effects
        if self.time_col in panel.columns:
            parts.append(f"C({self.time_col})")
        
        if 'matching_cohort' in panel.columns:
            n_cohorts = panel['matching_cohort'].dropna().nunique()
            if n_cohorts > 1:
                parts.append("C(matching_cohort)")
        
        return ' + '.join(parts)
    
    # ========================================================================
    # MODEL FITTING (DRY principle)
    # ========================================================================
    
    def _fit_model(self, formula: str, data: pd.DataFrame, cluster_col: Optional[str] = None) -> RegressionResults:
        """
        Fit OLS model with robust or clustered standard errors.
        
        Tries clustered SE first, falls back to HC3 robust SE if clustering fails.
        """
        try:
            return smf.ols(formula, data=data).fit(
                cov_type='cluster',
                cov_kwds={'groups': data[cluster_col or self.id_col]}
            )
        except Exception:
            return smf.ols(formula, data=data).fit(cov_type='HC3')
    
    @staticmethod
    def _extract_coefficient_results(model: RegressionResults, term: str) -> Dict:
        """Extract coefficient, SE, CI, and p-value from model."""
        if term not in model.params:
            return {
                'att': np.nan, 'se': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan
            }
        
        ci = model.conf_int().loc[term]
        return {
            'att': model.params[term],
            'se': model.bse[term],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'p_value': model.pvalues[term]
        }
    
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
        - Parallel trends
        
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
        
        # Parallel trends check
        pt_pvalue = self._test_parallel_trends(panel)
        checks['parallel_trends'], warn = self._check_parallel_trends(pt_pvalue)
        if warn:
            warnings.append(warn)
        
        # Store diagnostics
        self._diagnostics = {
            'n_treated': n_treated,
            'n_control': n_control,
            **time_counts,
            'pct_extreme_outliers': pct_outliers,
            'parallel_trends_pvalue': pt_pvalue,
            'checks': checks,
            'warnings': warnings,
        }
        
        if verbose:
            self._print_validation_report(checks, warnings, n_treated, n_control, time_counts)
        
        return checks
    
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
        """Check parallel trends assumption."""
        if pvalue is None:
            return None, "⚠️ Could not test parallel trends (insufficient pre-treatment data)"
        
        alpha = ValidationThresholds.PARALLEL_TRENDS_ALPHA
        passed = pvalue > alpha
        
        warning = None
        if not passed:
            warning = f"⚠️ Parallel trends violated: p-value = {pvalue:.3f} (should be > {alpha})"
        
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
    
    def _test_parallel_trends(self, panel: pd.DataFrame) -> Optional[float]:
        """Test parallel trends assumption in pre-treatment period."""
        pre_data = panel[panel['relative_time'] < 0].copy()
        
        if len(pre_data) < ValidationThresholds.MIN_PARALLEL_TRENDS_OBS:
            return None
        
        pre_data['time_trend'] = pre_data['relative_time']
        pre_data['treated_x_time'] = pre_data['is_treated'] * pre_data['time_trend']
        
        try:
            formula = f'{self.outcome_col} ~ is_treated + time_trend + treated_x_time'
            model = smf.ols(formula, data=pre_data).fit()
            return model.pvalues.get('treated_x_time')
        except Exception:
            return None
    
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
            {'metric': 'Parallel Trends p-value', 'value': d['parallel_trends_pvalue']},
            {'metric': '% Extreme Outliers', 'value': f"{d['pct_extreme_outliers']:.2%}"},
        ])
    
    # ========================================================================
    # MAIN ESTIMATION METHODS
    # ========================================================================
    
    def estimate_did(
        self,
        pre_periods: Optional[int] = None,
        post_periods: Optional[int] = None,
        cluster_col: Optional[str] = None,
    ) -> DIDResult:
        """
        Estimate Average Treatment Effect on Treated (ATT) using DID.
        
        Uses hybrid specification with time and cohort fixed effects:
        Y = α + β₁·Treated + β₂·Post + β₃·(Treated×Post) + Time_FE + Cohort_FE + ε
        
        ATT = β₃
        
        Parameters
        ----------
        pre_periods : int, optional
            Number of pre-treatment periods to include
        post_periods : int, optional
            Number of post-treatment periods to include
        cluster_col : str, optional
            Column to cluster standard errors on (default: id_col)
        
        Returns
        -------
        DIDResult
            Container with ATT estimate, SE, CI, p-value
        """
        # Filter panel
        panel = self._filter_panel_by_time(self._panel, pre_periods, post_periods)
        panel = panel.dropna(subset=[self.outcome_col])
        
        if len(panel) == 0:
            raise ValueError("No observations after filtering")
        
        # Build and fit model
        formula = self._build_did_formula(panel)
        model = self._fit_model(formula, panel, cluster_col)
        
        # Extract results
        results = self._extract_coefficient_results(model, 'treated_x_post')
        n_treated, n_control = self._count_units(panel)
        
        # Calculate percentage effect
        att_pct = self._calculate_percentage_effect(panel, results['att'])
        
        # Test parallel trends
        pre_pval = self._test_parallel_trends(panel)
        
        return DIDResult(
            att=results['att'],
            se=results['se'],
            ci_lower=results['ci_lower'],
            ci_upper=results['ci_upper'],
            p_value=results['p_value'],
            n_treated=n_treated,
            n_control=n_control,
            n_observations=len(panel),
            att_pct=att_pct,
            pre_parallel_trends_pvalue=pre_pval
        )
    
    @staticmethod
    def _filter_panel_by_time(
        panel: pd.DataFrame,
        pre_periods: Optional[int],
        post_periods: Optional[int]
    ) -> pd.DataFrame:
        """Filter panel to specified time window."""
        filtered = panel.copy()
        
        if pre_periods is not None:
            filtered = filtered[filtered['relative_time'] >= -pre_periods]
        if post_periods is not None:
            filtered = filtered[filtered['relative_time'] <= post_periods]
        
        return filtered
    
    def _calculate_percentage_effect(self, panel: pd.DataFrame, att: float) -> Optional[float]:
        """Calculate ATT as percentage of pre-treatment baseline."""
        pre_treated_mean = panel[
            (panel['is_treated'] == 1) & (panel['post'] == 0)
        ][self.outcome_col].mean()
        
        if pd.notna(pre_treated_mean) and pre_treated_mean != 0:
            return att / pre_treated_mean
        return None
    
    def estimate_event_study(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> pd.DataFrame:
        """
        Estimate dynamic treatment effects (Event Study).
        
        Returns coefficients for each relative time period to visualize
        how effects evolve and validate parallel trends.
        
        Parameters
        ----------
        pre_periods : int
            Number of periods before treatment
        post_periods : int
            Number of periods after treatment
        
        Returns
        -------
        pd.DataFrame
            Columns: relative_time, att, se, ci_lower, ci_upper, p_value
        """
        panel = self._filter_panel_by_time(self._panel, pre_periods, post_periods)
        results = self._run_event_study_regression(panel)
        return pd.DataFrame(results).sort_values('relative_time')
    
    def _run_event_study_regression(
        self,
        panel: pd.DataFrame,
        ref_period: int = -1
    ) -> List[Dict]:
        """Run event study regression and extract coefficients for each period."""
        relative_times = sorted(panel['relative_time'].unique())
        
        # Build and fit model
        formula = self._build_event_study_formula(panel, ref_period)
        model = self._fit_model(formula, panel)
        
        # Extract results for each time period
        results = []
        for t in relative_times:
            if t == ref_period:
                # Reference period: effect = 0 by construction
                results.append({
                    'relative_time': t,
                    'att': 0, 'se': 0, 'ci_lower': 0, 'ci_upper': 0, 'p_value': 1.0
                })
            else:
                term = f"is_treated:C(relative_time, Treatment(reference={ref_period}))[T.{t}]"
                coef_results = self._extract_coefficient_results(model, term)
                results.append({'relative_time': t, **coef_results})
        
        return results
    
    def estimate_cohort_event_study(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> pd.DataFrame:
        """
        Estimate event study separately for each cohort.
        
        Returns
        -------
        pd.DataFrame
            Columns: matching_cohort, relative_time, att, se, ci_lower, ci_upper, p_value
        """
        cohorts = sorted(self._panel['matching_cohort'].unique())
        all_results = []
        
        for cohort in cohorts:
            cohort_panel = self._panel[self._panel['matching_cohort'] == cohort].copy()
            cohort_panel = self._filter_panel_by_time(cohort_panel, pre_periods, post_periods)
            
            if len(cohort_panel) < ValidationThresholds.MIN_COHORT_ES_OBS:
                continue
            
            try:
                results = self._run_event_study_regression(cohort_panel)
                for r in results:
                    r['matching_cohort'] = cohort
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Failed to estimate event study for cohort {cohort}: {e}")
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.DataFrame(all_results).sort_values(['matching_cohort', 'relative_time'])
    
    def effect_by_cohort(self) -> pd.DataFrame:
        """
        Estimate treatment effect separately for each cohort.
        
        Uses a simpler specification (no time FE) to avoid overfitting in small cohorts.
        
        Returns
        -------
        pd.DataFrame
            Columns: cohort, att, se, ci_lower, ci_upper, p_value, n_pairs
        """
        cohorts = sorted(self._panel['matching_cohort'].unique())
        results = []
        
        for cohort in cohorts:
            cohort_data = self._panel[self._panel['matching_cohort'] == cohort]
            
            if len(cohort_data) < ValidationThresholds.MIN_COHORT_OBS:
                continue
            
            try:
                # Use simple DID without time fixed effects to avoid overfitting
                # (cohort-level samples are typically small)
                formula = f'{self.outcome_col} ~ is_treated + post + treated_x_post'
                
                # Try clustered SE, fallback to robust
                try:
                    model = smf.ols(formula, data=cohort_data).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': cohort_data[self.id_col]}
                    )
                except Exception:
                    model = smf.ols(formula, data=cohort_data).fit(cov_type='HC3')
                
                coef_results = self._extract_coefficient_results(model, 'treated_x_post')
                
                results.append({
                    'cohort': cohort,
                    **coef_results,
                    'n_pairs': cohort_data[cohort_data['is_treated'] == 1][self.id_col].nunique(),
                })
            except Exception as e:
                print(f"Warning: Could not estimate for cohort {cohort}: {e}")
        
        return pd.DataFrame(results)
    
    def effect_matrix(
        self,
        pre_periods: int = 6,
        post_periods: int = 6,
        metric: str = 'att'
    ) -> pd.DataFrame:
        """
        Create cohort × relative time matrix of treatment effects.
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods (0 = only post-treatment)
        post_periods : int
            Number of post-treatment periods
        metric : str
            Metric to display: 'att', 'se', or 'p_value'
        
        Returns
        -------
        pd.DataFrame
            Matrix with cohorts as rows, time periods as columns (M-x, M+0, M+1, ...)
        """
        cohort_es = self.estimate_cohort_event_study(max(pre_periods, 1), post_periods)
        
        if cohort_es.empty:
            return pd.DataFrame()
        
        # Filter to post-treatment only if requested
        if pre_periods == 0:
            cohort_es = cohort_es[cohort_es['relative_time'] >= 0]
        
        # Pivot to matrix
        matrix = cohort_es.pivot(
            index='matching_cohort',
            columns='relative_time',
            values=metric
        )
        
        # Format column names (M-3, M-2, M-1, M+0, M+1, ...)
        matrix.columns = [f"M{t:+d}" if t != 0 else "M+0" for t in matrix.columns]
        
        # Sort
        matrix = matrix.sort_index()
        col_order = sorted(matrix.columns, key=lambda x: int(x.replace('M', '').replace('+', '')))
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
            'p_value': self.effect_matrix(pre_periods, post_periods, 'p_value'),
        }
    
    # ========================================================================
    # HETEROGENEITY ANALYSIS
    # ========================================================================
    
    def effect_by_subgroup(
        self,
        filter_col: str,
        filter_values: Optional[List] = None,
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
        
        Returns
        -------
        pd.DataFrame
            Columns: subgroup, att, se, ci_lower, ci_upper, p_value, n_observations
        """
        panel = self._panel.copy()
        panel = self._attach_pre_treatment_feature(panel, filter_col)
        subgroup_col = f'pre_{filter_col}'
        
        if subgroup_col not in panel.columns:
            raise ValueError(f"Column '{filter_col}' not found in outcome data")
        
        if filter_values is None:
            filter_values = panel[subgroup_col].dropna().unique()
        
        results = []
        for value in filter_values:
            subgroup = panel[panel[subgroup_col] == value]
            
            if len(subgroup) < ValidationThresholds.MIN_COHORT_OBS:
                continue
            
            try:
                formula = self._build_did_formula(subgroup)
                model = self._fit_model(formula, subgroup)
                coef_results = self._extract_coefficient_results(model, 'treated_x_post')
                
                results.append({
                    'subgroup': f'{filter_col}={value}',
                    **coef_results,
                    'n_observations': len(subgroup),
                })
            except Exception as e:
                print(f"Warning: Could not estimate for {filter_col}={value}: {e}")
        
        return pd.DataFrame(results)
    
    def _attach_pre_treatment_feature(
        self,
        panel: pd.DataFrame,
        feature_col: str,
    ) -> pd.DataFrame:
        """
        Attach feature measured at treatment (matching) month.
        
        Creates a new column `pre_{feature_col}` with pre-treatment values.
        """
        if feature_col not in self.outcome_df.columns:
            raise ValueError(f"Column '{feature_col}' not found in outcome data")
        
        # Create lookup: (id, time) -> feature value
        feature_lookup = self.outcome_df[[self.id_col, self.time_col, feature_col]].copy()
        
        # Get unique (id, cohort) pairs
        id_cohort = panel[[self.id_col, 'matching_cohort']].drop_duplicates()
        
        # Join to get feature at matching month
        feature_at_cohort = id_cohort.merge(
            feature_lookup,
            left_on=[self.id_col, 'matching_cohort'],
            right_on=[self.id_col, self.time_col],
            how='left'
        )[[self.id_col, feature_col]]
        
        # Map to panel
        feature_map = feature_at_cohort.set_index(self.id_col)[feature_col].to_dict()
        panel[f'pre_{feature_col}'] = panel[self.id_col].map(feature_map)
        
        return panel
    
    def explore_subgroups(self, features: List[str]) -> pd.DataFrame:
        """
        Automated heterogeneity analysis across multiple features.
        
        Parameters
        ----------
        features : List[str]
            List of columns to explore (e.g., ['segmento', 'region'])
        
        Returns
        -------
        pd.DataFrame
            Subgroup effects sorted by magnitude
        """
        all_results = []
        
        for feature in features:
            if feature not in self._panel.columns:
                print(f"Skipping {feature}: not in panel columns")
                continue
            
            print(f"Analyzing heterogeneity by: {feature}")
            try:
                res = self.effect_by_subgroup(filter_col=feature)
                res['feature'] = feature
                res['subgroup_name'] = res['subgroup']
                all_results.append(res)
            except Exception as e:
                print(f"Error analyzing {feature}: {e}")
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine and sort by effect magnitude
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.sort_values('att', key=abs, ascending=False)
        
        cols = ['feature', 'subgroup_name', 'att', 'p_value', 'ci_lower', 'ci_upper', 'n_observations']
        return final_df[cols]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def effect_over_time(self) -> pd.DataFrame:
        """
        Calculate mean outcome by treatment status and relative time.
        
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
            Complete analysis results including main DID, cohort effects, and panel summary
        """
        did_result = self.estimate_did()
        cohort_effects = self.effect_by_cohort()
        time_effects = self.effect_over_time()
        
        return {
            'main_result': did_result,
            'cohort_effects': cohort_effects,
            'time_evolution': time_effects,
            'panel_summary': {
                'n_treated': self._panel[self._panel['is_treated'] == 1][self.id_col].nunique(),
                'n_control': self._panel[self._panel['is_treated'] == 0][self.id_col].nunique(),
                'n_observations': len(self._panel),
                'time_range': (self._panel['relative_time'].min(), self._panel['relative_time'].max()),
                'n_cohorts': self._panel['matching_cohort'].nunique(),
            }
        }
    
    def get_panel(self) -> pd.DataFrame:
        """Return copy of analysis panel for custom analysis."""
        return self._panel.copy()

