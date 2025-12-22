"""
Staggered Difference-in-Differences Estimator
==============================================
Implements Callaway & Sant'Anna (2021) style group-time ATT estimation
for matched pairs in staggered adoption settings.

This module provides:
- ATT(g,t): Treatment effects by cohort (g) and relative time (t)
- Aggregated overall ATT (weighted by number of matched pairs)
- Event study aggregation across cohorts
- Pre-trend diagnostics with joint statistical tests

References:
- Callaway & Sant'Anna (2021): Group-time ATT for staggered DiD
- Sun & Abraham (2021): Interaction-weighted event studies
- Roth et al.: Pre-trend testing and diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class StaggeredATTResult:
    """
    Container for staggered DiD estimation results.
    
    Attributes
    ----------
    overall_att : float
        Aggregated ATT across all cohorts and post-treatment periods
    overall_se : float
        Standard error of the overall ATT
    overall_ci_lower : float
        Lower bound of 95% CI for overall ATT
    overall_ci_upper : float
        Upper bound of 95% CI for overall ATT
    overall_pvalue : float
        P-value for overall ATT
    att_table : pd.DataFrame
        Cohort × relative_time matrix of ATT estimates
    event_study : pd.DataFrame
        Event study aggregated across cohorts
    pretrend_test : dict
        Pre-trend joint test results
    n_treated : int
        Number of unique treated units
    n_control : int
        Number of unique control units
    n_pairs : int
        Number of matched pairs
    att_pct : float, optional
        ATT as percentage of pre-treatment baseline
    """
    overall_att: float
    overall_se: float
    overall_ci_lower: float
    overall_ci_upper: float
    overall_pvalue: float
    att_table: pd.DataFrame
    event_study: pd.DataFrame
    pretrend_test: Dict
    n_treated: int
    n_control: int
    n_pairs: int
    att_pct: Optional[float] = None
    
    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        if self.overall_pvalue < 0.01:
            return "***"
        elif self.overall_pvalue < 0.05:
            return "**"
        elif self.overall_pvalue < 0.1:
            return "*"
        return ""
    
    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant at 5% level."""
        return self.overall_pvalue < 0.05
    
    @property
    def pretrend_passed(self) -> bool:
        """Check if pre-trend test passed (p > 0.05)."""
        if self.pretrend_test and 'pvalue' in self.pretrend_test:
            return self.pretrend_test['pvalue'] > 0.05
        return True  # No test = assume passed
    
    def __repr__(self):
        lines = [
            "=" * 50,
            "STAGGERED DiD RESULT (Callaway-Sant'Anna Style)",
            "=" * 50,
            f"  Overall ATT:  {self.overall_att:,.4f} {self.significance_stars}",
        ]
        
        if self.att_pct is not None:
            lines.append(f"  ATT (%):      {self.att_pct:.2%} (vs baseline)")
        
        lines.extend([
            f"  SE:           {self.overall_se:,.4f}",
            f"  95% CI:       [{self.overall_ci_lower:,.4f}, {self.overall_ci_upper:,.4f}]",
            f"  p-value:      {self.overall_pvalue:.4f}",
            "",
            f"  N treated:    {self.n_treated:,}",
            f"  N control:    {self.n_control:,}",
            f"  N pairs:      {self.n_pairs:,}",
            "",
            f"  Pre-trend test: {'PASS' if self.pretrend_passed else 'FAIL'}"
            f" (p={self.pretrend_test.get('pvalue', 'N/A'):.4f})" if self.pretrend_test else "",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# =============================================================================
# CORE STAGGERED ESTIMATOR
# =============================================================================

class StaggeredDiD:
    """
    Staggered Difference-in-Differences estimator using matched pairs.
    
    Computes ATT(g,t) for each cohort g and relative time t, then aggregates
    to an overall ATT weighted by the number of contributing matched pairs.
    
    Parameters
    ----------
    matched_df : pd.DataFrame
        Matched pairs from PropensityMatcher with columns:
        treated_id, control_id, matching_cohort, [ps_treated, ps_control, distance]
    outcome_df : pd.DataFrame
        Panel data with outcomes over time
    outcome_col : str
        Name of outcome variable column
    time_col : str
        Name of time column (YYYYMM format)
    id_col : str
        Name of unit ID column
    use_all_preperiods_baseline : bool
        If True (default), uses average of pre-treatment periods as baseline.
        If False, uses only the e=-1 period as baseline.
    baseline_max_periods : int, optional
        Maximum number of pre-treatment periods to use for baseline.
        Default is 12 (e=-12 to e=-1). This limits how far back the baseline
        calculation goes, which is recommended because:
        - Avoids structural breaks in distant past
        - Focuses on period closest to treatment
        - Reduces noise from very old data
        Set to None to use ALL available pre-treatment periods.
    """
    
    def __init__(
        self,
        matched_df: pd.DataFrame,
        outcome_df: pd.DataFrame,
        outcome_col: str = 'val_cap_liq',
        time_col: str = 'num_ano_mes',
        id_col: str = 'cod_conta',
        use_all_preperiods_baseline: bool = True,
        baseline_max_periods: Optional[int] = 12,
    ):
        self.matched_df = matched_df.copy()
        self.outcome_df = outcome_df.copy()
        self.outcome_col = outcome_col
        self.time_col = time_col
        self.id_col = id_col
        self.use_all_preperiods_baseline = use_all_preperiods_baseline
        self.baseline_max_periods = baseline_max_periods
        
        # Ensure matching_cohort column exists
        if 'matching_cohort' not in self.matched_df.columns:
            raise ValueError("matched_df must contain 'matching_cohort' column")
        
        # Computed results (cached)
        self._pair_level_did: Optional[pd.DataFrame] = None
        self._att_gt: Optional[pd.DataFrame] = None
    
    @staticmethod
    def _month_diff(start: Union[int, pd.Series], end: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Calculate month difference between YYYYMM values."""
        if isinstance(start, pd.Series):
            start, end = start.astype(int), end.astype(int)
        else:
            start, end = int(start), int(end)
        start_y, start_m = start // 100, start % 100
        end_y, end_m = end // 100, end % 100
        return (end_y - start_y) * 12 + (end_m - start_m)
    
    def _build_pair_outcomes(self) -> pd.DataFrame:
        """
        Build pair-level outcome data with relative time.
        
        Returns DataFrame with one row per (pair, time) with:
        - pair_id, cohort, relative_time
        - y_treated, y_control (outcome values)
        """
        pairs = self.matched_df[['treated_id', 'control_id', 'matching_cohort']].copy()
        pairs['pair_id'] = range(len(pairs))
        
        # Get outcomes for treated units
        treated_outcomes = pairs.merge(
            self.outcome_df[[self.id_col, self.time_col, self.outcome_col]],
            left_on='treated_id',
            right_on=self.id_col,
            how='inner'
        )
        treated_outcomes = treated_outcomes.rename(columns={self.outcome_col: 'y_treated'})
        treated_outcomes = treated_outcomes.drop(columns=[self.id_col], errors='ignore')
        
        # Get outcomes for control units
        control_outcomes = pairs.merge(
            self.outcome_df[[self.id_col, self.time_col, self.outcome_col]],
            left_on='control_id',
            right_on=self.id_col,
            how='inner'
        )
        control_outcomes = control_outcomes.rename(columns={self.outcome_col: 'y_control'})
        control_outcomes = control_outcomes.drop(columns=[self.id_col], errors='ignore')
        
        # Merge treated and control outcomes
        pair_outcomes = treated_outcomes.merge(
            control_outcomes[['pair_id', self.time_col, 'y_control']],
            on=['pair_id', self.time_col],
            how='inner'
        )
        
        # Calculate relative time
        pair_outcomes['relative_time'] = self._month_diff(
            pair_outcomes['matching_cohort'],
            pair_outcomes[self.time_col]
        )
        
        return pair_outcomes
    
    def _compute_pair_level_did(self, pair_outcomes: pd.DataFrame, baseline_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Compute pair-level DiD for each (pair, relative_time).
        
        For each pair, computes:
        DID = (Y_treated(t) - Y_treated(baseline)) - (Y_control(t) - Y_control(baseline))
        
        Baseline can be:
        - Average of all pre-treatment periods (use_all_preperiods_baseline=True, default)
        - Single period at e=-1 (use_all_preperiods_baseline=False)
        
        Parameters
        ----------
        pair_outcomes : pd.DataFrame
            Pair-level outcome data with relative_time
        baseline_periods : int, optional
            Number of pre-treatment periods to use for baseline averaging.
            If None (default), uses ALL available pre-treatment periods.
            This is separate from display pre_periods to ensure consistent estimates.
        
        Using all pre-treatment periods is more robust as it:
        - Reduces noise from any single period
        - Captures average pre-treatment level more accurately
        - Is less sensitive to outliers
        
        Returns DataFrame with pair_id, cohort, relative_time, did_effect
        """
        if self.use_all_preperiods_baseline:
            # Use average of pre-treatment periods as baseline
            # Determine the window for baseline calculation
            effective_baseline_periods = baseline_periods or self.baseline_max_periods
            
            if effective_baseline_periods is None:
                # Use ALL available pre-treatment periods
                pre_data = pair_outcomes[pair_outcomes['relative_time'] < 0]
            else:
                # Use specified number of pre-periods (e.g., e=-12 to e=-1)
                pre_data = pair_outcomes[
                    (pair_outcomes['relative_time'] < 0) & 
                    (pair_outcomes['relative_time'] >= -effective_baseline_periods)
                ]
            baseline = pre_data.groupby('pair_id').agg(
                y_treated_baseline=('y_treated', 'mean'),
                y_control_baseline=('y_control', 'mean'),
                n_pre_periods=('y_treated', 'count')
            ).reset_index()
        else:
            # Use single period (e=-1) as baseline
            baseline = pair_outcomes[pair_outcomes['relative_time'] == -1][
                ['pair_id', 'y_treated', 'y_control']
            ].copy()
            baseline = baseline.rename(columns={
                'y_treated': 'y_treated_baseline',
                'y_control': 'y_control_baseline'
            })
            baseline['n_pre_periods'] = 1
        
        # Merge baseline with all periods
        did_data = pair_outcomes.merge(baseline, on='pair_id', how='inner')
        
        # Compute DiD: (treated_t - treated_baseline) - (control_t - control_baseline)
        did_data['did_effect'] = (
            (did_data['y_treated'] - did_data['y_treated_baseline']) -
            (did_data['y_control'] - did_data['y_control_baseline'])
        )
        
        # Keep relevant columns
        result = did_data[[
            'pair_id', 'treated_id', 'control_id', 'matching_cohort',
            'relative_time', 'did_effect', 'y_treated', 'y_control',
            'y_treated_baseline', 'y_control_baseline'
        ]].copy()
        
        return result
    
    def compute_att_gt(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
        baseline_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute ATT(g,t) for each cohort and relative time.
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods to DISPLAY (for output/visualization)
        post_periods : int
            Number of post-treatment periods to DISPLAY
        baseline_periods : int, optional
            Number of pre-treatment periods to use for baseline calculation.
            If None (default), uses ALL available pre-treatment periods.
            This ensures consistent estimates regardless of display settings.
        
        Returns
        -------
        pd.DataFrame
            Columns: cohort, relative_time, att, se, ci_lower, ci_upper, pvalue, n_pairs
        
        Notes
        -----
        The baseline is computed INDEPENDENTLY of pre_periods to ensure estimates
        are consistent. When use_all_preperiods_baseline=True (default), the baseline
        is the average of ALL pre-treatment periods, providing the most robust estimate.
        
        Changing pre_periods only affects which periods are displayed, NOT the estimates.
        """
        # Build pair-level data
        pair_outcomes = self._build_pair_outcomes()
        # Baseline uses all available pre-periods (or baseline_periods if specified)
        # This is separate from display pre_periods to ensure consistent estimates
        pair_did = self._compute_pair_level_did(pair_outcomes, baseline_periods=baseline_periods)
        self._pair_level_did = pair_did
        
        # Filter to time window (include all pre-periods and post-periods)
        pair_did = pair_did[
            (pair_did['relative_time'] >= -pre_periods) &
            (pair_did['relative_time'] <= post_periods)
        ].copy()
        
        # Aggregate by (cohort, relative_time)
        att_gt = pair_did.groupby(['matching_cohort', 'relative_time']).agg(
            att=('did_effect', 'mean'),
            se=('did_effect', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            n_pairs=('pair_id', 'nunique')
        ).reset_index()
        
        att_gt = att_gt.rename(columns={'matching_cohort': 'cohort'})
        
        # Compute CI and p-value
        att_gt['ci_lower'] = att_gt['att'] - 1.96 * att_gt['se']
        att_gt['ci_upper'] = att_gt['att'] + 1.96 * att_gt['se']
        att_gt['pvalue'] = 2 * (1 - stats.norm.cdf(np.abs(att_gt['att'] / att_gt['se'].replace(0, np.nan))))
        
        att_gt = att_gt.sort_values(['cohort', 'relative_time']).reset_index(drop=True)
        
        self._att_gt = att_gt
        return att_gt
    
    def compute_event_study(self, att_gt: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate ATT(g,t) to event study (average across cohorts for each relative time).
        
        Weights each cohort by number of matched pairs.
        
        Parameters
        ----------
        att_gt : pd.DataFrame, optional
            Pre-computed ATT(g,t). If None, uses cached result.
        
        Returns
        -------
        pd.DataFrame
            Columns: relative_time, att, se, ci_lower, ci_upper, pvalue, n_pairs, n_cohorts
        """
        if att_gt is None:
            att_gt = self._att_gt
        if att_gt is None:
            raise ValueError("Must call compute_att_gt() first or provide att_gt")
        
        # Weighted aggregation by relative_time
        def weighted_agg(group):
            weights = group['n_pairs'].values
            total_w = weights.sum()
            
            if total_w == 0:
                return pd.Series({
                    'att': np.nan,
                    'se': np.nan,
                    'n_pairs': 0,
                    'n_cohorts': 0
                })
            
            # Weighted mean
            att_mean = np.average(group['att'].values, weights=weights)
            
            # Weighted SE (using influence function approach)
            # SE = sqrt(sum(w_i^2 * se_i^2)) / sum(w_i)
            w_norm = weights / total_w
            se_pooled = np.sqrt(np.sum((w_norm ** 2) * (group['se'].values ** 2)))
            
            return pd.Series({
                'att': att_mean,
                'se': se_pooled,
                'n_pairs': int(total_w),
                'n_cohorts': len(group)
            })
        
        event_study = att_gt.groupby('relative_time').apply(weighted_agg).reset_index()
        
        # Compute CI and p-value
        event_study['ci_lower'] = event_study['att'] - 1.96 * event_study['se']
        event_study['ci_upper'] = event_study['att'] + 1.96 * event_study['se']
        event_study['pvalue'] = 2 * (1 - stats.norm.cdf(
            np.abs(event_study['att'] / event_study['se'].replace(0, np.nan))
        ))
        
        event_study = event_study.sort_values('relative_time').reset_index(drop=True)
        
        return event_study
    
    def compute_overall_att(self, att_gt: Optional[pd.DataFrame] = None) -> Dict:
        """
        Aggregate ATT(g,t) to overall ATT (post-treatment periods only).
        
        Weights each cell by number of matched pairs.
        
        Parameters
        ----------
        att_gt : pd.DataFrame, optional
            Pre-computed ATT(g,t). If None, uses cached result.
        
        Returns
        -------
        dict
            Keys: att, se, ci_lower, ci_upper, pvalue, n_pairs, n_cells
        """
        if att_gt is None:
            att_gt = self._att_gt
        if att_gt is None:
            raise ValueError("Must call compute_att_gt() first or provide att_gt")
        
        # Filter to post-treatment only (relative_time >= 0)
        post = att_gt[att_gt['relative_time'] >= 0].copy()
        
        if len(post) == 0:
            return {
                'att': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'pvalue': np.nan,
                'n_pairs': 0,
                'n_cells': 0
            }
        
        # Weighted aggregation
        weights = post['n_pairs'].values
        total_w = weights.sum()
        
        if total_w == 0:
            return {
                'att': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'pvalue': np.nan,
                'n_pairs': 0,
                'n_cells': 0
            }
        
        # Weighted mean
        att_overall = np.average(post['att'].values, weights=weights)
        
        # Weighted SE
        w_norm = weights / total_w
        se_overall = np.sqrt(np.sum((w_norm ** 2) * (post['se'].values ** 2)))
        
        # CI and p-value
        ci_lower = att_overall - 1.96 * se_overall
        ci_upper = att_overall + 1.96 * se_overall
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(att_overall / se_overall))) if se_overall > 0 else np.nan
        
        return {
            'att': att_overall,
            'se': se_overall,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pvalue': pvalue,
            'n_pairs': int(total_w),
            'n_cells': len(post)
        }
    
    def pretrend_joint_test(
        self,
        att_gt: Optional[pd.DataFrame] = None,
        lead_periods: Optional[int] = None
    ) -> Dict:
        """
        Joint test that pre-treatment effects are zero (Roth-style diagnostic).
        
        Tests H0: ATT(e) = 0 for all e < 0 using Wald test.
        
        Parameters
        ----------
        att_gt : pd.DataFrame, optional
            Pre-computed ATT(g,t). If None, uses cached result.
        lead_periods : int, optional
            Number of lead periods to include in test. If None, uses all available.
        
        Returns
        -------
        dict
            Keys: statistic, pvalue, df, lead_estimates, passed
        """
        if att_gt is None:
            att_gt = self._att_gt
        if att_gt is None:
            raise ValueError("Must call compute_att_gt() first or provide att_gt")
        
        # Get event study for pre-treatment periods
        event_study = self.compute_event_study(att_gt)
        pre = event_study[event_study['relative_time'] < 0].copy()
        
        if lead_periods is not None:
            pre = pre[pre['relative_time'] >= -lead_periods]
        
        if len(pre) == 0 or pre['se'].isna().all():
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'df': 0,
                'lead_estimates': [],
                'passed': True  # No pre-periods = assume passed
            }
        
        # Wald test: sum((att/se)^2) ~ chi-squared(df)
        valid = pre['se'] > 0
        if not valid.any():
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'df': 0,
                'lead_estimates': pre.to_dict('records'),
                'passed': True
            }
        
        z_scores = pre.loc[valid, 'att'] / pre.loc[valid, 'se']
        wald_stat = (z_scores ** 2).sum()
        df = len(z_scores)
        pvalue = 1 - stats.chi2.cdf(wald_stat, df)
        
        return {
            'statistic': float(wald_stat),
            'pvalue': float(pvalue),
            'df': df,
            'lead_estimates': pre.to_dict('records'),
            'passed': pvalue > 0.05
        }
    
    def estimate(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
    ) -> StaggeredATTResult:
        """
        Run full staggered DiD estimation.
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        
        Returns
        -------
        StaggeredATTResult
            Complete estimation results
        """
        # Compute ATT(g,t)
        att_gt = self.compute_att_gt(pre_periods, post_periods)
        
        # Compute event study
        event_study = self.compute_event_study(att_gt)
        
        # Compute overall ATT
        overall = self.compute_overall_att(att_gt)
        
        # Pre-trend test
        pretrend = self.pretrend_joint_test(att_gt)
        
        # Pivot ATT table for cohort × time matrix
        att_table = att_gt.pivot(
            index='cohort',
            columns='relative_time',
            values='att'
        )
        # Format column names
        att_table.columns = [f"e={int(c):+d}" if c != 0 else "e=0" for c in att_table.columns]
        att_table = att_table.sort_index()
        
        # Count units
        n_treated = self.matched_df['treated_id'].nunique()
        n_control = self.matched_df['control_id'].nunique()
        n_pairs = len(self.matched_df)
        
        # Compute ATT as % of baseline
        att_pct = None
        if self._pair_level_did is not None:
            pre_mean = self._pair_level_did[
                self._pair_level_did['relative_time'] < 0
            ]['y_treated'].mean()
            if pd.notna(pre_mean) and pre_mean != 0:
                att_pct = overall['att'] / pre_mean
        
        return StaggeredATTResult(
            overall_att=overall['att'],
            overall_se=overall['se'],
            overall_ci_lower=overall['ci_lower'],
            overall_ci_upper=overall['ci_upper'],
            overall_pvalue=overall['pvalue'],
            att_table=att_table,
            event_study=event_study,
            pretrend_test=pretrend,
            n_treated=n_treated,
            n_control=n_control,
            n_pairs=n_pairs,
            att_pct=att_pct,
        )


# =============================================================================
# STACKED DiD (ROBUSTNESS ESTIMATOR)
# =============================================================================

class StackedDiD:
    """
    Stacked Difference-in-Differences estimator for robustness checks.
    
    Constructs cohort-specific 2×2 DiDs in windows around each treatment cohort,
    then stacks and computes event-time effects.
    
    This is a simpler, more transparent estimator that serves as a robustness
    check against the primary ATT(g,t) estimator.
    
    Parameters
    ----------
    matched_df : pd.DataFrame
        Matched pairs from PropensityMatcher
    outcome_df : pd.DataFrame
        Panel data with outcomes
    outcome_col : str
        Name of outcome variable
    time_col : str
        Name of time column
    id_col : str
        Name of unit ID column
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
    
    @staticmethod
    def _month_diff(start, end):
        """Calculate month difference between YYYYMM values."""
        # Handle Series inputs
        if isinstance(end, pd.Series):
            end = end.astype(int)
        else:
            end = int(end)
        
        if isinstance(start, pd.Series):
            start = start.astype(int)
        else:
            start = int(start)
        
        start_y, start_m = start // 100, start % 100
        end_y, end_m = end // 100, end % 100
        return (end_y - start_y) * 12 + (end_m - start_m)
    
    def estimate(
        self,
        pre_periods: int = 6,
        post_periods: int = 12,
        use_all_preperiods_baseline: bool = True,
    ) -> Dict:
        """
        Compute stacked DiD event study for robustness.
        
        For each cohort, constructs a clean comparison using:
        - Baseline: Average of all pre-treatment periods (default) or single e=-1 period
        - Post-periods: 0 to post_periods
        
        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        use_all_preperiods_baseline : bool
            If True (default), uses average of all pre-treatment periods as baseline.
            If False, uses only the e=-1 period.
        
        Returns
        -------
        dict
            Keys: overall_att, event_study, n_stacks
        """
        cohorts = self.matched_df['matching_cohort'].unique()
        stacked_results = []
        
        for cohort in cohorts:
            cohort_pairs = self.matched_df[self.matched_df['matching_cohort'] == cohort]
            
            for _, pair in cohort_pairs.iterrows():
                treated_id = pair['treated_id']
                control_id = pair['control_id']
                
                # Get outcomes for this pair
                treated_data = self.outcome_df[self.outcome_df[self.id_col] == treated_id].copy()
                control_data = self.outcome_df[self.outcome_df[self.id_col] == control_id].copy()
                
                if len(treated_data) == 0 or len(control_data) == 0:
                    continue
                
                # Calculate relative time
                treated_data['relative_time'] = self._month_diff(cohort, treated_data[self.time_col])
                control_data['relative_time'] = self._month_diff(cohort, control_data[self.time_col])
                
                # Compute baseline values
                if use_all_preperiods_baseline:
                    # Use average of ALL available pre-treatment periods
                    # (not limited by pre_periods to ensure consistent estimates)
                    treated_pre = treated_data[treated_data['relative_time'] < 0][self.outcome_col]
                    control_pre = control_data[control_data['relative_time'] < 0][self.outcome_col]
                    
                    if len(treated_pre) == 0 or len(control_pre) == 0:
                        continue
                    
                    y_t_base = treated_pre.mean()
                    y_c_base = control_pre.mean()
                else:
                    # Use single e=-1 period
                    treated_baseline = treated_data[
                        treated_data['relative_time'] == -1
                    ][self.outcome_col]
                    control_baseline = control_data[
                        control_data['relative_time'] == -1
                    ][self.outcome_col]
                    
                    if len(treated_baseline) == 0 or len(control_baseline) == 0:
                        continue
                    
                    y_t_base = treated_baseline.iloc[0]
                    y_c_base = control_baseline.iloc[0]
                
                # Compute DiD for each period (including pre-treatment for diagnostics)
                for e in range(-pre_periods, post_periods + 1):
                    y_t = treated_data[treated_data['relative_time'] == e][self.outcome_col]
                    y_c = control_data[control_data['relative_time'] == e][self.outcome_col]
                    
                    if len(y_t) == 0 or len(y_c) == 0:
                        continue
                    
                    did = (y_t.iloc[0] - y_t_base) - (y_c.iloc[0] - y_c_base)
                    
                    stacked_results.append({
                        'cohort': cohort,
                        'relative_time': e,
                        'did_effect': did
                    })
        
        if len(stacked_results) == 0:
            return {
                'overall_att': np.nan,
                'overall_se': np.nan,
                'event_study': pd.DataFrame(),
                'n_stacks': 0
            }
        
        stacked_df = pd.DataFrame(stacked_results)
        
        # Aggregate to event study
        event_study = stacked_df.groupby('relative_time').agg(
            att=('did_effect', 'mean'),
            se=('did_effect', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            n_obs=('did_effect', 'count')
        ).reset_index()
        
        event_study['ci_lower'] = event_study['att'] - 1.96 * event_study['se']
        event_study['ci_upper'] = event_study['att'] + 1.96 * event_study['se']
        event_study['pvalue'] = 2 * (1 - stats.norm.cdf(
            np.abs(event_study['att'] / event_study['se'].replace(0, np.nan))
        ))
        event_study = event_study.sort_values('relative_time').reset_index(drop=True)
        
        # Compute overall ATT (post-treatment only)
        post = stacked_df[stacked_df['relative_time'] >= 0]
        if len(post) > 0:
            overall_att = post['did_effect'].mean()
            overall_se = post['did_effect'].std() / np.sqrt(len(post))
        else:
            overall_att = np.nan
            overall_se = np.nan
        
        return {
            'overall_att': overall_att,
            'overall_se': overall_se,
            'event_study': event_study,
            'n_stacks': len(stacked_df)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def estimate_staggered_att(
    matched_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_col: str = 'val_cap_liq',
    time_col: str = 'num_ano_mes',
    id_col: str = 'cod_conta',
    pre_periods: int = 6,
    post_periods: int = 12,
    use_all_preperiods_baseline: bool = True,
    robustness: bool = False,
) -> Union[StaggeredATTResult, Tuple[StaggeredATTResult, Dict]]:
    """
    Convenience function to estimate staggered ATT.
    
    Parameters
    ----------
    matched_df : pd.DataFrame
        Matched pairs
    outcome_df : pd.DataFrame
        Panel outcomes
    outcome_col : str
        Outcome variable name
    time_col : str
        Time column name
    id_col : str
        ID column name
    pre_periods : int
        Number of pre-treatment periods
    post_periods : int
        Number of post-treatment periods
    use_all_preperiods_baseline : bool
        If True (default), uses average of all pre-treatment periods as baseline.
        This is more robust as it reduces noise from any single period.
    robustness : bool
        Whether to also run stacked DiD robustness check
    
    Returns
    -------
    StaggeredATTResult or (StaggeredATTResult, dict)
        Primary result, and optionally robustness results
    """
    estimator = StaggeredDiD(
        matched_df=matched_df,
        outcome_df=outcome_df,
        outcome_col=outcome_col,
        time_col=time_col,
        id_col=id_col,
        use_all_preperiods_baseline=use_all_preperiods_baseline,
    )
    
    result = estimator.estimate(pre_periods=pre_periods, post_periods=post_periods)
    
    if not robustness:
        return result
    
    # Run stacked DiD for robustness
    stacked = StackedDiD(
        matched_df=matched_df,
        outcome_df=outcome_df,
        outcome_col=outcome_col,
        time_col=time_col,
        id_col=id_col,
    )
    robustness_result = stacked.estimate(
        pre_periods=pre_periods,
        post_periods=post_periods,
        use_all_preperiods_baseline=use_all_preperiods_baseline,
    )
    
    return result, robustness_result

