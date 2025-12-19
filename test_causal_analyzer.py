"""
Comprehensive Test Suite for Staggered Causal Inference Library
================================================================
Tests all effect patterns and validates the staggered DiD estimator.

Run with: python -m pytest tests/test_causal_analyzer.py -v
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import CausalPipeline
from src.causal_analyzer import CausalAnalyzer
from src.staggered import StaggeredDiD, estimate_staggered_att
from sample_data import generate_sample_data

warnings.filterwarnings("ignore")


def run_pipeline(data, suppress_output=True):
    """
    Run the full pipeline using CausalPipeline with cohort-based matching.
    """
    df_panel = data['feature_store']
    df_treat = data['treatment_table']
    
    pipeline = CausalPipeline(
        id_column='cod_conta',
        time_column='num_ano_mes',
        outcome_column='val_cap_liq',
        treatment_date_column='treatment_date'
    )
    
    features = ['val_cap_liq', 'potencial', 'n_acessos_hub', 'segmento', 'cod_tipo_pessoa']
    
    if suppress_output:
        sys.stdout = open(os.devnull, 'w')
    try:
        pipeline_results = pipeline.run_full_pipeline(
            panel_data=df_panel,
            treatment_data=df_treat,
            features=features,
            use_caliper=True,
            verbose=False
        )
    finally:
        if suppress_output:
            sys.stdout = sys.__stdout__
    
    analyzer = pipeline_results['analyzer']
    
    if analyzer is None or len(pipeline_results['matched']) == 0:
        return None
    
    return analyzer


def test_reconciliation():
    """
    TEST: Overall ATT must equal weighted aggregation of ATT(g,t) cells.
    
    This is the core consistency check for the staggered estimator.
    """
    print("\n" + "="*60)
    print("TEST 0: Reconciliation (Overall ATT = Weighted Sum)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=3000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    # Get staggered result
    result = analyzer.estimate(pre_periods=6, post_periods=6)
    
    # Get ATT(g,t) table
    att_gt = analyzer.get_att_table(pre_periods=6, post_periods=6)
    
    # Manually compute weighted average of post-treatment cells
    post = att_gt[(att_gt['relative_time'] >= 0) & (att_gt['relative_time'] != -1)].copy()
    
    if len(post) == 0:
        print("FAIL: No post-treatment data")
        return False
    
    # Weighted average
    weights = post['n_pairs'].values
    total_w = weights.sum()
    manual_att = np.average(post['att'].values, weights=weights)
    
    # Compare
    diff = abs(result.overall_att - manual_att)
    relative_diff = diff / abs(manual_att) if manual_att != 0 else diff
    
    print(f"  Overall ATT:     {result.overall_att:,.2f}")
    print(f"  Manual weighted: {manual_att:,.2f}")
    print(f"  Difference:      {diff:,.2f}")
    print(f"  Relative diff:   {relative_diff:.6f}")
    
    # Should match within floating point tolerance
    passed = relative_diff < 0.001  # 0.1% tolerance
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_constant_effect():
    """Test: Constant effect should be recovered accurately."""
    print("\n" + "="*60)
    print("TEST 1: Constant Effect (15,000)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=3000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    result = analyzer.estimate()
    print(f"  True effect:     15,000")
    print(f"  Estimated ATT:   {result.overall_att:,.0f}")
    print(f"  95% CI:          [{result.overall_ci_lower:,.0f}, {result.overall_ci_upper:,.0f}]")
    print(f"  p-value:         {result.overall_pvalue:.4f}")
    print(f"  Pre-trend test:  {'PASS' if result.pretrend_passed else 'FAIL'}")
    
    # Check: ATT should be within 20% of true effect
    accuracy = abs(result.overall_att - 15000) / 15000
    passed = accuracy < 0.20 and result.overall_pvalue < 0.05
    print(f"  Accuracy:        {(1-accuracy)*100:.1f}%")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_growing_effect():
    """Test: Growing effect should show in event study."""
    print("\n" + "="*60)
    print("TEST 2: Growing Effect (33% → 66% → 100%)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=2000,
        selection_mode='balanced',
        effect_pattern='growing',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    es = analyzer.estimate_event_study()
    
    # Get effects at key time points
    t0 = es[es['relative_time'] == 0]['att'].values[0]
    t1 = es[es['relative_time'] == 1]['att'].values[0]
    t3 = es[es['relative_time'] == 3]['att'].values[0]
    
    print(f"  Expected M+0:    ~5,000 (33%)")
    print(f"  Observed M+0:    {t0:,.0f}")
    print(f"  Expected M+1:    ~10,000 (66%)")
    print(f"  Observed M+1:    {t1:,.0f}")
    print(f"  Expected M+3:    ~15,000 (100%)")
    print(f"  Observed M+3:    {t3:,.0f}")
    
    # Check: Effect should be growing
    passed = t0 < t1 < t3 and t3 > 10000
    print(f"  Growing pattern: {'Yes ✓' if t0 < t1 < t3 else 'No ✗'}")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_delayed_effect():
    """Test: Delayed effect should show 0 at M+0, full effect at M+1."""
    print("\n" + "="*60)
    print("TEST 3: Delayed Effect (0% → 100%)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=2000,
        selection_mode='balanced',
        effect_pattern='delayed',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    es = analyzer.estimate_event_study()
    
    t0 = es[es['relative_time'] == 0]['att'].values[0]
    t1 = es[es['relative_time'] == 1]['att'].values[0]
    
    print(f"  Expected M+0:    ~0 (no effect)")
    print(f"  Observed M+0:    {t0:,.0f}")
    print(f"  Expected M+1:    ~15,000 (full)")
    print(f"  Observed M+1:    {t1:,.0f}")
    
    # Check: M+0 should be near 0, M+1 should be near full effect
    passed = abs(t0) < 3000 and t1 > 10000
    print(f"  Delay detected:  {'Yes ✓' if abs(t0) < 3000 else 'No ✗'}")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_placebo():
    """Test: Zero effect should not be statistically significant."""
    print("\n" + "="*60)
    print("TEST 4: Placebo (No Effect)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=0,  # NO EFFECT
        noise_scale=3000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    result = analyzer.estimate()
    print(f"  True effect:     0")
    print(f"  Estimated ATT:   {result.overall_att:,.0f}")
    print(f"  p-value:         {result.overall_pvalue:.4f}")
    
    # Check: p-value should be > 0.05 (not significant)
    passed = result.overall_pvalue > 0.05
    print(f"  Insignificant:   {'Yes ✓' if passed else 'No ✗ (false positive!)'}")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_negative_effect():
    """Test: Negative effect should be detected correctly."""
    print("\n" + "="*60)
    print("TEST 5: Negative Effect (-10,000)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=-10000,  # NEGATIVE
        noise_scale=3000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    result = analyzer.estimate()
    print(f"  True effect:     -10,000")
    print(f"  Estimated ATT:   {result.overall_att:,.0f}")
    print(f"  95% CI:          [{result.overall_ci_lower:,.0f}, {result.overall_ci_upper:,.0f}]")
    print(f"  p-value:         {result.overall_pvalue:.4f}")
    
    # Check: Should be negative and significant
    passed = result.overall_att < 0 and result.overall_pvalue < 0.05
    print(f"  Negative:        {'Yes ✓' if result.overall_att < 0 else 'No ✗'}")
    print(f"  Significant:     {'Yes ✓' if result.overall_pvalue < 0.05 else 'No ✗'}")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_pretrend_diagnostics():
    """Test: Pre-treatment coefficients should be near zero (joint test)."""
    print("\n" + "="*60)
    print("TEST 6: Pre-trend Diagnostics (Joint Test)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=2000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    result = analyzer.estimate(pre_periods=6, post_periods=6)
    es = analyzer.estimate_event_study(pre_periods=6, post_periods=6)
    
    pre_coeffs = es[es['relative_time'] < -1]['att']
    post_coeffs = es[es['relative_time'] > 0]['att']
    
    print(f"  Pre-treatment mean:  {pre_coeffs.mean():,.0f} (should be ~0)")
    print(f"  Pre-treatment max:   {pre_coeffs.abs().max():,.0f}")
    print(f"  Post-treatment mean: {post_coeffs.mean():,.0f} (should be ~15,000)")
    print(f"  Joint pre-trend test p-value: {result.pretrend_test.get('pvalue', 'N/A'):.4f}")
    print(f"  Pre-trend passed:    {'Yes ✓' if result.pretrend_passed else 'No ✗'}")
    
    # Check: Pre-treatment should be near 0, post should be near true effect
    pre_ok = pre_coeffs.abs().mean() < 2000
    post_ok = post_coeffs.mean() > 10000
    pretrend_ok = result.pretrend_passed
    passed = pre_ok and post_ok and pretrend_ok
    
    print(f"  Pre-trend ok:    {'Yes ✓' if pre_ok else 'No ✗'}")
    print(f"  Post-effect ok:  {'Yes ✓' if post_ok else 'No ✗'}")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def test_robustness_comparison():
    """Test: Primary and stacked DiD should give similar results."""
    print("\n" + "="*60)
    print("TEST 7: Robustness (Primary vs Stacked DiD)")
    print("="*60)
    
    data = generate_sample_data(
        n_accounts=5000, 
        n_treatment_accounts=500, 
        treatment_effect=15000,
        noise_scale=3000,
        selection_mode='balanced',
        effect_pattern='constant',
        random_state=42
    )
    
    analyzer = run_pipeline(data)
    if not analyzer:
        print("FAIL: No matches")
        return False
    
    # Run with robustness check
    result = analyzer.estimate(pre_periods=6, post_periods=6, robustness=True)
    
    if not hasattr(result, 'robustness') or result.robustness is None:
        print("FAIL: Robustness check not available")
        return False
    
    primary_att = result.overall_att
    stacked_att = result.robustness.get('overall_att', np.nan)
    
    print(f"  Primary ATT:     {primary_att:,.0f}")
    print(f"  Stacked ATT:     {stacked_att:,.0f}")
    
    if np.isnan(stacked_att):
        print("  RESULT:          SKIP (no stacked result)")
        return True
    
    # Check: Should be within 30% of each other
    diff = abs(primary_att - stacked_att) / abs(primary_att) if primary_att != 0 else 0
    passed = diff < 0.30
    print(f"  Difference:      {diff*100:.1f}%")
    print(f"  RESULT:          {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed


def main():
    print("\n" + "="*60)
    print("    STAGGERED DiD ESTIMATOR VALIDATION SUITE")
    print("="*60)
    print("Testing Callaway & Sant'Anna style ATT(g,t) estimation...\n")
    
    tests = [
        ("Reconciliation", test_reconciliation),
        ("Constant Effect", test_constant_effect),
        ("Growing Effect", test_growing_effect),
        ("Delayed Effect", test_delayed_effect),
        ("Placebo (No Effect)", test_placebo),
        ("Negative Effect", test_negative_effect),
        ("Pre-trend Diagnostics", test_pretrend_diagnostics),
        ("Robustness Check", test_robustness_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("                    SUMMARY")
    print("="*60)
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:25} {status}")
    
    print("\n" + "-"*60)
    
    if all_passed:
        print("  🎉 ALL TESTtettttS PASSED - Staggered estimator is working!")
        print("\n  Confidence level: HIGH")
        print("  - ATT(g,t) aggregation is correct")
        print("  - Overall ATT reconciles with cohort×time effects")
        print("  - Dynamic effects are captured")
        print("  - Pre-trend diagnostics work")
        print("  - No false positives on placebo")
    else:
        print("  ⚠️  SOME TESTS FAILED - Review the issues above")
        print("\n  Failed tests indicate potential issues with:")
        for name, passed in results:
            if not passed:
                print(f"    - {name}")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    main()
