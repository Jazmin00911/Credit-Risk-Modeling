# Independent Model Validation Memo

**Probability of Default (PD) Model: Retail Credit Scorecard and Gradient-Boosting Challenger**

| Item | Detail |
|---|---|
| Model | `creditrisk` v0.1.0. Champion: WoE-transformed logistic regression. Challenger: monotone-constrained XGBoost (depth 2, 200 trees) |
| Intended use | Rank-ordering and accept/decline recommendation for retail unsecured credit applications; PD input to a portfolio expected-loss illustration |
| Development data | Public German Credit dataset as committed in `data/raw/`, n = 1,000, default rate 30.0%, no origination dates |
| Validation date | 30 October 2026 |
| Validation scope | Conceptual soundness, data quality, outcomes analysis, challenger benchmarking, sensitivity and stress, monitoring design |
| Reference frameworks | Federal Reserve SR 11-7 (2011); EBA/GL/2017/16 (PD estimation); EBA/GL/2020/06 (loan origination and monitoring). Applied as reference frameworks. No compliance assertion is made |
| Reproducibility | `python -m creditrisk.train` regenerates the development and holdout figures (`reports/metrics.json`); `python -m creditrisk.validate` regenerates all other figures (`reports/validation_results.json`). All random seeds are fixed (42 and offsets) |
| Environment of record | Python 3.12.3, scikit-learn 1.8.0, XGBoost 3.4.1, pandas 3.0.2, NumPy 2.4.4, SciPy 1.17.1, SHAP 0.54.1. Other library versions can shift figures in the third decimal, most likely for XGBoost-based results |

> **Independence statement.** The validator and the model developer are the same individual. SR 11-7 requires effective challenge from parties independent of model development, and that condition is not met. This memo applies a second-line validation framework as an exercise. The tests run in a separate module (`validate.py`) that re-estimates both models from source data through the production code path and does not load the stored artefacts. The independence limitation remains, and no approval authority is implied.

---

## 1. Executive Summary and Model Rating

### 1.1 Overall rating

| Rating | Scope |
|---|---|
| **SATISFACTORY WITH CONDITIONS** | **Approved use:** demonstration and research. **Production credit decisioning: not approved** pending closure of findings F1 and F2 |

| Rating scale | Definition |
|---|---|
| Satisfactory | No High findings; Medium findings have agreed remediation dates |
| Satisfactory with Conditions | Use permitted within stated restrictions; High findings block expansion of use |
| Unsatisfactory | Model output cannot be relied on for the stated purpose |

**Basis.** The champion scorecard meets the validator's criteria for discrimination, calibration and overfitting (Section 4.1). The evidence base does not support reliance beyond demonstration: 1,000 observations, no time dimension and a single 250-row holdout whose Gini confidence interval spans 0.27 (F1). One characteristic, absence of a checking account, carries a third of the model's discriminatory power without a documented economic rationale (F2). The challenger adds no measurable discrimination and fails calibration, so the scorecard is retained as champion.

### 1.2 Findings by severity

| Severity | Count | Findings | Definition |
|---|---|---|---|
| **High** | 2 | F1, F2 | Prevents reliance on model output for production use. Closure required before any such use |
| **Medium** | 4 | F3, F4, F5, F6 | Material weakness. Use permitted with compensating controls; remediation dated |
| **Low** | 2 | F7, F8 | Control or documentation deficiency with no direct effect on current outputs |

All findings are **Open** at the validation date.

### 1.3 Headline results (champion scorecard) against validator criteria

| Metric | Result | Criterion | Outcome |
|---|---|---|---|
| Gini, pooled out-of-fold (n = 750) | 0.493 (95% CI 0.420 to 0.564) | ≥ 0.40 | Pass |
| Gini, holdout (n = 250) | 0.514 (95% CI 0.375 to 0.640) | ≥ 0.40 | Inconclusive: CI lower bound below 0.40 |
| KS, out-of-fold / holdout | 0.400 / 0.453 | ≥ 0.30 | Pass |
| Brier score, repeated CV | 0.177 vs 0.210 no-skill | Below no-skill | Pass (skill score 0.158) |
| Hosmer-Lemeshow | χ²(8) = 8.09, p = 0.425 | p > 0.05 | Pass |
| Calibration slope | 0.86 | 0.80 to 1.20 | Pass, near lower bound |
| In-sample minus CV Gini | 0.046 | ≤ 0.05 | Pass, marginal |
| Score PSI, OOF vs holdout | 0.043 | < 0.10 | Pass, at sampling noise floor |
| Challenger uplift in Gini | -0.001 (CI -0.018 to +0.014) | CI lower bound > 0 | Complexity not justified |

Criteria are set by the validator for this memo. They are internal conventions, not regulatory thresholds.

### 1.4 Areas of conformance

- WoE is fitted inside each cross-validation fold. Fitting it on the full sample first inflates out-of-fold Gini by 0.041 (scorecard) and 0.020 (XGBoost), so the pipeline structure removes a measurable bias (Section 2.3).
- No class re-weighting is applied. Re-weighting raised mean PD from 0.301 to 0.449 against a 0.300 observed rate with no gain in rank ordering (Section 2.4).
- Input collinearity is low: maximum variance inflation factor 1.28, maximum pairwise WoE correlation 0.43.
- Both models enforce monotone risk direction on WoE inputs, and a challenger exists.
- Findings F1 to F8 are reproducible from `validate.py`; production code carries 14 tests and CI.

---

## 2. Conceptual Soundness and Methodology Review

### 2.1 Design

The target is `Risk == "bad"` (y = 1). The source does not state a default definition, days-past-due threshold or outcome window. Inputs are seven characteristics: Job, Credit amount, Duration, Housing, Saving accounts, Checking account, Purpose. Age and Sex are excluded from the model and used only in the fairness audit.

Each characteristic is transformed to Weight of Evidence, WoE = ln(%good / %bad), using five equal-frequency bins for high-cardinality numerics and category levels otherwise, with additive smoothing of 0.5. The champion is an L2-penalised logistic regression (C = 0.5) on WoE. The challenger is XGBoost with a -1 monotone constraint on every WoE input.

### 2.2 Feature selection and coefficients

| Characteristic | IV | IV strength | Coefficient | Expected sign | Leave-one-out change in Gini |
|---|---|---|---|---|---|
| Checking account | 0.659 | Above 0.50 review level | -0.877 | Negative | **-0.155** |
| Duration | 0.213 | Medium | -0.834 | Negative | -0.035 |
| Saving accounts | 0.188 | Medium | -0.707 | Negative | -0.018 |
| Credit amount | 0.092 | Weak | -0.334 | Negative | -0.002 |
| Housing | 0.084 | Weak | -0.535 | Negative | -0.003 |
| Purpose | 0.068 | Weak | -0.513 | Negative | +0.004 |
| Job | 0.009 | Below noise floor | **+0.148** | Negative | +0.000 |

WoE is defined so that higher values indicate lower risk, so every coefficient should be negative. Job violates this. Its IV of 0.009 is below the expected null IV of about 0.014 for four bins at this sample size (approximately (k-1)(1/G + 1/B) with G = 700 goods and B = 300 bads), and removing it leaves Gini unchanged. Removing Purpose raises Gini slightly. Leave-one-out figures are means of three repeated 5-fold runs (baseline 0.490).

**Assessment.** Two characteristics (Job, Purpose) contribute no out-of-sample discrimination. Checking account dominates the model (F2). Job's coefficient sign fails the conceptual-soundness check (F5).

### 2.3 Encoding and leakage

WoE is a supervised (target-based) encoding. If fitted before the train/validation split, each held-out label influences its own feature. The leakage test fits WoE on the full sample, then cross-validates the downstream estimator:

| Model | Gini, WoE fitted on full data (leaky) | Gini, WoE fitted within folds | Optimism |
|---|---|---|---|
| Scorecard | 0.522 | 0.480 | **+0.041** |
| XGBoost | 0.499 | 0.479 | **+0.020** |

Production code fits WoE inside a scikit-learn `Pipeline`, so cross-validation and holdout results carry no such optimism. Unseen categories at scoring time receive a neutral WoE of 0.0 without a log entry (F8).

**Binning limitation.** WoE binning makes PD a step function of each numeric input. For a median applicant, scorecard PD rises from 0.391 (Duration of 12 months or less) to 0.507 (16 to 24 months) to 0.642 (32 months or more) and is flat above 32 months. An 11.6 percentage-point discontinuity occurs between 12 and 16 months, and the model has no sensitivity to duration beyond 32.

### 2.4 Class imbalance

Prevalence is 30.0%, a moderate imbalance. The model applies no re-weighting, and cost asymmetry is handled at the decision stage through the cutoff. The alternative was tested on the challenger with the positive class weighted by the class ratio (2.33):

| Configuration | Mean PD (observed 0.300) | Calibration slope | Brier | Gini | Approval rate at Bayes cutoff |
|---|---|---|---|---|---|
| No re-weighting (as built) | 0.301 | 0.74 | 0.179 | 0.479 | 37.8% |
| Positive class × 2.33 | 0.449 | 0.72 | 0.207 | 0.477 | 17.1% |

Re-weighting leaves rank ordering unchanged (Gini 0.479 vs 0.477), moves mean PD 15 points above the observed rate, and degrades Brier to within 0.003 of the no-skill value of 0.210. The design choice is supported.

### 2.5 Model selection and regularisation

- **Scorecard.** Out-of-fold Gini is 0.479 to 0.480 across C from 0.05 to 100. The penalty is not binding because WoE has already reduced the model to eight parameters.
- **XGBoost.** Out-of-fold Gini is flat across tree depth (0.483, 0.479, 0.480, 0.480, 0.480 for depths 1, 2, 3, 4, 6) while in-sample Gini rises from 0.547 to 0.588. Depth 1, which is additive in WoE, performs as well as any deeper configuration. The data show no detectable interaction effects.

### 2.6 Design limitations

| Item | Assessment |
|---|---|
| Overfitting | In-sample minus CV Gini: scorecard 0.046, XGBoost 0.084. Challenger fixed-parameter, untuned |
| Leakage | Controlled by pipeline structure (2.3). No residual leakage path identified |
| Protected attributes | Excluded as inputs. Disparate outcomes persist (F4); single-input proxy correlations with Age are low (maximum \|r\| = 0.11) |
| Exposure blindness | PD is insensitive to loan size: doubling Credit amount moves mean PD from 0.300 to 0.311. Cost matrix counts applicants and ignores exposure (F3) |
| Default definition | Undocumented (F1) |

---

## 3. Data Quality and Preprocessing Validation

### 3.1 Dataset

n = 1,000, no duplicate rows, seven model inputs plus audit attributes. Provenance is not documented in the repository: collection period, sampling design, currency (Credit amount range 250 to 18,424) and category relabelling are unknown.

### 3.2 Missing data

Saving accounts has 183 missing values and Checking account 394 (478 rows have at least one). The raw file encodes these as the string `NA`, which `pandas.read_csv` converts to `NaN` by default. In the source coding the state means no account (checking) or unknown/no account (savings). It is a structural category, not a random omission.

| Treatment | Rows retained | Default rate of retained rows | Assessment |
|---|---|---|---|
| Complete-case deletion (alternative considered) | 522 (52.2%) | 44.3% | Rejected |
| Explicit "none" category (current) | 1,000 | 30.0% | Adopted |

Rows deleted under complete-case deletion defaulted at 14.4%, against 44.3% for retained rows. The deleted rows are the lowest-risk segment. Complete-case deletion selects on a regressor, so conditional estimates remain consistent on the retained support, but the model has no support for no-account applicants and its intercept is anchored to a 44% base rate instead of 30%. Mean-mode imputation would place no-account applicants (11.7% default) into the "little" level (49.3% default) and was not considered acceptable.

The current treatment is appropriate. The residual issue is the mixed meaning of the savings "none" state and the undocumented source-system definition (F2).

### 3.3 Outliers

| Variable | Median | 99th percentile | Maximum | Skewness | Above Tukey fence (1.5 × IQR) |
|---|---|---|---|---|---|
| Credit amount | 2,320 | 14,180 | 18,424 | 1.95 | 72 (7.2%) |
| Duration (months) | 18 | 60 | 72 | 1.09 | 70 (7.0%) |

No winsorisation is applied. Quantile binning places all tail observations in the top bin, so extreme values do not influence WoE estimates. The cost is information loss and saturation above the top bin (2.3, 6.2).

### 3.4 Sparse levels

Four levels have fewer than 30 observations: Job = 0 (22), Purpose = repairs (22), Purpose = domestic appliances (12), Purpose = vacation/others (12). With smoothing of 0.5, WoE for these bins is unstable. No pooling rule exists (F5).

### 3.5 Transformation audit

| Check | Result |
|---|---|
| WoE reproduces textbook formula on a hand-computed case | Pass (unit test) |
| Unseen category maps to 0.0 | Pass; not logged (F8) |
| NaN in a numeric input assigned to a separate bin | Pass (unit test) |
| Bin edges at ±∞ so out-of-range values fall in end bins | Pass; silent extrapolation (F8) |
| Non-monotone bad rate across ordered bins | Duration: 0.212, 0.181, 0.322, 0.333, 0.480. Credit amount: 0.303, 0.241, 0.270, 0.260, 0.425 |

Bad rate is non-monotone in the lowest Duration bins and across Credit amount. This is permitted by WoE but should be justified or coarse-classed.

---

## 4. Performance and Stability Testing

### 4.1 Validation criteria (validator-set)

| Test | Criterion |
|---|---|
| Gini | ≥ 0.40 |
| KS | ≥ 0.30 |
| Brier | Below no-skill value p(1 - p) = 0.210 |
| Hosmer-Lemeshow (10 groups) | p > 0.05 |
| Calibration slope | 0.80 to 1.20 |
| In-sample minus CV Gini | ≤ 0.05 |
| PSI / CSI | < 0.10 stable; 0.10 to < 0.25 warning; ≥ 0.25 action |

### 4.2 Discrimination

| Evidence | Model | AUC | Gini (95% CI) | KS | Brier |
|---|---|---|---|---|---|
| Pooled OOF, n = 750 | Scorecard | 0.746 | 0.493 (0.420, 0.564) | 0.400 | 0.178 |
| | XGBoost | 0.759 | 0.517 (0.444, 0.590) | 0.422 | 0.175 |
| Holdout, n = 250 | Scorecard | 0.757 | 0.514 (0.375, 0.640) | 0.453 | 0.173 |
| | XGBoost | 0.738 | 0.475 (0.330, 0.599) | 0.394 | 0.181 |
| Repeated 5×5-fold, n = 1,000 | Scorecard | | 0.496 (SD 0.012) | 0.410 | 0.177 |
| | XGBoost | | 0.491 (SD 0.009) | 0.407 | 0.179 |

Intervals are percentile bootstrap (1,000 resamples) and reflect holdout sampling variability for a fixed model, not refitting variability. The holdout interval width of 0.27 is the binding limitation on this evidence (F1).

### 4.3 Calibration

Out-of-fold predictions, deciles of predicted PD (scorecard):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 0.056 | 0.087 | 0.122 | 0.160 | 0.233 | 0.307 | 0.378 | 0.449 | 0.530 | 0.673 |
| Observed | 0.110 | 0.080 | 0.120 | 0.160 | 0.260 | 0.330 | 0.330 | 0.450 | 0.530 | 0.630 |

| Model | Mean PD / observed | Slope | Intercept | Hosmer-Lemeshow χ²(8) | p-value | Outcome |
|---|---|---|---|---|---|---|
| Scorecard | 0.2996 / 0.300 | 0.86 | -0.09 | 8.09 | 0.425 | Pass |
| XGBoost | 0.3012 / 0.300 | 0.74 | -0.18 | 24.93 | 0.002 | **Fail** |

A slope below 1.0 means predictions are more extreme than outcomes warrant. The scorecard slope is within tolerance but near the lower bound. The lowest decile under-predicts (5.6% predicted, 11.0% observed), which matters because the approved population is drawn from the low-PD deciles.

### 4.4 Overfitting

| Model | In-sample Gini | Repeated-CV Gini | Gap | Criterion (≤ 0.05) |
|---|---|---|---|---|
| Scorecard | 0.542 | 0.496 | 0.046 | Pass, marginal |
| XGBoost | 0.575 | 0.491 | 0.084 | Fail |

### 4.5 Population stability (score PSI)

PSI between training out-of-fold scores (n = 750) and holdout scores (n = 250), 10 bins: **0.043**. The expected PSI between two random samples of these sizes is approximately (bins - 1)(1/n₁ + 1/n₂) = 0.048. Across 1,000 random splits the null distribution has mean 0.049, 95th percentile 0.094 and 99th percentile 0.121.

The observed value is indistinguishable from sampling noise and, because the split is random, carries no information on temporal stability. At 250 monitoring observations the 0.10 warning threshold sits close to the 95th percentile of the no-drift distribution (F7).

### 4.6 Characteristic stability (CSI), holdout vs training

| Characteristic | Job | Credit amount | Duration | Housing | Saving accounts | Checking account | Purpose |
|---|---|---|---|---|---|---|---|
| CSI | 0.018 | 0.025 | 0.002 | 0.000 | 0.040 | 0.000 | 0.040 |
| Status | Green | Green | Green | Green | Green | Green | Green |

### 4.7 Monitoring-design test under simulated drift

Because no real drift is available, controlled shocks were applied to the holdout to test whether the PSI/CSI rules detect them. These are simulations.

| Shock to holdout | Mean PD (baseline 0.302) | Score PSI | Status | Maximum CSI | Characteristic |
|---|---|---|---|---|---|
| Duration + 12 months | 0.362 | 0.201 | Amber | 5.894 | Duration |
| Credit amount × 1.5 | 0.306 | 0.045 | **Green** | **0.288** | Credit amount |
| No-account applicants reclassified to "little" checking | 0.410 | 1.697 | Red | 5.438 | Checking account |
| All applicants set to "little" savings | 0.343 | 0.138 | Amber | 4.935 | Saving accounts |

A 50% increase in loan amounts leaves score PSI Green because the shift falls within the top bin, while CSI for that characteristic reaches 0.288, above the 0.25 action level. Score PSI is not sufficient on its own; characteristic-level CSI is required. CSI saturates when a bin empties.

---

## 5. Challenger Model Benchmarking

### 5.1 Design

The challenger uses the same WoE inputs, data and cross-validation folds as the champion. It adds non-linear and interaction capacity (200 depth-2 trees against eight scorecard parameters). Hyperparameters are fixed and untuned (F8).

### 5.2 Results

| Metric (repeated 5×5-fold, n = 1,000) | Scorecard | XGBoost |
|---|---|---|
| Gini | 0.496 | 0.491 |
| KS | 0.410 | 0.407 |
| Brier | 0.1768 | 0.1785 |
| Log-loss | 0.530 | 0.537 |
| Calibration slope | 0.86 | 0.74 |
| Hosmer-Lemeshow p | 0.425 | 0.002 |
| In-sample minus CV Gini | 0.046 | 0.084 |

### 5.3 Paired significance test

Paired bootstrap (2,000 resamples) of the Gini difference, XGBoost minus scorecard, on identical rows:

| Sample | Mean difference | 95% CI | Share of resamples > 0 |
|---|---|---|---|
| Out-of-fold, n = 1,000 | -0.001 | (-0.018, +0.014) | 44.9% |
| Holdout, n = 250 | -0.039 | (-0.071, -0.008) | 1.1% |

### 5.4 Conclusion

The challenger shows no measurable improvement in discrimination. It is worse on calibration, on the overfitting criterion and on the holdout. The depth sweep (2.5) shows an additive model performs as well as any tree depth. Added complexity is **not quantitatively justified**, and the scorecard is retained as champion. The holdout difference comes from one 250-row split and is supporting evidence only.

---

## 6. Sensitivity and Stress Analysis

### 6.1 One-at-a-time sensitivity

Shocks applied to all 1,000 applicants, models fitted on full data. Baseline decline rate uses each model's cutoff (scorecard 0.19, XGBoost 0.20).

| Shock | Scorecard mean PD | Scorecard decline rate | XGBoost mean PD | XGBoost decline rate |
|---|---|---|---|---|
| Baseline | 0.300 | 59.8% | 0.299 | 57.5% |
| Duration + 12 months | 0.366 | 70.4% | 0.375 | 66.4% |
| Duration × 1.5 | 0.349 | 67.9% | 0.355 | 63.9% |
| Credit amount × 1.5 | 0.306 | 60.6% | 0.303 | 56.4% |
| Credit amount × 2 | 0.311 | 61.0% | 0.309 | 57.5% |
| Checking account: all "little" | **0.451** | **99.4%** | 0.456 | 95.5% |
| Saving accounts: all "little" | 0.333 | 66.5% | 0.335 | 64.5% |
| Duration = 120 months (training max 72) | 0.422 | 82.9% | 0.424 | 73.5% |

Both models respond in the economically expected direction to duration and account status. Both are nearly insensitive to loan size. Checking account status is the most powerful lever by a wide margin (F2).

### 6.2 Monotonicity, saturation and extrapolation

PD is monotone non-decreasing in Duration for both models (verified on a 4-month grid from 4 to 72 months for a median applicant). It is a step function with plateaus (2.3). Inputs outside the training range map silently to the end bin: a 120-month duration receives the same WoE as 32 months, and no warning is raised (F8).

### 6.3 Macroeconomic stress overlay

**Method.** A Vasicek single-factor probit shift treats the model PD as the PD at the median state of the economy: PD(z) = Φ(Φ⁻¹(PD) - √(ρ/(1-ρ)) · z), so z = 0 returns the model PD. The systematic factor is a linear function of two shocks, z = -0.25 × (unemployment, pp) + 0.30 × (GDP, pp). Asset correlation ρ = 0.10 and LGD = 0.45 are assumptions. The loadings are not estimated from data. EAD is Credit amount. Expected loss uses out-of-fold PDs; in-sample PDs differ by 0.2%.

| Scenario | Unemployment / GDP shock | z | Mean PD | Expected loss (currency units) | Multiple of baseline | Share above cutoff |
|---|---|---|---|---|---|---|
| Baseline | 0 / 0 | 0.0 | 0.300 | 528,703 | 1.00 | 60.5% |
| Adverse | +3.0 pp / -2.5 pp | -1.5 | 0.457 | 764,228 | 1.45 | 85.9% |
| Severely adverse | +6.0 pp / -5.0 pp | -3.0 | 0.621 | 995,355 | 1.88 | 99.9% |

**Sensitivity of the expected-loss multiple to asset correlation ρ:**

| ρ | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| Adverse | 1.30 | 1.45 | 1.67 | 1.87 |
| Severely adverse | 1.61 | 1.88 | 2.25 | 2.49 |

**Reverse stress.** Mean portfolio PD reaches 50% at z = -1.89, between the adverse and severely adverse scenarios.

### 6.4 Assessment

Stress outputs are internally consistent (identity at z = 0, monotone in z) but illustrative. The multiple varies by 0.57 (adverse) across a plausible range of ρ, which is larger than the effect being measured in most comparisons. A fixed cutoff declines 86% of applicants under the adverse scenario and 99.9% under the severe one, so the decision rule is procyclical and not a through-the-cycle policy (F6).

---

## 7. Validation Findings and Remediation Plan

### 7.1 Principal findings

#### F1. [HIGH] Evidence base cannot support production reliance

| Element | Detail |
|---|---|
| **Finding** | Performance evidence rests on 1,000 observations with no origination dates. Out-of-time, walk-forward and vintage analysis cannot be performed. Holdout Gini 95% CI is (0.375, 0.640). The default definition, observation window and data provenance are undocumented. The reported PSI (0.043) equals the random-split noise floor and says nothing about temporal drift |
| **Impact** | Discriminatory power in a future population is unquantified. The holdout does not demonstrate Gini ≥ 0.40. Any production use relies on an untested stationarity assumption |
| **Required remediation** | (i) Obtain a dated development sample with at least eight quarterly vintages and a documented default definition and outcome window. (ii) Run `walk_forward` with a label-maturity gap at least equal to the outcome window; report Gini, KS and PSI by vintage. (iii) Document provenance and sampling design |
| **Owner / timeline** | Model Owner and Data Owner. Before any production use |

#### F2. [HIGH] Dominant risk driver lacks an economic rationale

| Element | Detail |
|---|---|
| **Finding** | Checking account has IV 0.659 (above the 0.50 review level) and the largest coefficient (-0.877). Removing it lowers Gini by 0.155, from 0.490 to 0.334. The effect comes from the 394 applicants (39.4%) with no checking account, who default at 11.7% against 22.2% to 49.3% for other levels. The direction is opposite to the usual thin-file prior. The savings "none" state mixes unknown and no-account applicants. No business rationale or source-system definition is documented |
| **Impact** | Model risk is concentrated in one field. If "none" means something different in a production population (unbanked, thin file, field not captured), PDs are misstated at portfolio level by up to 15 points of mean PD (sensitivity test, 6.1) with direct approval consequences. Reason codes would present this driver to declined applicants |
| **Required remediation** | (i) Document the source-system definition of the "none" state and its population. (ii) Estimate and report a challenger excluding the characteristic, with its Gini and approval-rate trade-off. (iii) Obtain business sign-off on the direction of effect or constrain it by coarse-classing "none" with an adjacent level. (iv) Add a data-quality control on the missing/none share of both account fields |
| **Owner / timeline** | Model Developer with Business Owner. Before any production use |

#### F3. [MEDIUM] Decision cutoff is unstable and rests on an unvalidated cost assumption

| Element | Detail |
|---|---|
| **Finding** | The deployed cutoff (0.19) is the argmin of a flat cost curve. Costs within 2% of the minimum span 0.16 to 0.22. The bootstrap 5th to 95th percentile of the optimum is 0.11 to 0.25. The cutoff from the 750-row training split was 0.15 against 0.19 on the full sample. The 5:1 cost matrix is borrowed from the UCI documentation, counts applicants and ignores exposure. At this ratio 60.5% of applicants are declined. The challenger fails calibration (Hosmer-Lemeshow p = 0.002) |
| **Impact** | Approval policy is sensitive to estimation noise and to a cost ratio that has no institutional basis. A change of cost ratio changes approvals materially |
| **Required remediation** | (i) Adopt the closed-form cutoff c_FP/(c_FP + c_FN) as the reference point and report the near-optimal range, not a point estimate. (ii) Replace flat costs with exposure-weighted LGD × EAD against lifetime margin from the lender. (iii) Report approval rate and expected cost for cost ratios from 2:1 to 10:1. (iv) Recalibrate the challenger (Platt or isotonic) if retained, then repeat the Hosmer-Lemeshow test |
| **Owner / timeline** | Model Developer with Credit Policy. 60 days |

#### F4. [MEDIUM] Disparate outcomes for applicants under 30 despite exclusion of Age

| Element | Detail |
|---|---|
| **Finding** | At the deployed cutoff the under-30 approval rate is 32.1% against 47.2% for age 50+ (adverse-impact ratio 0.68, below the 0.80 screen). The false-positive rate (creditworthy applicants declined) is 56.0% against 42.9% for 50+. Sex ratio is 0.91. Group default rates differ (36.9% under 30 against 25.6% to 27.2%), so part of the gap reflects base rates. Within-group Gini is lowest under 30 (0.447 against 0.476 to 0.548). No single input correlates with Age above \|r\| = 0.11, so the effect arises from the joint distribution |
| **Impact** | Exposure under applicable fair-lending and non-discrimination requirements. Reason codes could cite characteristics that act jointly as age proxies |
| **Required remediation** | (i) Run a less-discriminatory-alternative search: re-estimate with coarse-classed or removed variables and report the Gini and adverse-impact trade-off. (ii) Document the business justification for the retained specification and obtain compliance review. (iii) Add adverse-impact ratio and group error rates to monitoring (Section 8) |
| **Owner / timeline** | Model Owner with Compliance. 90 days |

### 7.2 Additional findings

| ID | Severity | Finding | Impact | Required remediation | Owner / timeline |
|---|---|---|---|---|---|
| F5 | **Medium** | Job has a positive coefficient (+0.148) against an expected negative sign, IV 0.009 (below noise floor), and no leave-one-out contribution. Four levels have fewer than 30 observations (Job = 0; Purpose = repairs, domestic appliances, vacation/others) | Noise fitting. Unstable WoE for sparse bins. Coefficient-sign failure in a conceptual-soundness check | Remove Job or constrain coefficients to ≤ 0. Pool levels below 30 observations into "other". Re-run Sections 2.2 and 4.2 | Model Developer. 30 days |
| F6 | **Medium** | Stress loadings and ρ are assumptions. The adverse expected-loss multiple ranges from 1.30 to 1.87 across ρ from 0.05 to 0.30. A fixed cutoff declines 85.9% (adverse) and 99.9% (severe) of applicants. PIT/TTC status of the PD is undocumented | Outputs cannot support provisioning or capital. Decision rule is procyclical | Label outputs illustrative. Estimate loadings from dated default and macro series once F1 is closed. Document PIT/TTC status and define a through-the-cycle cutoff policy | Model Developer. After F1 |
| F7 | **Low** | Score PSI misses characteristic-level shifts: Credit amount × 1.5 gives PSI 0.045 but CSI 0.288. The 0.10 threshold is close to the no-drift 95th percentile (0.094) at 250 observations. Monitoring functions exist but no scheduled job runs | Drift could go undetected, or false warnings could occur at small sample sizes | Implement the monitoring job in Section 8 with PSI, CSI, performance and adverse-impact ratio. Require at least 1,000 scored applications per window | Model Owner. 60 days |
| F8 | **Low** | The app accepts Duration up to 120 months (training maximum 72) with no out-of-domain flag. Unseen categories receive neutral WoE silently. Artefacts load through `joblib` with no integrity hash. Challenger hyperparameters are fixed and untuned | Silent extrapolation. Deserialisation executes code from the artefact file. Challenger benchmark understated | Add range validation with a refer-to-manual-review flag. Log unseen-category rate. Store a SHA-256 checksum in bundle metadata and verify on load. Tune the challenger by nested cross-validation | Model Developer. 30 days |

---

## 8. Ongoing Monitoring Plan and Governance Triggers

### 8.1 Population stability rules

| Score PSI | Status | Required action |
|---|---|---|
| **PSI < 0.10** | Green | No action required. Record value in the monitoring log |
| **0.10 ≤ PSI < 0.25** | Amber (warning) | Increase monitoring to weekly. Compute CSI for every characteristic and identify the driver. Confirm data-pipeline integrity. Root-cause note to Model Owner within 15 business days |
| **PSI ≥ 0.25** | Red (recalibration trigger) | Notify Model Owner within 2 business days. Independent Validation assesses within 30 days. Move automated decisions to manual-review overlay until the assessment concludes. Recalibration or re-estimation required unless the shift is shown to be a data error |

**Sample-size rule.** Expected PSI under no drift is approximately (bins - 1)(1/n₁ + 1/n₂). For 10 bins and a 750-observation reference, a 250-observation window gives 0.048 and a 1,000-observation window 0.021. Windows below 1,000 scored applications are reported but do not trigger Amber or Red without confirmation by CSI.

### 8.2 Metric thresholds

| Metric | Frequency | Green | Amber | Red |
|---|---|---|---|---|
| Score PSI | Monthly | < 0.10 | 0.10 to < 0.25 | ≥ 0.25 |
| CSI, each characteristic | Monthly | < 0.10 | 0.10 to < 0.25 | ≥ 0.25 on any characteristic |
| Share of "none" in either account field (development: 39.4% checking, 18.3% savings) | Monthly | within ± 5 pp | ± 5 to 10 pp | beyond ± 10 pp |
| Unseen-category rate / out-of-range input rate | Monthly | < 1% / < 2% | 1% to 3% / 2% to 5% | > 3% / > 5% |
| Gini (when ≥ 100 observed defaults) | Quarterly | ≥ 0.44 | 0.40 to < 0.44 | < 0.40 |
| KS | Quarterly | ≥ 0.35 | 0.30 to < 0.35 | < 0.30 |
| Observed / predicted default rate | Quarterly | 0.85 to 1.15 | 0.70 to < 0.85 or > 1.15 to 1.30 | outside 0.70 to 1.30 |
| Hosmer-Lemeshow p-value | Quarterly | ≥ 0.05 | 0.01 to < 0.05 | < 0.01 |
| Approval rate (development: 39.5%) | Monthly | within ± 5 pp | ± 5 to 10 pp | beyond ± 10 pp |
| Adverse-impact ratio, all monitored groups | Quarterly | ≥ 0.80 | 0.70 to < 0.80 | < 0.70 |

The Gini Amber band begins at 0.44, five points below the development estimate of 0.49. The adverse-impact baseline is already in breach for applicants under 30 (0.68) pending F4.

### 8.3 Escalation and revalidation

| Trigger | Owner | Action | Deadline |
|---|---|---|---|
| Any Amber | Model Owner | Root-cause note; increase frequency | 15 business days |
| Any Red | Model Owner → Independent Validation | Impact assessment; decide on overlay, recalibration or suspension | 2 business days to notify; 30 days to assess |
| Two consecutive Amber on the same metric | Model Owner | Treat as Red | Immediate |
| Change in cost matrix, product, channel or data source | Model Owner | Targeted revalidation of affected sections | Before change is deployed |
| Scheduled review | Independent Validation | Full revalidation | Annual |

### 8.4 Implementation status

`psi`, `bootstrap_ci` and `fairness_report` exist in `metrics.py`. No scheduled job, outcome-capture process or monitoring log exists, so none of the rules in this section is operating (F7).

---

## Appendix A. Test inventory

| Test | Module section (`validate.py`) | Result key (`validation_results.json`) |
|---|---|---|
| Data audit, outliers, sparse levels | 1 | `data` |
| Repeated CV, in-sample gap, calibration | 2 | `performance` |
| PSI, noise floor, CSI, simulated drift | 3 | `stability` |
| Paired bootstrap challenger test | 4 | `challenger` |
| Leakage experiment | 5 | `leakage_test` |
| Class-weighting experiment | 5 | `imbalance_test` |
| Regularisation and depth sweep | 5 | `hyperparameter_sensitivity` |
| Coefficients, VIF, bin monotonicity | 5 | `features` |
| Leave-one-out ablation, age-proxy screen | 5b | `ablation_scorecard`, `age_proxy_corr_with_woe` |
| Cutoff stability | 6 | `cutoff` |
| One-at-a-time sensitivity, monotonicity | 7 | `sensitivity` |
| Vasicek stress, ρ sensitivity, reverse stress | 8 | `stress` |
| Fairness by Sex and age band | 9 | `fairness` |

Unit tests in `tests/` cover WoE known-answer, PSI, bootstrap interval, closed-form cutoff, stress identity and monotonicity, SHAP additivity (both models), bundle round-trip, walk-forward row count and a cold-start smoke test of the app.

## Appendix B. Limitations of this memo

1. Independence is not achieved (see header).
2. All conclusions are conditional on the committed dataset. No statement is made about any production portfolio.
3. Criteria in Section 4.1 and thresholds in Section 8 are validator-set conventions, not regulatory limits.
4. Stress results use assumed parameters and are illustrative.
5. Reference frameworks (SR 11-7, EBA/GL/2017/16, EBA/GL/2020/06) guide the structure of the review. The memo does not assert compliance with them.
