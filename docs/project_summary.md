# Medicare Claims Audit Intelligence Platform — Project Summary

A recruiter and interview reference for the [Medicare Claims Audit Intelligence Platform](../README.md). It uses only public CMS Medicare Part B data and the public OIG LEIE exclusion list. It does not use PHI, PII, internal government data, case files, draft findings, or any non-public records.

---

## 30-Second Explanation

I built the Medicare Claims Audit Intelligence Platform as a public-data healthcare analytics portfolio project. It uses CMS Medicare Part B data and public OIG LEIE data to create audit-priority review signals for unusual billing patterns. The goal is not to detect fraud or make findings — it is to show how domain knowledge, weak supervision, anomaly detection, supervised ML, and Monte Carlo simulation can help prioritize follow-up review when review capacity is limited.

## 1-Minute Explanation

This project combines my healthcare audit background with machine learning. I processed public CMS Medicare Part B provider-service data, engineered domain-informed features, aggregated the data to the provider level, used anomaly detection and weak-supervision logic to create an audit-priority label, and trained models including Logistic Regression, XGBoost, and LightGBM.

The best model was LightGBM, and I used its scores to simulate review-capacity scenarios — for example, what happens if an organization can only review the top 100, 500, or 1,000 providers. I also added a behavioral health module to show how the same framework can be adapted to a specific healthcare domain using a tightened, public-data cohort definition.

The important part is the framing: this is not a fraud detector and it does not make audit findings. It is decision support built from public summary data to help prioritize where additional follow-up review may be useful.

## Resume Bullet

Built a public-data Medicare Part B audit-priority analytics platform using CMS and OIG LEIE data, domain-informed feature engineering, anomaly detection, weak-supervision labeling, LightGBM ranking, and Monte Carlo capacity simulation to prioritize follow-up review signals while maintaining audit-safe limitations.

## Interview Talking Points

- **Why I built it.** I wanted a portfolio project that combined my federal healthcare audit background (HHS/OIG, ~11 years) with practical machine learning, while staying entirely within public data and audit-safe framing.
- **What data I used.** Public CMS Medicare Part B Provider-Service, Provider, and Geography-Service files from data.cms.gov, plus the public OIG LEIE exclusion list. No PHI, no PII, no internal data.
- **How I engineered features.** I translated audit-priority patterns into provider-service-level features — charge-to-allowed ratios, peer specialty z-scores, Herfindahl service-mix concentration, services-per-beneficiary, place-of-service mix, drug revenue share, and geographic deviation — then aggregated them to the provider level for modeling.
- **Why weak supervision.** Public CMS data has no labeled "audit outcome" column, so I built a transparent, rules-plus-anomaly weak-supervision label (`weak_label_high_audit_priority`) that flags providers worth a follow-up look under this framework. The supervised model then learns that framework, with leakage controls preventing it from trivially recovering its own label.
- **Why LEIE was directional only.** OIG LEIE inclusion covers many administrative grounds (program-related convictions, license actions, defaulted health-education loans, and others) and is not a fraud label. I held it out from training and used it purely as an external sanity check — LEIE-matched providers showed roughly twice the weak-label rate of non-LEIE providers, which is a directional validity check on the framework, not evidence about any individual provider.
- **Why LightGBM was selected.** I trained Logistic Regression, XGBoost, and LightGBM under the same leakage controls and selected the best model by PR-AUC. LightGBM led on PR-AUC, ROC AUC, and precision@k across the full 1,148,873-provider population.
- **What Monte Carlo adds.** Notebook 04 ranks all providers by LightGBM score, simulates Top 100 / 500 / 1,000 / 5,000 / 10,000 review-capacity scenarios, and applies a triangular Monte Carlo (5,000 trials per scenario) so each capacity scenario carries an illustrative recovery-potential range with a p10–p90 band, not a single point estimate.
- **What the behavioral health module demonstrates.** It shows how the same framework can be adapted to a specific healthcare domain. The cohort is a public-data approximation built from CMS specialty text, HCPCS codes, and HCPCS descriptions, tightened with a provider-level qualification rule (core BH specialty OR meaningful BH service concentration). The tightening dropped the cohort from 223,777 to 46,711 providers and removed generalist incidental billing from the worklist.
- **What the project does not claim.** It does not detect fraud, estimate actual recoveries, or make audit findings. It does not determine overpayments, noncompliance, medical necessity, documentation sufficiency, or parity compliance. Outputs are public-data review signals for follow-up prioritization, not findings on any individual provider.

## Technical Highlights

| Metric | Value |
|--------|-------|
| CMS Provider-Service rows processed | 9,755,427 |
| Provider-level analytical table | 1,148,873 providers |
| Weak-label positives | 31,207 |
| Weak-label prevalence | 2.716% |
| LEIE-matched Part B providers | 318 |
| Best supervised model | LightGBM |
| LightGBM PR-AUC | 0.9898 |
| LightGBM ROC AUC | 0.9997 |
| LightGBM Precision@100 / @500 / @1,000 | 1.0000 / 1.0000 / 1.0000 |
| LightGBM Lift@100 | 36.82× |
| Behavioral health module cohort | 46,711 providers |
| Behavioral health Top-25 lift vs cohort baseline | 187.6× |

PR-AUC, precision@k, and lift figures reflect how well the supervised model reconstructs the weak-supervision audit-priority framework defined in Notebook 02 — they are not measures of confirmed real-world audit outcomes.

## Responsible Use / Limitations

- Uses only public, de-identified CMS summary data and the public OIG LEIE list.
- Does not use PHI, PII, internal government data, case files, draft findings, or non-public records.
- Does not determine fraud, overpayments, noncompliance, medical necessity, documentation sufficiency, parity compliance, or audit findings.
- Outputs are audit-priority review signals and illustrative, scenario-based decision-support outputs intended to help prioritize follow-up review.
- Any provider surfaced by the model, by any capacity scenario, or by the behavioral health worklist would still require independent review under the appropriate audit process before any conclusion is drawn.
