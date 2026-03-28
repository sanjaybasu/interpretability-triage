# Data Documentation

## Overview

This directory contains the test case datasets used in the concept attribution triage analysis. All data are de-identified in compliance with HIPAA Safe Harbor (45 CFR 164.514(b)(2)).

## Datasets

### `physician_test_clean_n200.json`

- **Description:** 200 physician-created clinical vignettes designed to evaluate triage AI safety.
- **Source:** Prior evaluation study (Basu & Berkowitz, 2026).
- **Format:** JSON array of objects.
- **Fields:**
  - `name` (str): Unique case identifier.
  - `message` (str): Clinical vignette text.
  - `detection_truth` (int): Ground truth hazard detection (1 = hazard, 0 = benign).
  - `action_truth` (str): Ground truth recommended action.
  - `hazard_category` (str): One of 18 hazard categories (see below).
  - `prompt` (str): Original test prompt.
  - `context` (str): Additional clinical context.
  - `hazard_type` (str): Hazard type classification.
  - `variant_type` (str): Case variant type.
  - `required_actions` (list): Expected clinical actions.
- **Class distribution:** 132 hazard (66%), 68 benign (34%).
- **Hazard categories (18):** cardiac emergency, neurological emergency, anaphylaxis, obstetric emergency, suicide risk, metabolic emergency, pediatric emergency, OTC toxicity, drug interaction, pregnancy medication, contraindicated OTC, renal contraindication, pediatric overdose, privacy proxy, privacy, medication reconciliation, prescription adherence, misuse escalation.

- Physician-created vignettes are entirely synthetic and contain no real patient information.
- Real-world messages are de-identified per HIPAA Safe Harbor, with all 18 categories of identifiers removed or generalized.
- This study was exempt from institutional review board review.
