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

### `realworld_test_n200_complete.json`

- **Description:** 200 de-identified real-world patient messages from a Medicaid population health program.
- **Source:** Extracted from 2.98 million messages in Medicaid care coordination programs (Ohio, Virginia, Washington).
- **De-identification:** HIPAA Safe Harbor method (45 CFR 164.514(b)(2)).
- **Format:** JSON array of objects.
- **Fields:**
  - `case_id` (str): Unique case identifier.
  - `message` (str): Patient message text.
  - `word_count` (int): Number of words in message.
  - `ground_truth_detection` (int): Ground truth hazard detection (1 = hazard, 0 = benign).
  - `ground_truth_severity` (str): Ground truth severity level.
  - `ground_truth_action` (str): Ground truth recommended action.
  - `ground_truth_hazard_category` (str): Hazard category, if applicable.
  - `patient_age` (int): Patient age in years.
  - `patient_sex` (str): Patient sex (M/F).
  - `patient_race` (str): Patient race/ethnicity from insurance enrollment records.
- **Class distribution:** 12 hazard (6%), 188 benign (94%).
- **Note:** The low hazard prevalence reflects the true base rate in routine care coordination messages.

## Provenance

These datasets were originally created for:

> Basu S, Berkowitz SA. Comparative evaluation of reinforcement learning controllers and large language models for clinical triage safety. Preprint. 2026.

The physician-created vignettes were authored by a licensed physician (JM) and independently adjudicated by three board-certified physicians, with disagreements resolved by consensus. Real-world messages were extracted from operational systems and independently adjudicated by three board-certified physicians, with disagreements resolved by consensus.

## Ethical Considerations

- No real patient identifiers are present in any dataset.
- Physician-created vignettes are entirely synthetic and contain no real patient information.
- Real-world messages are de-identified per HIPAA Safe Harbor, with all 18 categories of identifiers removed or generalized.
- This study was exempt from institutional review board review.
