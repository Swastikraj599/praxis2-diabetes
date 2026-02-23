# 🩺 Praxis 2.0 — Counterfactual Clinical Decision Support System

> **Praxis 2.0 · GDG on Campus — GB Pant**
> Theme: Preventive Risk & Clinical Decision Support

---

## What This Project Does

Most diabetes risk tools give you a number — *"your risk is 87%"* — and stop there. That number alone is clinically useless without knowing what to do about it.

This system answers the question that actually matters:

> **"What is the minimum realistic change in this patient's lifestyle that would bring their diabetes risk below the clinical threshold — and how do we explain that differently to the doctor versus the patient?"**

This concept is called **algorithmic recourse** — a technique from ML fairness research that generates actionable, patient-specific intervention pathways rather than passive risk scores.

---

## Live Demo

🌐 **[praxis-diabetes.streamlit.app](https://praxis2-diabetes-ahs3hkk8mpkkkzpnvehmf7.streamlit.app/)**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT APP                            │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐  │
│  │   Risk   │  │Counterfactual│  │  What-If  │  │ Reports  │  │
│  │ Overview │  │ Interventions│  │ Simulator │  │ Doctor / │  │
│  │  Tab 1   │  │    Tab 2     │  │   Tab 3   │  │ Patient  │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └────┬─────┘  │
└───────┼───────────────┼────────────────┼──────────────┼────────┘
        │               │                │              │
        ▼               ▼                ▼              ▼
┌───────────────┐ ┌───────────────┐              ┌───────────────┐
│   MODULE 1    │ │   MODULE 2    │              │   MODULE 3    │
│ Gradient      │ │ Counterfactual│              │ Gemini 1.5    │
│ Boosting      │ │ Engine        │              │ Flash         │
│ Classifier    │ │ (3 strategies)│              │ Narratives    │
│ AUC: 0.979    │ │               │              │ (2 audiences) │
└───────┬───────┘ └───────┬───────┘              └───────────────┘
        │                 │
        ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              model_artifacts.pkl (serialised)                   │
│    model · encoders · threshold · feature metadata             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET                                     │
│   100,541 patients · 8 features · 8.5% diabetes prevalence     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Provided by Praxis 2.0 committee |
| Rows | 100,992 (100,541 after cleaning) |
| Features | 8 (gender, age, hypertension, heart disease, smoking history, BMI, HbA1c, blood glucose) |
| Target | `diabetes` (binary: 0 = no, 1 = yes) |
| Class balance | 91,956 negative / 8,585 positive (8.5% prevalence) |
| Missing values | None |
| Duplicates | None |

**Key statistical findings from EDA:**

- HbA1c mean: **6.94%** (diabetic) vs **5.40%** (non-diabetic) — largest separation of any feature
- Blood glucose mean: **194 mg/dL** (diabetic) vs **133 mg/dL** (non-diabetic)
- Hypertension patients: **27.9%** diabetes rate vs 6.9% baseline
- Heart disease patients: **32.2%** diabetes rate
- Age 60+: **20%** diabetes rate vs 0.5% for under-18

---

## Module Breakdown

### Module 1 — Data Pipeline + Model Training (`module1_model.py`)

**Cleaning decisions:**
- Dropped 451 rows where `age = 0` (clinically invalid)
- Consolidated `'Other'` gender (18 rows, 0% diabetes rate) into `'Female'` to avoid sparse encoding
- Merged `'ever'` and `'not current'` smoking categories into `'former'` (semantically equivalent)
- Clipped BMI outliers above 70 (19 rows — physiologically extreme)

**Model:** `GradientBoostingClassifier`
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`
- Class imbalance handled with **sample weights** (`compute_sample_weight('balanced')`) rather than SMOTE — avoids synthetic data distorting the counterfactual search space

**Threshold tuning:**
- Default threshold 0.50 gives 92% recall but 1,810 false positives per 20k patients
- Optimal threshold **0.876** (maximises F1) gives 97% precision, 70% recall, 41 false positives
- Both thresholds are preserved — app can switch between **screening mode** (catch everyone) and **clinical mode** (high precision referrals)

**Performance:**

| Metric | Value |
|---|---|
| ROC-AUC | **0.9790** |
| 5-Fold CV AUC | **0.9777 ± 0.0016** |
| Precision (diabetic class) | 97% |
| Recall (diabetic class) | 70% at optimal threshold |
| Sensitivity | 92% at default threshold |

**Feature importances:**

| Feature | Importance |
|---|---|
| HbA1c Level | 50.1% |
| Blood Glucose | 34.3% |
| Age | 11.0% |
| BMI | 2.7% |
| Hypertension | 0.9% |
| Heart Disease | 0.4% |
| Smoking History | 0.3% |
| Gender | 0.1% |

---

### Module 2 — Counterfactual Engine (`module2_counterfactual.py`)

The core differentiator. For any patient above the risk threshold, this module computes three types of intervention pathways:

**Strategy 1 — Single-Feature Counterfactuals**
For each modifiable feature independently, finds the **minimum change** that flips the prediction below threshold. Returns the result ranked by a feasibility score (1–10) that accounts for clinical difficulty and magnitude of required change.

Modifiable features (non-modifiable features — age, gender — are never touched):
- HbA1c Level, Blood Glucose, BMI, Smoking Status, Hypertension

**Strategy 2 — Multi-Feature Counterfactuals**
Searches 2-feature combinations (HbA1c × Glucose, HbA1c × BMI, Glucose × BMI) for the pair with the **smallest normalised combined delta** that crosses the threshold. A patient may find it more realistic to make two moderate changes than one large one.

**Strategy 3 — Safest Path**
Greedy search ordered by model feature importance. Applies the minimum incremental change to each feature in sequence until the threshold is crossed. Produces the **smallest total perturbation** — the "least disruptive" intervention for low-adherence patients.

**Feasibility scoring formula:**
```
feasibility = 10 - (clinical_difficulty × 1.2) - (normalised_magnitude × 4)
clamped to [1, 10]
```

Where `clinical_difficulty` is a manually assigned score (1–5) based on ADA guidelines:
blood glucose = 2, HbA1c = 3, BMI = 3, smoking = 4, hypertension = 5

---

### Module 3 — Gemini Narrative Generator (`module3_gemini_narrative.py`)

Takes the structured counterfactual output and uses **Gemini 1.5 Flash** to generate two audience-specific narratives from the same underlying data.

**Clinical Note (for the doctor):**
- Temperature: 0.25 (low — precision matters)
- Sections: Patient Summary, Key Risk Drivers, Recommended Interventions, Next Steps, Model Limitations
- Uses medical terminology, references ADA thresholds, includes bias caveat
- 250–300 words — readable in under 90 seconds during OPD

**Patient Report (for the patient):**
- Temperature: 0.55 (higher — natural, warm language)
- Sections: Your Results, What This Means, Your Action Plan, What to Expect, A Final Note
- Zero jargon (7th-grade reading level), explains any clinical term used, focuses on what the patient can do
- 280–350 words

**What-If Explanation:**
- Called live when the simulator sliders change
- 3 sentences, plain English, explains *why* a specific value change moved the risk score

**Fallback:** If the API is unavailable or quota is exceeded, structured template-based outputs are served automatically. The app never shows an error to the user.

---

### Module 4 — Streamlit App (`app.py`)

Four-tab interface. All computation results stored in `st.session_state` — switching tabs never recomputes anything.

| Tab | What it shows |
|---|---|
| Risk Overview | Gauge chart, clinical indicator cards (HbA1c / glucose / BMI vs WHO thresholds), feature contribution bar chart, action banner |
| Interventions | 3 counterfactual cards ranked by feasibility, risk comparison bar chart, expandable doctor/patient detail per card |
| What-If Simulator | Live sliders pre-filled with patient values, risk meter updates on every change, Gemini explanation of why risk changed |
| Reports | Toggle between clinical note and patient report, Gemini or template, download button |

---

## How ML and GenAI Are Integrated

This is not a project where ML predicts and GenAI simply describes the prediction. The integration is structural:

1. **ML produces the counterfactuals** — Gemini cannot compute minimum-delta feature changes. That requires the trained model's predict function and a search algorithm.
2. **Gemini reasons over the counterfactuals** — it receives the full structured intervention data (feasibility scores, timelines, clinical mechanisms, delta values) and synthesises it into coherent, audience-calibrated language that a template cannot match.
3. **The What-If simulator closes the loop** — the user's slider interaction feeds back into the ML model in real time, and Gemini explains the result. Neither component works without the other.

---

## Ethical Considerations and Limitations

**What the model does not do:**
- It does not diagnose diabetes. It estimates risk based on population-level patterns.
- It does not recommend medication. It suggests lifestyle interventions and flags when clinical escalation is appropriate.
- It never suggests changing non-modifiable features (age, gender, genetic history).

**Bias considerations:**
- Dataset has 59% female, 41% male. The model may perform differently across demographic subgroups — this is documented and surfaced in the Model Limitations section of every clinical note.
- `'Other'` gender (18 patients, 0% diabetes rate) is too sparse for reliable predictions and is consolidated. This is a limitation of the dataset, not the model.
- Counterfactuals are ranked by feasibility, but feasibility is population-averaged. A recommendation to reduce BMI by 3 units may be equally difficult for a patient with a physical disability as reducing HbA1c is for someone without dietary support. The clinician must contextualise.

**This tool is decision support, not a diagnostic instrument.** Every output includes this caveat explicitly.

---

## Project Structure

```
praxis2-diabetes/
├── app.py                          # Streamlit app (all modules inlined)
├── module1_model.py                # Data pipeline + model training
├── module2_counterfactual.py       # Counterfactual engine
├── module3_gemini_narrative.py     # Gemini narrative generator
├── requirements.txt                # Python dependencies
├── artifacts/
│   └── model_artifacts.pkl         # Trained model + encoders + threshold
└── README.md
```

---

## Setup and Installation

**Run locally:**
```bash
git clone https://github.com/Swastikraj599/praxis2-diabetes.git
cd praxis2-diabetes
pip install -r requirements.txt
streamlit run app.py
```

**Environment variable required for Gemini:**
```bash
export GEMINI_API_KEY="your-key-here"
# Get free key at: https://aistudio.google.com/app/apikey
```

The app runs fully without a Gemini key — template-based narratives are served as fallback.

**Requirements:**
```
streamlit
scikit-learn
plotly
google-generativeai
numpy
pandas
```

---

## Reproducing the Model

Run `module1_model.py` with `diabetes_dataset.csv` in the working directory:
```bash
python module1_model.py
```
This trains the model, evaluates it, tunes the threshold, and saves `artifacts/model_artifacts.pkl`. Expected AUC: **0.979**. Expected CV AUC: **0.9777 ± 0.0016**.

---

## Key Assumptions

| Assumption | Justification |
|---|---|
| Only modifiable features are targeted in counterfactuals | Ethical — age and gender cannot and should not be change targets |
| Sample weights used for class imbalance instead of SMOTE | SMOTE generates synthetic patients that distort the feature space used for counterfactual search |
| Optimal threshold set at 0.876 (max F1) for clinical mode | In a clinical referral context, false positives cause unnecessary patient anxiety and resource waste — precision matters more than raw recall |
| Smoking categories 'ever' and 'not current' merged into 'former' | Both indicate past smoking with no current exposure — clinically equivalent |
| Feature importance order for safest path: HbA1c → Glucose → BMI → Hypertension → Smoking | Matches model's learned feature importances (50.1%, 34.3%, 2.7%, 0.9%, 0.3%) |
| Gemini temperature 0.25 for clinical note, 0.55 for patient report | Lower temperature for clinical precision, higher for natural empathetic language |

---

## Team

Built for **Praxis 2.0 — GenAI + ML Innovation Showcase**
Organised by **GDG on Campus, GB Pant**

---

*This project is for educational and research purposes. It is not a medical device and should not be used for clinical diagnosis.*
