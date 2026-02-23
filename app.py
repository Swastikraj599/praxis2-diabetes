import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os, time, textwrap, itertools

st.set_page_config(
    page_title="Praxis 2.0 — Diabetes Risk & Intervention",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR     = "/content/drive/MyDrive/Praxis 2.0"
ARTIFACT_PATH = "artifacts/model_artifacts.pkl"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SMOKING_OPTIONS = ["never", "No Info", "current", "former"]

FEAT_IDX = {
    "gender_enc": 0, "age": 1, "hypertension": 2, "heart_disease": 3,
    "smoking_enc": 4, "bmi": 5, "HbA1c_level": 6, "blood_glucose_level": 7,
}
FEAT_META = {
    "HbA1c_level":         {"display": "HbA1c Level",   "unit": "%",     "difficulty": 3, "timeline": "3-6 months",             "mechanism": "Dietary carb reduction + 150 min/week exercise."},
    "blood_glucose_level": {"display": "Blood Glucose", "unit": "mg/dL", "difficulty": 2, "timeline": "4-8 weeks",              "mechanism": "Reduce refined carbs and sugar intake."},
    "bmi":                 {"display": "BMI",            "unit": "kg/m2", "difficulty": 3, "timeline": "3-9 months",             "mechanism": "Caloric deficit of 500 kcal/day."},
    "smoking_enc":         {"display": "Smoking Status", "unit": "",      "difficulty": 4, "timeline": "Immediate to 1 year",    "mechanism": "NRT or pharmacotherapy recommended."},
    "hypertension":        {"display": "Hypertension",   "unit": "",      "difficulty": 5, "timeline": "1-3 months (medication)","mechanism": "Antihypertensives or DASH diet."},
}
SMOKING_LABELS = {0: "No Info", 1: "Current Smoker", 2: "Former Smoker", 3: "Never Smoked"}

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-card {
    background:#1e1e2e; border-radius:12px; padding:20px;
    border-left:5px solid; margin-bottom:16px;
}
.cf-card {
    background:#16213e; border-radius:10px; padding:18px;
    border:1px solid #0f3460; margin-bottom:14px;
}
.feasibility-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:13px; font-weight:600; margin-bottom:8px;
}
.narrative-box {
    background:#0d1117; border-radius:10px; padding:24px;
    border:1px solid #21262d; font-size:15px; line-height:1.75;
    white-space:pre-wrap;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


# ── Prediction ───────────────────────────────────────────────────
def predict_patient(patient_dict, artifacts):
    model     = artifacts["model"]
    encoders  = artifacts["encoders"]
    threshold = artifacts["threshold"]
    gender_enc  = encoders["gender_map"].get(patient_dict["gender"], 0)
    smoking_enc = encoders["smoking_map"].get(patient_dict["smoking_history"], 3)
    x = np.array([[gender_enc, patient_dict["age"], patient_dict["hypertension"],
                   patient_dict["heart_disease"], smoking_enc, patient_dict["bmi"],
                   patient_dict["HbA1c_level"], patient_dict["blood_glucose_level"]]])
    prob = model.predict_proba(x)[0][1]
    if prob < 0.15:   tier, color, icon = "Low Risk",      "#27ae60", "🟢"
    elif prob < 0.40: tier, color, icon = "Moderate Risk", "#f39c12", "🟡"
    elif prob < 0.70: tier, color, icon = "High Risk",     "#e67e22", "🟠"
    else:             tier, color, icon = "Very High Risk", "#e74c3c", "🔴"
    return {"probability": round(prob,4), "prediction": int(prob>=threshold),
            "risk_tier": tier, "risk_color": color, "risk_icon": icon,
            "feature_vector": x[0].tolist()}

def predict_from_vector(x_vec, artifacts):
    prob = artifacts["model"].predict_proba(x_vec.reshape(1,-1))[0][1]
    if prob < 0.15:   tier, color = "Low Risk",      "#27ae60"
    elif prob < 0.40: tier, color = "Moderate Risk", "#f39c12"
    elif prob < 0.70: tier, color = "High Risk",     "#e67e22"
    else:             tier, color = "Very High Risk", "#e74c3c"
    return round(prob,4), tier, color


# ── Counterfactual Engine ────────────────────────────────────────
def _prob(model, x):
    return model.predict_proba(x.reshape(1,-1))[0][1]

def _grid(feat, val):
    cfg = {
        "HbA1c_level":         {"min": 3.5,  "step": 0.1},
        "blood_glucose_level": {"min": 79,   "step": 5},
        "bmi":                 {"min": 15.0, "step": 0.5},
        "smoking_enc":         {"min": 0,    "step": 1},
        "hypertension":        {"min": 0,    "step": 1},
    }[feat]
    if feat in ("smoking_enc", "hypertension"):
        return [v for v in np.arange(cfg["min"], val, cfg["step"])]
    return list(np.arange(val, cfg["min"] - cfg["step"], -cfg["step"]))

def _feas(feat, orig, new):
    diff = FEAT_META[feat]["difficulty"]
    rng  = {"HbA1c_level":5.5,"blood_glucose_level":224,"bmi":55,"smoking_enc":3,"hypertension":1}[feat]
    return max(1.0, min(10.0, round(10 - diff*1.2 - (abs(orig-new)/rng)*4, 1)))

def _desc(feat, orig, new):
    if feat == "smoking_enc":
        return SMOKING_LABELS.get(int(orig),"?") + " -> " + SMOKING_LABELS.get(int(new),"?")
    if feat == "hypertension":
        return "Controlled (1 -> 0)"
    d = abs(orig - new)
    if feat == "HbA1c_level":
        return str(round(orig,1)) + "% -> " + str(round(new,1)) + "% (down " + str(round(d,1)) + "%)"
    if feat == "blood_glucose_level":
        return str(int(orig)) + " -> " + str(int(new)) + " mg/dL (down " + str(int(d)) + ")"
    if feat == "bmi":
        return str(round(orig,1)) + " -> " + str(round(new,1)) + " kg/m2 (down " + str(round(d,1)) + ")"
    return str(orig) + " -> " + str(new)

def run_counterfactuals(patient_dict, artifacts):
    model     = artifacts["model"]
    encoders  = artifacts["encoders"]
    threshold = artifacts["threshold"]
    gender_enc  = encoders["gender_map"].get(patient_dict["gender"], 0)
    smoking_enc = encoders["smoking_map"].get(patient_dict["smoking_history"], 3)
    x = np.array([gender_enc, patient_dict["age"], patient_dict["hypertension"],
                  patient_dict["heart_disease"], smoking_enc, patient_dict["bmi"],
                  patient_dict["HbA1c_level"], patient_dict["blood_glucose_level"]], dtype=float)
    orig_prob = _prob(model, x)

    if orig_prob < threshold:
        return {"status":"low_risk","original_prob":round(orig_prob,4),"summary":[],"single":[],"multi":[],"safest_path":None}

    # Single-feature
    single = []
    for feat in FEAT_META:
        idx, ov = FEAT_IDX[feat], x[FEAT_IDX[feat]]
        for nv in _grid(feat, ov):
            xc = x.copy()
            xc[idx] = nv
            p = _prob(model, xc)
            if p < threshold:
                single.append({
                    "features_changed": [feat],
                    "display_name": FEAT_META[feat]["display"],
                    "changes": {feat: {"from": ov, "to": nv}},
                    "new_prob": round(p,4),
                    "risk_reduction": round((orig_prob-p)*100,1),
                    "feasibility": _feas(feat,ov,nv),
                    "timeline": FEAT_META[feat]["timeline"],
                    "description": _desc(feat,ov,nv),
                    "mechanism": FEAT_META[feat]["mechanism"],
                    "x_counterfactual": xc.tolist(),
                })
                break
    single.sort(key=lambda r: r["feasibility"], reverse=True)

    # Multi-feature
    multi  = []
    RANGES = {"HbA1c_level":5.5,"blood_glucose_level":224,"bmi":55}
    for f1, f2 in itertools.combinations(["HbA1c_level","blood_glucose_level","bmi"], 2):
        i1, i2 = FEAT_IDX[f1], FEAT_IDX[f2]
        o1, o2 = x[i1], x[i2]
        c1 = _grid(f1,o1)[::3] or _grid(f1,o1)
        c2 = _grid(f2,o2)[::3] or _grid(f2,o2)
        best, bd = None, float("inf")
        for v1 in c1:
            for v2 in c2:
                xc = x.copy()
                xc[i1] = v1
                xc[i2] = v2
                p = _prob(model, xc)
                if p < threshold:
                    d = abs(o1-v1)/RANGES[f1] + abs(o2-v2)/RANGES[f2]
                    if d < bd:
                        bd = d
                        best = {"v1":v1,"v2":v2,"prob":p}
        if best:
            xc = x.copy()
            xc[i1] = best["v1"]
            xc[i2] = best["v2"]
            feas = round((_feas(f1,o1,best["v1"]) + _feas(f2,o2,best["v2"]))/2, 1)
            multi.append({
                "features_changed": [f1,f2],
                "changes": {f1:{"from":o1,"to":best["v1"]},f2:{"from":o2,"to":best["v2"]}},
                "new_prob": round(best["prob"],4),
                "risk_reduction": round((orig_prob-best["prob"])*100,1),
                "feasibility": feas,
                "timeline": FEAT_META[f1]["timeline"] + " + " + FEAT_META[f2]["timeline"],
                "description": (FEAT_META[f1]["display"] + ": " + _desc(f1,o1,best["v1"])
                                + "  +  " + FEAT_META[f2]["display"] + ": " + _desc(f2,o2,best["v2"])),
                "x_counterfactual": xc.tolist(),
            })
    multi.sort(key=lambda r: r["feasibility"], reverse=True)

    # Safest path
    STEPS = {"HbA1c_level":0.1,"blood_glucose_level":5,"bmi":0.5,"hypertension":1,"smoking_enc":1}
    MINS  = {"HbA1c_level":3.5,"blood_glucose_level":79,"bmi":15.0,"hypertension":0,"smoking_enc":0}
    xw = x.copy()
    changes  = {}
    path_log = []
    for feat in ["HbA1c_level","blood_glucose_level","bmi","hypertension","smoking_enc"]:
        idx, ov = FEAT_IDX[feat], x[FEAT_IDX[feat]]
        v = xw[idx]
        while round(v - STEPS[feat], 2) >= MINS[feat]:
            v = round(v - STEPS[feat], 2)
            xw[idx] = v
        if xw[idx] != ov:
            changes[feat] = {"from": ov, "to": xw[idx]}
            path_log.append(FEAT_META[feat]["display"] + ": " + _desc(feat, ov, xw[idx]))
        if _prob(model, xw) < threshold:
            break
    fp = _prob(model, xw)
    safest = None
    if fp < threshold and changes:
        fl = list(changes.keys())
        safest = {
            "features_changed": fl,
            "changes": changes,
            "new_prob": round(fp,4),
            "risk_reduction": round((orig_prob-fp)*100,1),
            "feasibility": round(float(np.mean([_feas(f,changes[f]["from"],changes[f]["to"]) for f in fl])),1),
            "timeline": "Incremental - 3-6 months",
            "description": " | ".join(path_log),
            "path_log": path_log,
            "x_counterfactual": xw.tolist(),
        }

    # Summary
    summary = []
    if single:
        b = single[0]
        f = b["features_changed"][0]
        chg = b["changes"][f]
        summary.append({
            "rank": 1,
            "label": "Focus on " + FEAT_META[f]["display"] + " alone",
            "type": "single",
            "feasibility": b["feasibility"],
            "new_prob": b["new_prob"],
            "risk_reduction": b["risk_reduction"],
            "change_summary": b["description"],
            "timeline": b["timeline"],
            "doctor_note": "Target " + FEAT_META[f]["display"] + " of " + str(round(chg["to"],1)) + FEAT_META[f]["unit"] + ". " + b["mechanism"],
            "patient_note": b["mechanism"],
        })
    if multi:
        b = multi[0]
        summary.append({
            "rank": 2,
            "label": "Combined lifestyle intervention",
            "type": "multi",
            "feasibility": b["feasibility"],
            "new_prob": b["new_prob"],
            "risk_reduction": b["risk_reduction"],
            "change_summary": b["description"],
            "timeline": b["timeline"],
            "doctor_note": "Dual-target: " + b["description"],
            "patient_note": "Work on both: " + b["description"].replace("  +  ", " and "),
        })
    if safest:
        summary.append({
            "rank": 3,
            "label": "Gradual minimum-change pathway",
            "type": "safest_path",
            "feasibility": safest["feasibility"],
            "new_prob": safest["new_prob"],
            "risk_reduction": safest["risk_reduction"],
            "change_summary": safest["description"],
            "timeline": safest["timeline"],
            "doctor_note": "Minimum path: " + safest["description"],
            "patient_note": "Steps: " + " then ".join(safest["path_log"]),
        })

    return {
        "status": "high_risk",
        "original_prob": round(orig_prob,4),
        "threshold": round(threshold,4),
        "single": single,
        "multi": multi,
        "safest_path": safest,
        "summary": summary,
        "x_orig": x.tolist(),
    }


# ── Narratives ───────────────────────────────────────────────────
def _build_context(patient_dict, cf_results):
    p  = patient_dict
    op = cf_results["original_prob"] * 100

    hflag = ("above diabetic threshold (>=6.5%)" if p["HbA1c_level"] >= 6.5
             else "prediabetic range" if p["HbA1c_level"] >= 5.7 else "normal")
    gflag = ("diabetic range (>=200)" if p["blood_glucose_level"] >= 200
             else "prediabetic range" if p["blood_glucose_level"] >= 140 else "normal")
    bflag = ("obese" if p["bmi"] >= 30 else "overweight" if p["bmi"] >= 25 else "normal weight")

    recs_lines = []
    for r in cf_results.get("summary", []):
        recs_lines.append(
            "  #" + str(r["rank"]) + " " + r["label"] + ": " + r["change_summary"]
            + " -> " + str(round(r["new_prob"]*100,1)) + "% risk"
            + " | feasibility " + str(r["feasibility"]) + "/10"
            + " | " + r["timeline"]
        )
    recs = "\n".join(recs_lines)

    ctx = (
        "PATIENT: " + p["gender"] + ", " + str(p["age"]) + "y"
        + " | BMI " + str(p["bmi"]) + " (" + bflag + ")"
        + " | HTN: " + ("Yes" if p["hypertension"] else "No")
        + " | Smoking: " + p["smoking_history"] + "\n"
        + "HbA1c: " + str(p["HbA1c_level"]) + "% (" + hflag + ")"
        + " | Blood Glucose: " + str(p["blood_glucose_level"]) + " mg/dL (" + gflag + ")"
        + " | Heart Disease: " + ("Yes" if p["heart_disease"] else "No") + "\n"
        + "RISK: " + str(round(op,1)) + "% | Drivers: HbA1c (50.1%), Glucose (34.3%), Age (11%)\n"
        + "INTERVENTIONS:\n" + recs + "\n"
        + "CAVEATS: Population model, 100k patients. Clinician review required."
    )
    return ctx

def _call_gemini(gemini_model, prompt, temperature=0.4):
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
        resp = gemini_model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=temperature, max_output_tokens=700)
        )
        return resp.text.strip()
    except Exception as e:
        return "[Gemini error: " + str(e) + "]"

def generate_doctor_note(patient_dict, cf_results, gemini_model=None):
    p    = patient_dict
    recs = cf_results.get("summary", [])
    lines = "\n".join([
        "  " + str(r["rank"]) + ". " + r["change_summary"]
        + " -> " + str(round(r["new_prob"]*100,1)) + "% | " + r["timeline"]
        for r in recs
    ])
    fallback = (
        "CLINICAL SUMMARY NOTE\n"
        + "-"*50 + "\n"
        + "PATIENT SUMMARY\n"
        + p["gender"] + ", " + str(p["age"]) + "y"
        + " | BMI: " + str(p["bmi"])
        + " | HTN: " + ("Yes" if p["hypertension"] else "No")
        + " | Smoking: " + p["smoking_history"] + "\n"
        + "Diabetes Risk: " + str(round(cf_results["original_prob"]*100,1)) + "% (Very High)\n\n"
        + "KEY RISK DRIVERS\n"
        + "- HbA1c: " + str(p["HbA1c_level"]) + "% — " + ("above" if p["HbA1c_level"]>=6.5 else "near") + " diabetic threshold (6.5%)\n"
        + "- Blood Glucose: " + str(p["blood_glucose_level"]) + " mg/dL — " + ("diabetic" if p["blood_glucose_level"]>=200 else "elevated") + " range\n"
        + "- BMI: " + str(p["bmi"]) + " — " + ("obese" if p["bmi"]>=30 else "overweight" if p["bmi"]>=25 else "normal") + "\n"
        + ("- Hypertension: 4x baseline risk\n" if p["hypertension"] else "")
        + ("- Heart Disease: compound cardiometabolic risk\n" if p["heart_disease"] else "")
        + "\nRECOMMENDED INTERVENTIONS\n" + lines + "\n\n"
        + "NEXT STEPS\n"
        + "OGTT recommended. Lifestyle intervention first-line.\n"
        + "Retest HbA1c in 3 months. Consider metformin if targets unmet.\n\n"
        + "MODEL LIMITATIONS\n"
        + "Population-level model (n=100,541). Individual response may vary.\n"
        + "This tool supports — does not replace — clinical judgement."
    )
    if gemini_model is None:
        return fallback
    ctx = _build_context(patient_dict, cf_results)
    prompt = (
        "Write a structured CLINICAL SUMMARY NOTE for a physician interpreting a diabetes risk assessment.\n"
        "Use these section headers: PATIENT SUMMARY, KEY RISK DRIVERS, RECOMMENDED INTERVENTIONS, NEXT STEPS, MODEL LIMITATIONS.\n"
        "Tone: precise, clinical. Length: 250-300 words.\n\nDATA:\n" + ctx + "\n\nWrite the note now:"
    )
    result = _call_gemini(gemini_model, prompt, temperature=0.25)
    return result if not result.startswith("[Gemini error") else fallback

def generate_patient_report(patient_dict, cf_results, gemini_model=None):
    p    = patient_dict
    recs = cf_results.get("summary", [])
    top  = recs[0] if recs else {}
    fallback = (
        "YOUR DIABETES RISK REPORT\n"
        + "-"*50 + "\n"
        + "YOUR RESULTS\n"
        + "Your diabetes risk score is " + str(round(cf_results["original_prob"]*100,1)) + "%. This is in the high range.\n"
        + "The good news: your risk is driven by things that can change.\n\n"
        + "WHAT THIS MEANS\n"
        + "Two numbers are most responsible for your score:\n"
        + "- HbA1c (" + str(p["HbA1c_level"]) + "%): your average blood sugar over 3 months\n"
        + "- Blood glucose (" + str(p["blood_glucose_level"]) + " mg/dL): your current fasting blood sugar\n"
        + "Both can be reduced through diet and lifestyle changes.\n\n"
        + "YOUR MOST ACHIEVABLE STEP\n"
        + top.get("patient_note", "Focus on diet and exercise as advised by your doctor.") + "\n\n"
        + "WHAT TO EXPECT\n"
        + "Timeline: " + top.get("timeline","3-6 months") + "\n"
        + "Projected risk after changes: " + str(round(top.get("new_prob",0)*100,1)) + "%\n\n"
        + "A FINAL NOTE\n"
        + "These are the minimum changes needed, not a complete lifestyle overhaul.\n"
        + "Start with one step, track your progress, and revisit with your doctor in 6-8 weeks."
    )
    if gemini_model is None:
        return fallback
    ctx = _build_context(patient_dict, cf_results)
    prompt = (
        "Write a PATIENT HEALTH REPORT in plain language (7th-grade reading level).\n"
        "Use these headers: YOUR RESULTS, WHAT THIS MEANS, YOUR ACTION PLAN, WHAT TO EXPECT, A FINAL NOTE.\n"
        "Tone: warm, empowering. No jargon. Length: 280-350 words.\n\nDATA:\n" + ctx + "\n\nWrite the report now:"
    )
    result = _call_gemini(gemini_model, prompt, temperature=0.55)
    return result if not result.startswith("[Gemini error") else fallback

def generate_whatif_text(orig_prob, new_prob, changes, gemini_model=None):
    delta     = (orig_prob - new_prob) * 100
    direction = "reduced" if delta > 0 else "increased"
    ch_text   = ", ".join([
        k.replace("_"," ") + ": " + str(v["from"]) + " -> " + str(v["to"])
        for k, v in changes.items()
    ])
    fallback = (
        "These changes " + direction + " your diabetes risk by " + str(round(abs(delta),1)) + " percentage points. "
        + "Adjusting " + ch_text + " moves your biomarkers "
        + ("closer to" if delta > 0 else "further from")
        + " the healthy range, which the model recognises as "
        + ("lower" if delta > 0 else "higher") + " risk."
    )
    if gemini_model is None:
        return fallback
    prompt = (
        "In 3 plain-English sentences, explain WHY changing " + ch_text
        + " " + ("reduced" if delta>0 else "increased")
        + " diabetes risk from " + str(round(orig_prob*100,1))
        + "% to " + str(round(new_prob*100,1)) + "%. "
        + "Be specific. No bullet points."
    )
    result = _call_gemini(gemini_model, prompt, temperature=0.4)
    return result if not result.startswith("[Gemini error") else fallback


# ── Charts ───────────────────────────────────────────────────────
def gauge_chart(prob, tier, color):
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = round(prob*100, 1),
        delta  = {"reference": 87.6,
                  "increasing": {"color": "#e74c3c"},
                  "decreasing": {"color": "#27ae60"}},
        title  = {"text": "<b>" + tier + "</b><br><span style='font-size:14px'>Diabetes Risk Probability</span>",
                  "font": {"size": 18}},
        number = {"suffix": "%", "font": {"size": 52, "color": color}},
        gauge  = {
            "axis": {"range": [0,100], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1e1e2e",
            "steps": [
                {"range": [0,15],   "color": "#1a3a2a"},
                {"range": [15,40],  "color": "#3a2e10"},
                {"range": [40,70],  "color": "#3a2010"},
                {"range": [70,100], "color": "#3a1010"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3}, "value": 87.6},
        },
    ))
    fig.update_layout(
        height=300, margin=dict(t=40,b=0,l=30,r=30),
        paper_bgcolor="#0d1117", font_color="#e0e0e0",
    )
    return fig

def feature_contribution_chart(patient_dict):
    labels = ["HbA1c","Blood Glucose","Age","BMI","Hypertension","Heart Disease","Smoking","Gender"]
    raw    = [
        patient_dict["HbA1c_level"],
        patient_dict["blood_glucose_level"],
        patient_dict["age"],
        patient_dict["bmi"],
        patient_dict["hypertension"] * 100,
        patient_dict["heart_disease"] * 100,
        {"never":0,"No Info":25,"current":75,"former":50}.get(patient_dict["smoking_history"], 0),
        {"Female":40,"Male":60}.get(patient_dict["gender"], 50),
    ]
    imps   = [50.1, 34.3, 11.0, 2.7, 0.9, 0.4, 0.3, 0.1]
    maxv   = max(raw) if max(raw) > 0 else 1
    contribs = [round((v/maxv)*i, 2) for v, i in zip(raw, imps)]
    fig = px.bar(
        x=contribs, y=labels, orientation="h",
        color=contribs,
        color_continuous_scale=["#27ae60","#f39c12","#e74c3c"],
        title="Feature Risk Contributions",
        labels={"x":"Risk Contribution Score","y":""},
    )
    fig.update_layout(
        height=320, margin=dict(t=40,b=10,l=10,r=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e0e0e0", coloraxis_showscale=False,
    )
    return fig

def cf_comparison_chart(orig_prob, summary):
    labels = ["Current Risk"] + ["#" + str(r["rank"]) + " " + r["label"][:22] + "..." for r in summary]
    values = [orig_prob*100] + [r["new_prob"]*100 for r in summary]
    colors = ["#e74c3c"] + [
        "#27ae60" if v < 40 else "#f39c12" if v < 70 else "#e67e22"
        for v in values[1:]
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[str(round(v,1))+"%" for v in values],
        textposition="outside",
        textfont={"size":13,"color":"#e0e0e0"},
    ))
    fig.add_hline(y=87.6, line_dash="dash", line_color="#aaa",
                  annotation_text="Threshold (87.6%)", annotation_font_color="#aaa")
    fig.update_layout(
        title="Risk After Each Intervention",
        height=320, margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e0e0e0",
        yaxis=dict(range=[0,115], title="Risk %"),
        showlegend=False,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:10px 0 20px">
      <h2 style="margin:0">🩺 Praxis 2.0</h2>
      <p style="color:#aaa;margin:4px 0 0;font-size:13px">
        Counterfactual Clinical Decision Support
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### Patient Profile")
    gender  = st.sidebar.selectbox("Gender", ["Male","Female"])
    age     = st.sidebar.slider("Age", 1, 81, 45)
    bmi     = st.sidebar.slider("BMI (kg/m2)", 10.0, 70.0, 28.0, 0.1)
    hba1c   = st.sidebar.slider("HbA1c Level (%)", 3.5, 9.0, 5.5, 0.1)
    glucose = st.sidebar.slider("Blood Glucose (mg/dL)", 79, 303, 130)
    smoking = st.sidebar.selectbox("Smoking History", SMOKING_OPTIONS)
    htn     = st.sidebar.checkbox("Hypertension")
    hd      = st.sidebar.checkbox("Heart Disease")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    use_gemini = st.sidebar.toggle("Enable Gemini AI Narratives", value=False)
    api_key    = ""
    if use_gemini:
        api_key = st.sidebar.text_input("Gemini API Key", type="password",
                                         value=GEMINI_API_KEY,
                                         help="Free key at aistudio.google.com")
    st.sidebar.markdown("---")
    analyse = st.sidebar.button("Analyse Risk", use_container_width=True, type="primary")

    patient = {
        "gender": gender, "age": age, "bmi": bmi,
        "HbA1c_level": hba1c, "blood_glucose_level": glucose,
        "smoking_history": smoking,
        "hypertension": int(htn), "heart_disease": int(hd),
    }
    return patient, analyse, use_gemini, api_key


# ── Tab 1: Risk Overview ─────────────────────────────────────────
def render_tab_overview(patient, result, cf_results):
    st.markdown("### Risk Overview")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.plotly_chart(
            gauge_chart(result["probability"], result["risk_tier"], result["risk_color"]),
            use_container_width=True
        )
    with col2:
        st.markdown("#### Clinical Indicators")
        hba_label = ("Diabetic" if patient["HbA1c_level"]>=6.5
                     else "Prediabetic" if patient["HbA1c_level"]>=5.7 else "Normal")
        st.metric("HbA1c Level", str(patient["HbA1c_level"]) + "%", hba_label)

        glc_label = ("Diabetic" if patient["blood_glucose_level"]>=200
                     else "Prediabetic" if patient["blood_glucose_level"]>=140 else "Normal")
        st.metric("Blood Glucose", str(patient["blood_glucose_level"]) + " mg/dL", glc_label)

        bmi_label = ("Obese" if patient["bmi"]>=30
                     else "Overweight" if patient["bmi"]>=25 else "Normal")
        st.metric("BMI", str(round(patient["bmi"],1)), bmi_label)

        st.markdown("---")
        comorbidities = []
        if patient["hypertension"]:  comorbidities.append("Hypertension")
        if patient["heart_disease"]: comorbidities.append("Heart Disease")
        if comorbidities:
            st.warning("Comorbidities: " + ", ".join(comorbidities))
        else:
            st.success("No comorbidities recorded")

    st.plotly_chart(feature_contribution_chart(patient), use_container_width=True)

    actions = {
        "Low Risk":       "Routine annual screening. Maintain current lifestyle.",
        "Moderate Risk":  "Lifestyle modification advised. Retest HbA1c in 6 months.",
        "High Risk":      "Consult physician soon. Dietary and exercise intervention required.",
        "Very High Risk": "Urgent clinical evaluation. Possible medication indicated.",
    }
    color = result["risk_color"]
    tier  = result["risk_tier"]
    st.markdown(
        '<div class="risk-card" style="border-color:' + color + '">'
        + '<b style="color:' + color + '">' + tier + '</b><br>'
        + actions[tier] + '</div>',
        unsafe_allow_html=True
    )


# ── Tab 2: Counterfactual Interventions ─────────────────────────
def render_tab_counterfactuals(result, cf_results):
    st.markdown("### Counterfactual Interventions")

    if cf_results.get("status") == "low_risk":
        st.success("Patient is below risk threshold (" + str(round(cf_results["original_prob"]*100,1)) + "%). No interventions needed.")
        return

    summary = cf_results.get("summary", [])
    if not summary:
        st.warning("No counterfactuals generated.")
        return

    st.markdown(
        "**Original Risk: " + str(round(cf_results["original_prob"]*100,1)) + "%** — "
        "changes below would bring this under the "
        + str(round(cf_results["threshold"]*100,1)) + "% threshold."
    )

    st.plotly_chart(cf_comparison_chart(cf_results["original_prob"], summary), use_container_width=True)
    st.markdown("#### Ranked Interventions")

    FEAS_COLOR = lambda f: "#27ae60" if f>=7 else "#f39c12" if f>=4 else "#e74c3c"
    TYPE_LABEL = {
        "single":      "Single-feature change",
        "multi":       "Combined intervention",
        "safest_path": "Minimum-change pathway",
    }

    for rec in summary:
        fc  = FEAS_COLOR(rec["feasibility"])
        tl  = TYPE_LABEL.get(rec["type"], "Intervention")
        np_ = round(rec["new_prob"]*100, 1)
        rr  = rec["risk_reduction"]

        st.markdown(
            '<div class="cf-card">'
            + '<span class="feasibility-badge" style="background:' + fc + '22;color:' + fc + ';border:1px solid ' + fc + '">'
            + 'Feasibility: ' + str(rec["feasibility"]) + '/10</span>'
            + '&nbsp;<span class="feasibility-badge" style="background:#1a2a3a;color:#7ec8e3;border:1px solid #7ec8e3">'
            + tl + '</span>'
            + '<h4 style="margin:10px 0 6px">#' + str(rec["rank"]) + ' — ' + rec["label"] + '</h4>'
            + '<p style="color:#ccc;margin:4px 0">Change: ' + rec["change_summary"] + '</p>'
            + '<p style="color:#ccc;margin:4px 0">New risk: ' + str(np_) + '%  |  Risk reduction: down ' + str(rr) + 'pp</p>'
            + '<p style="color:#aaa;margin:4px 0">Timeline: ' + rec["timeline"] + '</p>'
            + '</div>',
            unsafe_allow_html=True
        )
        with st.expander("Details for Recommendation #" + str(rec["rank"])):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**For the Clinician:**")
                st.info(rec.get("doctor_note", "-"))
            with c2:
                st.markdown("**For the Patient:**")
                st.success(rec.get("patient_note", "-"))


# ── Tab 3: What-If Simulator ─────────────────────────────────────
def render_tab_whatif(patient, cf_results, artifacts, gemini_model=None):
    st.markdown("### What-If Simulator")
    st.markdown("Adjust sliders to explore how changes in modifiable factors affect risk.")

    orig_prob = cf_results.get("original_prob", 0)
    orig_x    = np.array(cf_results.get("x_orig", [0]*8), dtype=float)
    enc       = artifacts["encoders"]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Adjust Features")
        wi_hba1c   = st.slider("HbA1c Level (%)",       3.5,  9.0, float(patient["HbA1c_level"]),          0.1, key="wi_hba")
        wi_glucose = st.slider("Blood Glucose (mg/dL)", 79,   303, int(patient["blood_glucose_level"]),         key="wi_glu")
        wi_bmi     = st.slider("BMI (kg/m2)",           10.0, 70.0, float(patient["bmi"]),                  0.5, key="wi_bmi")
        wi_smoking = st.selectbox("Smoking History", SMOKING_OPTIONS,
                                   index=SMOKING_OPTIONS.index(patient["smoking_history"]), key="wi_smk")
        wi_htn     = st.checkbox("Hypertension", value=bool(patient["hypertension"]), key="wi_htn")

    x_wi      = orig_x.copy()
    x_wi[5]   = wi_bmi
    x_wi[6]   = wi_hba1c
    x_wi[7]   = wi_glucose
    x_wi[4]   = enc["smoking_map"].get(wi_smoking, 3)
    x_wi[2]   = int(wi_htn)

    new_prob, new_tier, new_color = predict_from_vector(x_wi, artifacts)
    delta_pp  = (orig_prob - new_prob) * 100
    direction = "down" if delta_pp > 0 else "up"
    arr_color = "#27ae60" if delta_pp > 0 else "#e74c3c"

    with col2:
        st.markdown("#### Live Risk Update")
        st.markdown(
            '<div style="text-align:center;padding:20px;background:#1e1e2e;'
            'border-radius:12px;border:2px solid ' + new_color + '">'
            + '<p style="color:#aaa;margin:0;font-size:14px">Modified Risk</p>'
            + '<p style="font-size:56px;font-weight:800;color:' + new_color + ';margin:8px 0">'
            + str(round(new_prob*100,1)) + '%</p>'
            + '<p style="font-size:18px;color:' + new_color + '">' + new_tier + '</p>'
            + '<p style="color:' + arr_color + ';font-size:18px;margin-top:8px">'
            + direction + ' ' + str(round(abs(delta_pp),1)) + 'pp from original ('
            + str(round(orig_prob*100,1)) + '%)</p>'
            + '</div>',
            unsafe_allow_html=True
        )

        changes_made = {}
        if wi_hba1c   != patient["HbA1c_level"]:
            changes_made["HbA1c_level"]        = {"from": patient["HbA1c_level"],        "to": wi_hba1c}
        if wi_glucose != patient["blood_glucose_level"]:
            changes_made["blood_glucose_level"] = {"from": patient["blood_glucose_level"],"to": wi_glucose}
        if wi_bmi     != patient["bmi"]:
            changes_made["bmi"]                 = {"from": patient["bmi"],                "to": wi_bmi}
        if wi_smoking != patient["smoking_history"]:
            changes_made["smoking_history"]     = {"from": patient["smoking_history"],    "to": wi_smoking}
        if int(wi_htn) != patient["hypertension"]:
            changes_made["hypertension"]        = {"from": patient["hypertension"],       "to": int(wi_htn)}

        if changes_made and abs(delta_pp) > 0.5:
            st.markdown("#### Why did risk change?")
            explanation = generate_whatif_text(orig_prob, new_prob, changes_made, gemini_model)
            st.info(explanation)
        elif not changes_made:
            st.info("Adjust the sliders to see how risk changes.")


# ── Tab 4: Reports ───────────────────────────────────────────────
def render_tab_reports(patient, cf_results, gemini_model=None):
    st.markdown("### Clinical and Patient Reports")

    view = st.radio("Select report:", ["Clinical Note (Doctor)", "Patient Report"], horizontal=True)

    cache_key = "narratives_" + str(hash(str(sorted(patient.items()))))
    if cache_key not in st.session_state:
        with st.spinner("Generating report..."):
            st.session_state[cache_key] = {
                "doctor" : generate_doctor_note(patient, cf_results, gemini_model),
                "patient": generate_patient_report(patient, cf_results, gemini_model),
            }

    narratives = st.session_state[cache_key]
    source_label = "Gemini AI" if gemini_model else "Template"

    if view == "Clinical Note (Doctor)":
        st.caption("For clinician review | Source: " + source_label)
        st.markdown(
            '<div class="narrative-box">' + narratives["doctor"] + '</div>',
            unsafe_allow_html=True
        )
        st.download_button("Download Clinical Note", data=narratives["doctor"],
                           file_name="clinical_note.txt", mime="text/plain")
    else:
        st.caption("Plain language for patient | Source: " + source_label)
        st.markdown(
            '<div class="narrative-box">' + narratives["patient"] + '</div>',
            unsafe_allow_html=True
        )
        st.download_button("Download Patient Report", data=narratives["patient"],
                           file_name="patient_report.txt", mime="text/plain")


# ── Main ─────────────────────────────────────────────────────────
def main():
    try:
        artifacts = load_model_artifacts()
    except FileNotFoundError:
        st.error("Model artifacts not found at: " + ARTIFACT_PATH + ". Run Module 1 first.")
        st.stop()

    patient, analyse, use_gemini, api_key = render_sidebar()

    gemini_model = None
    if use_gemini and api_key and api_key != "YOUR_API_KEY_HERE":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            st.sidebar.error("Gemini init failed: " + str(e))

    if "result" not in st.session_state or analyse:
        if not analyse:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px">
              <h1>Praxis 2.0</h1>
              <h3 style="color:#aaa">Counterfactual Clinical Decision Support System</h3>
              <p style="color:#666;max-width:600px;margin:20px auto">
                Enter a patient profile in the sidebar and click
                <b>Analyse Risk</b> to generate a diabetes risk assessment
                with personalised intervention pathways.
              </p>
              <p style="color:#444;font-size:13px;margin-top:40px">
                Praxis 2.0 · GDG on Campus GB Pant ·
                ML: Gradient Boosting (AUC 0.979) · GenAI: Gemini 1.5 Flash
              </p>
            </div>
            """, unsafe_allow_html=True)
            return

        with st.spinner("Analysing patient risk..."):
            result = predict_patient(patient, artifacts)
            st.session_state["result"]  = result
            st.session_state["patient"] = patient

        with st.spinner("Computing counterfactual interventions..."):
            cf_results = run_counterfactuals(patient, artifacts)
            st.session_state["cf_results"] = cf_results

        for k in list(st.session_state.keys()):
            if k.startswith("narratives_"):
                del st.session_state[k]

    result     = st.session_state.get("result", {})
    cf_results = st.session_state.get("cf_results", {})
    patient    = st.session_state.get("patient", patient)

    if not result:
        return

    t1, t2, t3, t4 = st.tabs([
        "Risk Overview",
        "Interventions",
        "What-If Simulator",
        "Reports",
    ])
    with t1: render_tab_overview(patient, result, cf_results)
    with t2: render_tab_counterfactuals(result, cf_results)
    with t3: render_tab_whatif(patient, cf_results, artifacts, gemini_model)
    with t4: render_tab_reports(patient, cf_results, gemini_model)


if __name__ == "__main__":
    main()
