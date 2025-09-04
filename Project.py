import os
import re
import streamlit as st
from transformers import pipeline

HF_TOKEN = "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"

# ---------------- Load NER Model ----------------
@st.cache_resource
def load_ner_pipeline():
    return pipeline(
        "token-classification",
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        token=HF_TOKEN,
        device=-1
    )

def extract_entities(text):
    ner = load_ner_pipeline()
    return ner(text)

# ---------------- Helpers ----------------
def normalize_drug_names(entities, text):
    drugs = []
    for e in entities:
        label = e.get("entity_group", "").upper()
        if any(key in label for key in ["MEDICATION", "DRUG", "CHEMICAL"]):
            drugs.append(e["word"])
    regex_drugs = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:ine|ol|cin|vir|mycin|azole)\b", text)
    drugs.extend(regex_drugs)
    return list(set(drugs))

def extract_diseases(entities, text):
    diseases = []
    for e in entities:
        label = e.get("entity_group", "").upper()
        if any(key in label for key in ["DISEASE", "CONDITION", "SYMPTOM"]):
            diseases.append(e["word"])
    regex_diseases = re.findall(r"\b(fever|diabetes|hypertension|asthma|cough|infection)\b", text, re.IGNORECASE)
    diseases.extend(regex_diseases)
    return list(set(diseases))

def extract_dosage_info(text, entities):
    dosages = [e["word"] for e in entities if "STRENGTH" in e.get("entity_group", "").upper()]
    regex_matches = re.findall(r"\b\d+\s?(mg|ml|g|mcg|tablet|capsule|unit|units)\b", text, re.IGNORECASE)
    dosages.extend(regex_matches)
    return list(set(dosages))

def extract_frequency(text):
    frequency_patterns = {
        r"\bonce (daily|a day)\b": ["Morning"],
        r"\btwice (daily|a day)\b": ["Morning", "Evening"],
        r"\bthrice (daily|a day)\b": ["Morning", "Afternoon", "Night"],
        r"\bevery 8 hours\b": ["Morning", "Afternoon", "Night"],
        r"\bevery 12 hours\b": ["Morning", "Night"],
        r"\bat night\b": ["Night"],
        r"\bin morning\b": ["Morning"],
        r"\bin evening\b": ["Evening"],
        r"\bafter food\b": ["After meals"],
        r"\bbefore food\b": ["Before meals"],
    }
    schedule = []
    for pattern, times in frequency_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            schedule.extend(times)
    return list(set(schedule)) if schedule else ["Not specified"]

def check_drug_interactions(drugs):
    if len(drugs) > 1:
        return [f"⚠️ Potential interaction between {drugs[0]} and {drugs[1]}"]
    return ["No interactions detected"]

# ---------------- NEW FEATURE: Age Suitability ----------------
def check_age_suitability(drug, dosage, age_group):
    rules = {
        "Paracetamol": {
            "children": "Safe in lower doses (10-15mg/kg). Avoid overdose.",
            "adults": "Safe up to 500–1000 mg every 6–8 hours.",
            "elderly": "Generally safe but monitor liver function."
        },
        "Ibuprofen": {
            "children": "Avoid under 6 months. Use syrup form for kids.",
            "adults": "Safe 200–400 mg every 6–8 hours.",
            "elderly": "Use cautiously. May cause stomach/kidney issues."
        },
        "Amoxicillin": {
            "children": "Safe but dosage based on weight.",
            "adults": "Safe 250–500 mg every 8 hours.",
            "elderly": "Safe but monitor kidney function."
        }
    }
    drug = drug.capitalize()
    if drug in rules and age_group in rules[drug]:
        return rules[drug][age_group]
    return "No specific age guideline available."

# ---------------- NEW FEATURE: Side Effects ----------------
def get_side_effects(drug):
    side_effects_db = {
        "Paracetamol": ["Nausea", "Liver damage (overdose)", "Allergic reaction (rare)"],
        "Ibuprofen": ["Stomach pain", "Heartburn", "Kidney issues", "Increased blood pressure"],
        "Amoxicillin": ["Diarrhea", "Nausea", "Rash", "Yeast infection"],
        "Metformin": ["Stomach upset", "Diarrhea", "Low blood sugar", "Vitamin B12 deficiency"]
    }
    drug = drug.capitalize()
    return side_effects_db.get(drug, ["No side effect info available"])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Prescription Analyser", page_icon="🩺")
st.title("🩺 AI Medical Prescription Analyser")

prescription_text = st.text_area("Enter prescription text:")
age_group = st.selectbox("Select Patient Age Group", ["children", "adults", "elderly"])

if st.button("Analyze"):
    if prescription_text.strip():
        with st.spinner("Analyzing prescription..."):
            entities = extract_entities(prescription_text)
            drugs = normalize_drug_names(entities, prescription_text)
            dosages = extract_dosage_info(prescription_text, entities)
            diseases = extract_diseases(entities, prescription_text)
            interactions = check_drug_interactions(drugs)
            schedule = extract_frequency(prescription_text)

            # Generate description
            description = "### 📋 Prescription Analysis Report\n\n"
            description += f"**Detected Drugs:** {', '.join(drugs) if drugs else 'None found'}\n\n"
            description += f"**Detected Dosages:** {', '.join(dosages) if dosages else 'None found'}\n\n"
            description += f"**Detected Diseases/Conditions:** {', '.join(diseases) if diseases else 'None found'}\n\n"
            description += f"**Drug Interaction Check:** {', '.join(interactions)}\n\n"
            description += f"**Suggested Schedule:** {', '.join(schedule)}\n\n"

            # Age suitability
            if drugs:
                description += "### 🧒 Age Suitability\n"
                for d in drugs:
                    guideline = check_age_suitability(d, dosages, age_group)
                    description += f"- {d}: {guideline}\n"
                description += "\n"

            # Side effects
            if drugs:
                description += "### ⚠️ Possible Side Effects\n"
                for d in drugs:
                    effects = get_side_effects(d)
                    description += f"- {d}: {', '.join(effects)}\n"
                description += "\n"

            description += "⚠️ *Disclaimer: This analysis is AI-generated and not a substitute for professional medical advice.*"

            st.markdown(description)
    else:
        st.warning("⚠️ Please enter a prescription text to analyze.")
