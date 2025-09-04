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

    # Regex fallback for common drug-like words (ends with -ine, -ol, etc.)
    regex_drugs = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:ine|ol|cin|vir|mycin|azole)\b", text)
    drugs.extend(regex_drugs)

    return list(set(drugs))

def extract_diseases(entities, text):
    diseases = []
    for e in entities:
        label = e.get("entity_group", "").upper()
        if any(key in label for key in ["DISEASE", "CONDITION", "SYMPTOM"]):
            diseases.append(e["word"])

    # Regex fallback for disease words
    regex_diseases = re.findall(r"\b(fever|diabetes|hypertension|asthma|cough|infection)\b", text, re.IGNORECASE)
    diseases.extend(regex_diseases)

    return list(set(diseases))

def extract_dosage_info(text, entities):
    dosages = [e["word"] for e in entities if "STRENGTH" in e.get("entity_group", "").upper()]

    # Regex fallback
    regex_matches = re.findall(r"\b\d+\s?(mg|ml|g|mcg|tablet|capsule|unit|units)\b", text, re.IGNORECASE)
    dosages.extend(regex_matches)

    return list(set(dosages))

def check_drug_interactions(drugs):
    if len(drugs) > 1:
        return [f"⚠ Potential interaction between {drugs[0]} and {drugs[1]}"]
    return ["No interactions detected"]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Prescription Analyser", page_icon="🩺")
st.title("🩺 AI Medical Prescription Analyser")

prescription_text = st.text_area("Enter prescription text:")

if st.button("Analyze"):
    if prescription_text.strip():
        with st.spinner("Analyzing prescription..."):
            entities = extract_entities(prescription_text)
            drugs = normalize_drug_names(entities, prescription_text)
            dosages = extract_dosage_info(prescription_text, entities)
            diseases = extract_diseases(entities, prescription_text)
            interactions = check_drug_interactions(drugs)

            # Generate description
            description = "### 📋 Prescription Analysis Report\n\n"
            description += f"*Detected Drugs:* {', '.join(drugs) if drugs else 'None found'}\n\n"
            description += f"*Detected Dosages:* {', '.join(dosages) if dosages else 'None found'}\n\n"
            description += f"*Detected Diseases/Conditions:* {', '.join(diseases) if diseases else 'None found'}\n\n"
            description += f"*Drug Interaction Check:* {', '.join(interactions)}\n\n"
            description += "⚠ Disclaimer: This analysis is AI-generated and not a substitute for professional medical advice."

            st.markdown(description)
    else:
        st.warning("⚠ Please enter a prescription text to analyze.")
