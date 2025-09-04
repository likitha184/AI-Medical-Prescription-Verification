import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("💊 AI Medical Prescription Verifier")

# -------------------------------
# Upload Prescription (OCR)
# -------------------------------
st.header("1. Upload Prescription (OCR)")

def extract_text(file):
    # Debug log
    st.write(f"📡 Sending request to: {BACKEND_URL}/ocr_extract")
    res = requests.post(f"{BACKEND_URL}/ocr_extract", files={"file": file.getvalue()})
    return res.json()

file = st.file_uploader("Upload prescription image", type=["jpg","png","jpeg"])
if file and st.button("Extract Text"):
    result = extract_text(file)
    st.json(result)

# -------------------------------
# NLP-Based Extraction
# -------------------------------
st.header("2. NLP-Based Drug Info Extraction")
text = st.text_area("Enter prescription text")
if st.button("Extract Info"):
    st.write(f"📡 Sending request to: {BACKEND_URL}/extract_drug_info")
    res = requests.post(f"{BACKEND_URL}/extract_drug_info", json={"text": text})
    st.json(res.json())

# -------------------------------
# Drug Interaction Check
# -------------------------------
st.header("3. Drug Interaction Detection")
drug_list = st.text_input("Enter drugs (comma-separated)")
if st.button("Check Interactions"):
    drugs = [d.strip() for d in drug_list.split(",")]
    st.write(f"📡 Sending request to: {BACKEND_URL}/check_interactions with {drugs}")
    res = requests.post(f"{BACKEND_URL}/check_interactions", json={"drugs": drugs})
    st.json(res.json())

# -------------------------------
# Age-Specific Dosage
# -------------------------------
st.header("4. Age-Specific Dosage Recommendation")
drug_name = st.text_input("Drug name for dosage")
age = st.number_input("Enter patient age", min_value=0, max_value=120)
if st.button("Get Dosage"):
    st.write(f"📡 Sending request to: {BACKEND_URL}/dosage_recommendation with drug={drug_name}, age={age}")
    res = requests.post(f"{BACKEND_URL}/dosage_recommendation", json={"drug": drug_name, "age": age})
    st.json(res.json())

# -------------------------------
# Alternative Suggestions
# -------------------------------
st.header("5. Alternative Medication Suggestions")
alt_drug = st.text_input("Drug name for alternatives")
if st.button("Suggest Alternatives"):
    st.write(f"📡 Sending request to: {BACKEND_URL}/alternative_suggestions with drug={alt_drug}")
    res = requests.post(f"{BACKEND_URL}/alternative_suggestions", json={"drug": alt_drug})
    st.json(res.json())