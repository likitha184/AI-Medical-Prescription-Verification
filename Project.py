import streamlit as st
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer_utils import set_seed
import torch
from huggingface_hub import HfApi, login

# Hugging Face Token (hardcoded)
HF_TOKEN = "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"

# ---------------------------
# Hugging Face Token Validation
# ---------------------------
def validate_huggingface_token():
    if not HF_TOKEN.startswith("hf_"):
        st.error("Invalid Hugging Face token format.")
        return False
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        user = api.whoami()
        st.success(f"✅ Hugging Face login successful! Logged in as: {user['name']}")
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

# ---------------------------
# Entity Extraction (NER)
# ---------------------------
def extract_entities(text):
    try:
        ner = pipeline(
            "token-classification",
            model="blaze999/Medical-NER",
            aggregation_strategy="simple",
            token=HF_TOKEN  # ✅ correct argument
        )
        entities = ner(text)
    except Exception as e:
        st.error(f"Entity extraction error: {e}")
        entities = []
    # Organize entities (depends on model labels)
    drugs = [e for e in entities if 'DRUG' in e.get('entity_group', '')]
    return drugs, entities

# ---------------------------
# Normalize Drug Names (Dummy)
# ---------------------------
def normalize_drug_names(drugs):
    return [{"brand": d['word'], "generic": d['word'].lower()} for d in drugs]

# ---------------------------
# Validate Dosages (Dummy)
# ---------------------------
def validate_dosages(dosages, age=None, weight=None, conditions=None):
    return [{"dosage": d['word'], "valid": True, "reason": "Within safe limits"} for d in dosages]

# ---------------------------
# Check Drug Interactions (Dummy)
# ---------------------------
def check_interactions(drugs):
    return ["No interactions detected"]

# ---------------------------
# IBM Watson NLP Stub
# ---------------------------
def ibm_watson_analysis(text):
    return "IBM Watson NLP analysis (stub)."

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 AI Medical Prescription Analyzer")

if validate_huggingface_token():
    uploaded = st.file_uploader("Upload Prescription (image/pdf)", type=["jpg","png","jpeg","pdf"])
    extracted_text = ""
    
    if uploaded:
        from pdf2image import convert_from_bytes
        import io
        try:
            if uploaded.type == "application/pdf":
                images = convert_from_bytes(uploaded.read())
                img = images[0]
            else:
                img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded Prescription")
            extracted_text = st.text_area("Enter prescription text:")
        except Exception as e:
            st.error(f"File processing error: {e}")

    age = st.text_input("Patient Age:")
    weight = st.text_input("Patient Weight (kg):")
    conditions = st.text_area("Special Conditions:")

    if extracted_text and st.button("🔍 Extract & Analyze"):
        with st.spinner("Extracting entities..."):
            drugs, entities = extract_entities(extracted_text)
            st.subheader("Entities Extracted")
            st.write("Drugs:", drugs)
            st.write("All Entities:", entities)

        with st.spinner("Normalizing drug names..."):
            normalized = normalize_drug_names(drugs)
            st.subheader("Normalized Drug Names")
            st.write(normalized)

        with st.spinner("Validating dosages..."):
            dosage_res = validate_dosages(drugs, age, weight, conditions)
            st.subheader("Dosage Validation")
            st.write(dosage_res)

        with st.spinner("Checking interactions..."):
            inter_res = check_interactions([d['word'] for d in drugs])
            st.subheader("Drug Interactions")
            st.write(inter_res)

        with st.spinner("IBM Watson NLP..."):
            watson_res = ibm_watson_analysis(extracted_text)
            st.subheader("IBM Watson Analysis")
            st.write(watson_res)

    if extracted_text and st.button("🤖 Analyze with AI"):
        with st.spinner("Loading AI model..."):
            try:
                mdl = "ibm-granite/granite-3.3-2b-instruct"
                device = "cpu"
                model = AutoModelForCausalLM.from_pretrained(mdl, torch_dtype=torch.float32).to(device)
                tokenizer = AutoTokenizer.from_pretrained(mdl)

                input_ids = tokenizer.apply_chat_template(
                    [{"role":"user","content":f"Please analyze:\n\n{extracted_text}"}],
                    return_tensors="pt", return_dict=True
                )
                input_ids = {k: v.to(device) for k,v in input_ids.items()}
                set_seed(42)
                output = model.generate(**input_ids, max_new_tokens=256)
                start = input_ids["input_ids"].shape[1]
                prediction = tokenizer.decode(output[0, start:], skip_special_tokens=True) if output.shape[1] > start else "No output"

                st.subheader("AI Analysis")
                st.write(prediction)
            except Exception as e:
                st.error(f"AI model error: {e}")

