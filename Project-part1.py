from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import easyocr
from transformers import pipeline
import io

app = FastAPI()

# -------------------------------
# Mock drug database
# -------------------------------
drug_db = {
    "paracetamol": {
        "max_dose_mg": {"adult": 4000, "child": 2000},
        "alternatives": ["acetaminophen"],
        "interactions": ["ibuprofen"]
    },
    "ibuprofen": {
        "max_dose_mg": {"adult": 3200, "child": 1200},
        "alternatives": ["naproxen"],
        "interactions": ["paracetamol"]
    }
}

# -------------------------------
# 1. Drug Interaction Detection
# -------------------------------
class DrugList(BaseModel):
    drugs: list

@app.post("/check_interactions")
def check_interactions(data: DrugList):
    try:
        drugs = [d.lower() for d in data.drugs]
        flagged = []
        for d in drugs:
            if d in drug_db:
                for inter in drug_db[d]["interactions"]:
                    if inter in drugs:
                        flagged.append(f"{d} interacts with {inter}")
        return {"interactions": flagged or ["No harmful interactions detected"]}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# 2. Age-Specific Dosage
# -------------------------------
class DosageRequest(BaseModel):
    drug: str
    age: int

@app.post("/dosage_recommendation")
def dosage_recommendation(req: DosageRequest):
    try:
        drug = req.drug.lower()
        age = req.age
        if drug in drug_db:
            category = "adult" if age >= 18 else "child"
            dose = drug_db[drug]["max_dose_mg"][category]
            return {"drug": drug, "recommended_max_dose_mg": dose}
        return {"error": "Drug not found"}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# 3. Alternative Medication
# -------------------------------
class AltRequest(BaseModel):
    drug: str

@app.post("/alternative_suggestions")
def alternative_suggestions(req: AltRequest):
    try:
        drug = req.drug.lower()
        if drug in drug_db:
            return {"drug": drug, "alternatives": drug_db[drug]["alternatives"]}
        return {"error": "No alternatives found"}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# 4. NLP-Based Extraction
# -------------------------------
nlp = pipeline("ner", model="dslim/bert-base-NER")

class TextInput(BaseModel):
    text: str

@app.post("/extract_drug_info")
def extract_drug_info(data: TextInput):
    try:
        entities = nlp(data.text)
        extracted = [{"word": e["word"], "label": e["entity"]} for e in entities]
        return {"extracted_info": extracted}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# 5. OCR with EasyOCR
# -------------------------------
reader = easyocr.Reader(['en'])

@app.post("/ocr_extract")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = reader.readtext(io.BytesIO(content), detail=0)
        return {"ocr_text": " ".join(text) if text else "No text detected"}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Root health check
# -------------------------------
from fastapi import FastAPI, File, UploadFile
import easyocr
import io

app = FastAPI()

@app.post("/ocr_extract")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)

        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image_stream)

        text_output = [res[1] for res in result]

        return {"extracted_text": text_output}

    except Exception as e:
        return {"error": str(e)}
@app.get("/")
def home():
    return {"message": "Backend is running"}
