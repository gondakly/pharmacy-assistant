import streamlit as st
import pandas as pd
import os
import json
import zipfile  # Imported for handling compressed datasets on cloud servers
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Set up page configuration using relative path for the logo
st.set_page_config(page_title="Pharmacy GPT", page_icon="logo.jpg", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .logo-container { text-align: center; margin: 10px 0 20px 0; }
    .disclaimer { color: #D32F2F; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DATASET INITIALIZATION & UTILITIES (ROBUST SCRIPT)
# ----------------------------------------------------
DATASET_PATH = "Disease_and_symptoms_dataset.csv"
ZIP_DATASET_PATH = "Disease_and_symptoms_dataset.zip"

def initialize_dataset():
    """Ensures the dataset file exists, unzipping it if necessary for cloud environments."""
    # Step 1: Check if the CSV already extracted and fully readable
    try:
        if os.path.exists(DATASET_PATH):
            pd.read_csv(DATASET_PATH, nrows=5)  # Quick check read
            return
    except Exception:
        pass

    # Step 2: If CSV is missing or corrupted, check if the ZIP archive exists to extract it
    if os.path.exists(ZIP_DATASET_PATH):
        try:
            with zipfile.ZipFile(ZIP_DATASET_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")  # Extract the CSV directly into the root folder
            return
        except Exception as e:
            st.error(f"Failed to automatically extract dataset zip file: {str(e)}")
            return

    # Step 3: Fallback creation if both files are entirely missing
    try:
        df = pd.DataFrame(columns=["Disease", "Symptoms", "Precautions", "ML_Insight"])
        seed_data = [
            {"Disease": "Viral Respiratory Infection", "Symptoms": "fever, cough, sore throat, flu, cold", "Precautions": "Rest, fluids", "ML_Insight": "High probability of Viral Respiratory Infection"},
            {"Disease": "Allergic Rhinitis", "Symptoms": "sneezing, runny nose, itchy eyes", "Precautions": "Avoid allergens", "ML_Insight": "High probability of Allergy"}
        ]
        df = pd.concat([df, pd.DataFrame(seed_data)], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False, encoding='utf-8')
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")

# Safe automatic initialization/extraction
initialize_dataset()

def search_local_dataset(query_text):
    """Searches the dataset safely and returns the record alongside its row index number."""
    if not query_text:
        return None, None
    try:
        if not os.path.exists(DATASET_PATH):
            return None, None
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')
        query_lower = str(query_text).lower().strip()
        
        for idx, row in df.iterrows():
            if query_lower in str(row.get('Symptoms', '')).lower() or query_lower in str(row.get('Disease', '')).lower():
                return row.to_dict(), idx + 1
    except PermissionError:
        st.error("Cannot read dataset because it is currently locked by another program (like Excel).")
    except Exception:
        st.warning("Note: Dataset reading skipped temporarily due to a layout issue.")
    return None, None

def append_new_record_to_dataset(disease, symptoms, precautions, insight):
    """Saves records safely and returns the newly generated record row number."""
    try:
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')
    except Exception:
        df = pd.DataFrame(columns=["Disease", "Symptoms", "Precautions", "ML_Insight"])
        
    clean_row = {
        "Disease": str(disease).replace("\n", " ").strip(),
        "Symptoms": str(symptoms).replace("\n", " ").strip(),
        "Precautions": str(precautions).replace("\n", " ").strip(),
        "ML_Insight": str(insight).replace("\n", " ").strip()
    }
    
    df = pd.concat([df, pd.DataFrame([clean_row])], ignore_index=True)
    
    try:
        df.to_csv(DATASET_PATH, index=False, encoding='utf-8')
        return len(df)
    except PermissionError:
        st.error("Could not append the new record because the CSV file is open in another window or Excel.")
        return "Not Saved (File Locked)"

# CORE LOGIC & SIDEBAR UI
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    try:
        st.image("logo.jpg", width=280, use_column_width=False)
    except:
        st.markdown('<div class="logo-container"><h1>Pharmacy GPT</h1></div>', unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Pharmacy GPT</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Medicine Recommendation System**")
st.markdown('<p class="disclaimer"><strong>WARNING: EDUCATIONAL PROTOTYPE ONLY</strong> — Not a substitute for professional medical advice. Always consult a doctor.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    user_symptoms = st.text_area(
        "Describe Symptoms", 
        placeholder="Enter symptoms here (e.g., persistent dry cough, mild fever for two days, headache)..."
    )
    
    additional_notes = st.text_area(
        "Additional Notes / History",
        placeholder="Allergies, chronic conditions, current medications..."
    )
    st.markdown("---")
    st.caption("Powered by Ollama (via Cloud Tunneling)")
    
    if st.button("Refresh Connection"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def get_llm():
    try:
        llm_instance = ChatOllama(
            model="llama3.2:3b", 
            temperature=0.3, 
            num_ctx=4096,
            base_url="https://unmade-sprinkled-despair.ngrok-free.dev"
        )
        return llm_instance
    except Exception:
        return None

llm = get_llm()

if llm is None:
    st.error("Ollama connection failed. Ensure your ngrok tunnel is live and the model is running.")

prompt_template = ChatPromptTemplate.from_template("""
You are Pharmacy GPT, an expert AI pharmacist assistant specializing in the Egyptian pharmaceutical market.
Patient: {age} years old, {gender}.
Symptoms: {symptoms}.
Notes: {notes}.
ML Prediction Insight: {ml_insight}

Provide a clear, structured response adhering to these sections:
1. Possible Conditions
2. Recommended Over-The-Counter (OTC) Medicines & Egyptian Market Status:
3. Home Care & Lifestyle Advice
4. Red Flags — when to see a doctor immediately
5. Precautions & Contraindications
""")

structuring_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "The user entered symptoms or a condition that wasn't found in our historical dataset. Analyze these symptoms and map them perfectly to fit into the dataset records. Return ONLY a raw JSON object string without code blocks or markdown wrappers."),
    ("user", "Analyze the symptoms: \"{user_symptoms}\"\n\nProduce valid JSON matching this exact structure: \n{{\"Disease\": \"...\", \"Symptoms\": \"...\", \"Precautions\": \"...\", \"ML_Insight\": \"...\"}}")
])

def get_ml_prediction_and_sync(symptoms_text):
    if not symptoms_text:
        return "No symptoms specified for analysis.", "Unknown", "N/A"
    
    matched_record, record_num = search_local_dataset(symptoms_text)
    
    if matched_record:
        st.info(f"Record match found directly in Dataset! [Record Row Number: #{record_num}]")
        return matched_record["ML_Insight"], matched_record["Disease"], record_num
    
    if llm is not None:
        st.warning(" Not found in current dataset. Querying Ollama to synthesize a new record entry...")
        try:
            structuring_chain = structuring_prompt_template | llm
            raw_json = structuring_chain.invoke({"user_symptoms": symptoms_text}).content.strip()
            
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]
            
            parsed_record = json.loads(raw_json.strip())
            
            new_record_num = append_new_record_to_dataset(
                disease=parsed_record.get("Disease", "Unknown Condition"),
                symptoms=parsed_record.get("Symptoms", symptoms_text),
                precautions=parsed_record.get("Precautions", "Consult Doctor"),
                insight=parsed_record.get("ML_Insight", "Further evaluation required.")
            )
            
            if isinstance(new_record_num, int):
                st.toast(f"✅ Dynamically saved entry into dataset as Record Entry #{new_record_num}!")
            return parsed_record.get("ML_Insight", "Analysis ready"), parsed_record.get("Disease", "Unknown"), new_record_num
            
        except Exception:
            pass

    normalized_text = symptoms_text.lower()
    trigger_words = ["fever", "cough", "sore throat", "flu", "cold"]
    if any(word in normalized_text for word in trigger_words):
        return "High probability of Viral Respiratory Infection (Cold/Flu)", "Viral Respiratory Infection", "Dynamic"
    return "Symptoms suggest possible mild infection or allergy.", "Undetermined Condition", "Dynamic"

# PROCESS ACTION
if st.button("Analyze & Get Recommendations", type="primary", use_container_width=True):
    if llm is None:
        st.error("Cannot proceed. Ollama server connection is offline.")
    elif not user_symptoms.strip():
        st.warning("Please provide a description of the symptoms before analyzing.")
    else:
        with st.spinner("Processing Dataset Pipelines & LLM Synthesis..."):
            ml_insight, inferred_disease, dataset_record_id = get_ml_prediction_and_sync(user_symptoms)
            st.session_state.dataset_record_id = dataset_record_id
            
            try:
                chain = prompt_template | llm
                response = chain.invoke({
                    "age": age,
                    "gender": gender,
                    "symptoms": user_symptoms,
                    "notes": additional_notes or "None provided.",
                    "ml_insight": ml_insight
                })
                
                st.session_state.analysis = response.content
                st.session_state.ml_insight = ml_insight
            except Exception as e:
                st.error(f"Execution error: {str(e)}")

if "analysis" in st.session_state:
    st.subheader("Pharmacy GPT Analysis & Recommendations")
    st.markdown(st.session_state.analysis)
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("Dataset / ML Model Insight")
        st.write(st.session_state.ml_insight)
    with c2:
        st.info("Dataset Record Index Tracked")
        st.metric(label="Record Entry Row ID", value=f"#{st.session_state.get('dataset_record_id', 'N/A')}")
    with c3:
        st.error("Medical Disclaimer")
        st.write("This AI tool is for learning and simulation only. Do not self-medicate based on this output.")

# CHAT SYSTEM INTERACTION
st.markdown("---")
st.subheader("Follow-up Chat with Pharmacy GPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask more questions..."):
    if llm is None:
        st.error("Ollama is not running.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting Pharmacy GPT..."):
                try:
                    chat_resp = llm.invoke(
                        f"Context from previous diagnosis: {st.session_state.get('analysis', '')}\n"
                        f"User question: {prompt}\n"
                        f"Respond helpfully as Pharmacy GPT focusing on the Egyptian pharmaceutical landscape, localized pricing, and available equivalents."
                    )
                    st.markdown(chat_resp.content)
                    st.session_state.messages.append({"role": "assistant", "content": chat_resp.content})
                except Exception as e:
                    st.error(f"Error: {str(e)}")