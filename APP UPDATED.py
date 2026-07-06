import streamlit as st
import pandas as pd
import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Pharmacy GPT", page_icon="D:/pharmacy Assistant/logo.jpg", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .logo-container { text-align: center; margin: 10px 0 20px 0; }
    .disclaimer { color: #D32F2F; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# DATASET INITIALIZATION & UTILITIES (ROBUST VERSION)
# Using raw string (r"") to handle Windows backslashes flawlessly
DATASET_PATH = r"D:\pharmacy Assistant\Disease and symptoms dataset 2023\Disease and symptoms dataset.csv"

def initialize_dataset():
    """Ensures a dataset file exists based on the Mendeley 2023 schema structure."""
    try:
        # Check if file exists and is readable. If it's corrupted, we handle it.
        if os.path.exists(DATASET_PATH):
            pd.read_csv(DATASET_PATH)
            return
    except Exception:
        # If the file is corrupted and can't be read, remove it to start fresh
        try:
            os.remove(DATASET_PATH)
        except:
            pass

    # Ensure directories exist
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # Build fresh default dataset structure
    df = pd.DataFrame(columns=["Disease", "Symptoms", "Precautions", "ML_Insight"])
    seed_data = [
        {"Disease": "Viral Respiratory Infection", "Symptoms": "fever, cough, sore throat, flu, cold", "Precautions": "Rest, fluids", "ML_Insight": "High probability of Viral Respiratory Infection"},
        {"Disease": "Allergic Rhinitis", "Symptoms": "sneezing, runny nose, itchy eyes", "Precautions": "Avoid allergens", "ML_Insight": "High probability of Allergy"}
    ]
    df = pd.concat([df, pd.DataFrame(seed_data)], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False, encoding='utf-8')

# Run the safe initialization
initialize_dataset()

def search_local_dataset(query_text):
    """Searches the dataset safely. Prevents parser crashes if data is malformed."""
    if not query_text:
        return None
    try:
        # on_bad_lines='skip' ensures pandas drops broken rows instead of crashing
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')
        query_lower = str(query_text).lower().strip()
        
        for idx, row in df.iterrows():
            if query_lower in str(row.get('Symptoms', '')).lower() or query_lower in str(row.get('Disease', '')).lower():
                return row.to_dict()
    except Exception as e:
        st.warning(f"Note: Dataset reading skipped temporarily due to a layout issue.")
    return None

def append_new_record_to_dataset(disease, symptoms, precautions, insight):
    """Saves records safely, cleaning up raw newlines that disrupt CSV tokens."""
    try:
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')
    except Exception:
        df = pd.DataFrame(columns=["Disease", "Symptoms", "Precautions", "ML_Insight"])
        
    # Clean text inputs to remove unescaped newlines/tabs that break token patterns
    clean_row = {
        "Disease": str(disease).replace("\n", " ").strip(),
        "Symptoms": str(symptoms).replace("\n", " ").strip(),
        "Precautions": str(precautions).replace("\n", " ").strip(),
        "ML_Insight": str(insight).replace("\n", " ").strip()
    }
    
    df = pd.concat([df, pd.DataFrame([clean_row])], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False, encoding='utf-8')

# CORE LOGIC & SIDEBAR UI
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    try:
        st.image("D:/Pharmacy Gpt/logo.jpg", width=280, use_column_width=False)
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
    st.caption("Powered by Ollama and ML Pipelines")
    
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
            base_url="http://127.0.0.1:11434"
        )
        return llm_instance
    except Exception as e:
        return None

llm = get_llm()

if llm is None:
    st.error("Ollama connection failed. Ensure the Ollama server is running and the llama3.2:3b model has been pulled.")

# Prompt template localized for the Egyptian pharmaceutical market
prompt_template = ChatPromptTemplate.from_template("""
You are Pharmacy GPT, an expert AI pharmacist assistant specializing in the Egyptian pharmaceutical market.
Patient: {age} years old, {gender}.
Symptoms: {symptoms}.
Notes: {notes}.
ML Prediction Insight: {ml_insight}

Provide a clear, structured response adhering to these sections:
1. Possible Conditions

2. Recommended Over-The-Counter (OTC) Medicines & Egyptian Market Status:
For each medicine you recommend, you MUST provide:
- The generic/scientific name.
- Egyptian Market Availability Status: State explicitly if it is commonly available in Egypt (e.g., "Available", "Currently facing shortages", or "Imported/Rare").
- Egyptian Brand Names: Mention common local commercial names found in Egyptian pharmacies (e.g., Panadol, Adol, Congestal, Alphintern, Cetal, etc.).
- Estimated Price: Give a rough estimation of the price in Egyptian Pounds (EGP).
- Egyptian Alternatives: If a primary choice is imported, expensive, or experiencing shortages, explicitly list 1-2 reliable local alternatives/substitutes (المثائل والبدائل) manufactured by local Egyptian companies that are readily available in the local market along with their approximate prices.
- General dosage guidance.

3. Home Care & Lifestyle Advice
4. Red Flags — when to see a doctor immediately
5. Precautions & Contraindications

Always emphasize professional medical consultation and explicitly state that prices are approximate estimates.
""")

# Structured parser to extract new records matching dataset columns
structuring_prompt_template = ChatPromptTemplate.from_template("""
The user entered symptoms or a condition that wasn't found in our historical Stark & Bran (2025) dataset: "{user_symptoms}".
Analyze these symptoms and map them perfectly to fit into the dataset records.

Return ONLY a valid JSON object matching this structure without markdown formatting or code blocks:
{{
  "Disease": "Name of the inferred medical condition",
  "Symptoms": "Comma-separated list of core clinical symptoms identified",
  "Precautions": "Primary lifestyle action or safety step",
  "ML_Insight": "A clear 'High probability of...' text prediction snippet"
}}
""")

def get_ml_prediction_and_sync(symptoms_text):
    if not symptoms_text:
        return "No symptoms specified for analysis.", "Unknown"
    
    # 1. Search locally first
    matched_record = search_local_dataset(symptoms_text)
    
    if matched_record:
        st.info("💡 Record match found directly in Stark & Bran Dataset.")
        return matched_record["ML_Insight"], matched_record["Disease"]
    
    # 2. Fallback: Query Ollama to construct dataset record details if it's missing
    if llm is not None:
        st.warning("🔍 Not found in current dataset. Querying Ollama to create a structured dataset record...")
        try:
            structuring_chain = structuring_prompt_template | llm
            raw_json = structuring_chain.invoke({"user_symptoms": symptoms_text}).content.strip()
            
            # Clean possible markdown block wrappers from Ollama output
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]
            
            parsed_record = json.loads(raw_json)
            
            # 3. Save new record into the dataset dynamically
            append_new_record_to_dataset(
                disease=parsed_record.get("Disease", "Unknown Condition"),
                symptoms=parsed_record.get("Symptoms", symptoms_text),
                precautions=parsed_record.get("Precautions", "Consult Doctor"),
                insight=parsed_record.get("ML_Insight", "Further evaluation required.")
            )
            st.toast("✅ Added new structured record to local dataset file!")
            return parsed_record.get("ML_Insight"), parsed_record.get("Disease")
            
        except Exception as e:
            pass # Fail gracefully back to basic matching if JSON generation fails

    # 4. Standard hardcoded fallback rule if everything else fails
    normalized_text = symptoms_text.lower()
    trigger_words = ["fever", "cough", "sore throat", "flu", "cold"]
    if any(word in normalized_text for word in trigger_words):
        return "High probability of Viral Respiratory Infection (Cold/Flu)", "Viral Respiratory Infection"
    return "Symptoms suggest possible mild infection or allergy. Further clinical evaluation recommended.", "Undetermined Condition"


# PROCESS ACTION
if st.button("Analyze & Get Recommendations", type="primary", use_container_width=True):
    if llm is None:
        st.error("Cannot proceed. Ollama server is offline.")
    elif not user_symptoms.strip():
        st.warning("Please provide a description of the symptoms before analyzing.")
    else:
        with st.spinner("Processing Dataset Pipelines & LLM Synthesis..."):
            ml_insight, inferred_disease = get_ml_prediction_and_sync(user_symptoms)
            
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
    c1, c2 = st.columns(2)
    with c1:
        st.success("Dataset / ML Model Insight")
        st.write(st.session_state.ml_insight)
    with c2:
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