import streamlit as st
import pandas as pd
import os
import json
import zipfile
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# PRODUCTION CODE ARCHITECTURE & SYSTEM CONFIGURATION
# ENHANCEMENT: Using strict relative execution pathing to prevent Linux deployment crashes.
# The absolute windows drive paths ('D:/pharmacy Assistant/...') are completely decoupled.
st.set_page_config(
    page_title="Pharmacy GPT - Enterprise Suite", 
    page_icon="logo.jpg", 
    layout="wide"
)

# Architectural CSS injection supporting UX telemetry visual assets
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; font-weight: 700; }
    .disclaimer { color: #D32F2F; font-size: 0.9rem; font-weight: bold; background-color: #FFEBEE; padding: 10px; border-radius: 5px; }
    .ad-banner { background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .metric-card { background-color: #F8F9FA; border: 1px solid #E0E0E0; padding: 15px; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TELEMETRY & DATA METRICS INITIALIZATION (STATE ENGINE)
# -----------------------------------------------------------------------------
# INITIALIZATION: Instantiating tracking telemetry inside Streamlit's global session state.
if "total_analyses" not in st.session_state:
    st.session_state.total_analyses = 0
if "affiliate_clicks" not in st.session_state:
    st.session_state.affiliate_clicks = 0
if "estimated_revenue" not in st.session_state:
    st.session_state.estimated_revenue = 0.0
if "execution_times" not in st.session_state:
    st.session_state.execution_times = []

# -----------------------------------------------------------------------------
# DATASET LIFECYCLE MANAGEMENT (COMPRESSION & MEMORY PIPELINE)
# -----------------------------------------------------------------------------
DATASET_PATH = "Disease_and_symptoms_dataset.csv"
ZIP_DATASET_PATH = "Disease_and_symptoms_dataset.zip"

def initialize_dataset():
    """
    Manages cold-start initialization of compressed medical assets on the hosting runtime.
    Unzips source vectors dynamically into the localized sandbox workspace.
    """
    try:
        if os.path.exists(DATASET_PATH):
            pd.read_csv(DATASET_PATH, nrows=5)
            return
    except Exception:
        pass

    if os.path.exists(ZIP_DATASET_PATH):
        try:
            with zipfile.ZipFile(ZIP_DATASET_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            return
        except Exception as e:
            st.error(f"Critical Workspace Exception (Decompression Failure): {str(e)}")
            return

    # Fallback structure generation if clean deploy state contains no vectors
    try:
        df = pd.DataFrame(columns=["Disease", "Symptoms", "Precautions", "ML_Insight"])
        seed_data = [
            {"Disease": "Viral Respiratory Infection", "Symptoms": "fever, cough, sore throat, flu, cold", "Precautions": "Rest, fluids", "ML_Insight": "High probability of Viral Respiratory Infection"},
            {"Disease": "Allergic Rhinitis", "Symptoms": "sneezing, runny nose, itchy eyes", "Precautions": "Avoid allergens", "ML_Insight": "High probability of Allergy"}
        ]
        df = pd.concat([df, pd.DataFrame(seed_data)], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False, encoding='utf-8')
    except Exception as e:
        st.error(f"Pipeline Initialization Core Crash: {str(e)}")

# Trigger automated dynamic asset extraction
initialize_dataset()

def search_local_dataset(query_text):
   # Performs high-speed safe evaluation scan over the extracted pandas DataFrame structures.
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
        st.error("I/O File Lock Error: File resource currently captured by external OS process (e.g. Excel).")
    except Exception:
        pass
    return None, None

def append_new_record_to_dataset(disease, symptoms, precautions, insight):
    #Appends out-of-vocabulary synthesised inferences back to the base historical tracking tables.
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
    except Exception:
        return "Write Lock / IO Volatile"

# GRAPHICAL UI LAYER & PROFILE INPUT MATRICES
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    try:
        st.image("logo.jpg", width=280)
    except Exception:
        st.markdown('<div class="logo-container"><h1>Pharmacy GPT</h1></div>', unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Pharmacy GPT - Enterprise</h1>', unsafe_allow_html=True)
st.markdown('<p class="disclaimer">CRITICAL DEMONSTRATION WARNING: This is an educational simulator system. Not a physical medical practice proxy.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Patient Vitals Profiler")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Biological Gender", ["Male", "Female", "Other"])
    user_symptoms = st.text_area("Symptomatic Diagnostics Query", placeholder="Enter symptoms (e.g., severe migraine, dry cough, high fever)...")
    additional_notes = st.text_area("Clinical History / Manifestation Notes", placeholder="Allergies, chronic hypertension, current medications...")
    
    st.markdown("---")
    if st.button("Force Invalidate Memory Caches"):
        st.cache_resource.clear()
        st.rerun()

# LLM INFERENCE GATEWAY (LANGCHAIN OLLAMA DRIVER)
@st.cache_resource
def get_llm():
    try:
        return ChatOllama(
            model="llama3.2:3b", 
            temperature=0.3, 
            num_ctx=4096,
            base_url="https://unmade-sprinkled-despair.ngrok-free.dev"
        )
    except Exception:
        return None

llm = get_llm()
if llm is None:
    st.error("Infrastructure Offline: Tunnel connection or local Ollama engine returned a broken connection array.")

prompt_template = ChatPromptTemplate.from_template("""
You are Pharmacy GPT, an expert AI pharmacist assistant specializing in the Egyptian pharmaceutical market.
Contextualize recommendations matching local economic structures and actual Egyptian trade names.

Patient Profile: {age} years old, {gender}.
Symptoms Entered: {symptoms}.
Medical Notes: {notes}.
Dataset Reference Mapping: {dataset_context}
ML Prediction Insight: {ml_insight}

Provide a clear, highly structured response adhering to these exact sections:
1. Possible Conditions (Based on symptoms and dataset analysis)
2. Recommended Over-The-Counter (OTC) Medicines & Egyptian Market Status:
   - Provide the specific commercial medicine name(s) relevant to the symptoms and matched dataset info.
   - For EACH medicine, state the estimated localized price in Egyptian Pounds (EGP).
   - Explicitly state the Market Status: Whether it is currently "Available" or facing shortages/"Not Available" in Egyptian pharmacies. If a shortage exists, name a locally produced equivalent/substitute.
3. Home Care & Lifestyle Advice
4. Red Flags — when to see a doctor immediately
5. Precautions & Contraindications
""")

structuring_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Maps novel diagnostic strings onto structural JSON profiles. Return ONLY raw valid JSON matching schema bounds. Zero formatting strings outside structural arrays."),
    ("user", "Analyze the symptoms: \"{user_symptoms}\"\n\nProduce valid JSON matching this exact structure: \n{{\"Disease\": \"...\", \"Symptoms\": \"...\", \"Precautions\": \"...\", \"ML_Insight\": \"...\"}}")
])

def get_ml_prediction_and_sync(symptoms_text):
    if not symptoms_text:
        return "No diagnostic data.", "Unknown", "N/A", {}
    
    matched_record, record_num = search_local_dataset(symptoms_text)
    if matched_record:
        st.info(f"In-Memory Cache Lookup Hit! Vector Row ID extracted: #{record_num}")
        return matched_record["ML_Insight"], matched_record["Disease"], record_num, matched_record
    
    if llm is not None:
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
                precautions=parsed_record.get("Precautions", "Clinic Check Required"),
                insight=parsed_record.get("ML_Insight", "Evaluated via Synthetic Real-time Vectorization")
            )
            return parsed_record.get("ML_Insight", "Completed"), parsed_record.get("Disease", "Unknown"), new_record_num, parsed_record
        except Exception:
            pass

    return "Dynamic fallback assessment activated.", "Undetermined Path", "Dynamic State", {}

# -----------------------------------------------------------------------------
# CORE APP RUNTIME EXECUTION & TELEMETRY RECORDING
# -----------------------------------------------------------------------------
if st.button("Run Pipeline Diagnostics and Treatment Architecture", type="primary", use_container_width=True):
    if llm is None:
        st.error("Pipeline blocked: LLM backend interface is currently unavailable.")
    elif not user_symptoms.strip():
        st.warning("Diagnostics aborted: Input buffer for user symptoms cannot be evaluated empty.")
    else:
        # Performance Logging Initialization
        start_time = time.time()
        
        with st.spinner("Executing Dataset Vector Mappings and LLM Hyperparameter Inferences..."):
            ml_insight, inferred_disease, dataset_record_id, absolute_record = get_ml_prediction_and_sync(user_symptoms)
            st.session_state.dataset_record_id = dataset_record_id
            
            try:
                dataset_context_str = json.dumps(absolute_record, ensure_ascii=False)
                chain = prompt_template | llm
                response = chain.invoke({
                    "age": age,
                    "gender": gender,
                    "symptoms": user_symptoms,
                    "notes": additional_notes or "None logged.",
                    "ml_insight": ml_insight,
                    "dataset_context": dataset_context_str
                })
                
                # Commit operational telemetry records to application state
                st.session_state.analysis = response.content
                st.session_state.ml_insight = ml_insight
                st.session_state.total_analyses += 1
                
                # Track compute runtime metrics
                duration = time.time() - start_time
                st.session_state.execution_times.append(duration)
                
            except Exception as e:
                st.error(f"Critical Runtime Exception encountered: {str(e)}")

# PRESENTATION LAYER & MONETIZATION ACTION ARCHITECTURE
if "analysis" in st.session_state:
    # MONETIZATION PIPELINE: Context-aware programmatic banner injection
    symptom_kw = user_symptoms.lower()
    if any(k in symptom_kw for k in ["cough", "fever", "cold", "flu", "sore throat"]):
        st.markdown("""
        <div class="ad-banner">
            <strong>SPONSORED EGYPTIAN PHARMA ALERT:</strong> Facing common cold or low immunity markers? 
            Order authentic multivitamins or relief therapeutics directly via our verified affiliates. 
            Use promo code <strong>PGPT2026</strong> for 12% instant cashback on <strong>Yodawy / Chefaa</strong> apps!
        </div>
        """, unsafe_allow_html=True)

    st.subheader("System Diagnostic Output")
    st.markdown(st.session_state.analysis)
    
    # MONETIZATION PIPELINE: Strategic Affiliate Lead Conversion Triggers
    st.markdown("### Affiliate Fulfillment and Real-time Local Supply Query")
    st.write("Convert this recommendation into a physical order. Selecting a partner below tracks commission parameters:")
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        # Dynamic telemetry hook simulation for Affiliate A
        if st.button("Order via Yodawy Gateway", use_container_width=True, type="secondary"):
            st.session_state.affiliate_clicks += 1
            # Assuming an average ticket size of 150 EGP per OTC basket with a 4% standard referral commission
            st.session_state.estimated_revenue += (150.0 * 0.04)
            st.toast("Redirecting to Yodawy API Context Link... (Affiliate Tracking Active ID: #YOD-01)")
            
    with btn_col2:
        # Dynamic telemetry hook simulation for Affiliate B
        if st.button("Check Stocks and Dispatch via El-Ezaby Hub", use_container_width=True, type="secondary"):
            st.session_state.affiliate_clicks += 1
            st.session_state.estimated_revenue += (150.0 * 0.03) # 3% commission tier
            st.toast("Opening secure tunnel payload to El-Ezaby digital checkout inventory...")
            
    with btn_col3:
        if st.button("Order via Chefaa Digital Health Network", use_container_width=True, type="secondary"):
            st.session_state.affiliate_clicks += 1
            st.session_state.estimated_revenue += (120.0 * 0.05) # 5% commission tier
            st.toast("Connecting pipeline parameters to Chefaa cart optimization platform...")

    st.divider()

    # ADVANCED BI SYSTEM & REAL-TIME PERFORMANCE METRICS (DASHBOARD)
    st.subheader("Live Monetization Analytics and Core System Telemetry")
    st.write("This metrics panel tracks operational health, user engagement, and conversion revenue yield data directly from volatile RAM registers:")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    # Calculation of statistical operational parameters
    ctr = (st.session_state.affiliate_clicks / st.session_state.total_analyses * 100) if st.session_state.total_analyses > 0 else 0.0
    avg_latency = (sum(st.session_state.execution_times) / len(st.session_state.execution_times)) if st.session_state.execution_times else 0.0
    
    with m_col1:
        st.metric(label="Total App Core Evaluations", value=st.session_state.total_analyses)
    with m_col2:
        st.metric(label="Affiliate Link Interactions", value=st.session_state.affiliate_clicks)
    with m_col3:
        st.metric(label="Click-Through Rate (CTR %)", value=f"{ctr:.1f}%")
    with m_col4:
        st.metric(label="Tracked Affiliate Revenue", value=f"{st.session_state.estimated_revenue:.2f} EGP", delta="Gross Yield")
        
    st.info(f"System Core Latency Telemetry: Mean response compilation duration: `{avg_latency:.2f} seconds` per standard inference pass.")

# -----------------------------------------------------------------------------
# DISCRETE MULTI-TURN COMPANION CHAT INTERACTION
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Active Multi-Turn Consultation Chatroom")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Input secondary queries regarding substitutes, pricing variance or dosages..."):
    if llm is None:
        st.error("Conversational state failure: Active LLM instance engine is unmapped.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing tailored pharmaceutical response..."):
                try:
                    chat_resp = llm.invoke(
                        f"Context from previous diagnosis: {st.session_state.get('analysis', '')}\n"
                        f"User question: {prompt}\n"
                        f"Respond helpfully as Pharmacy GPT focusing on the Egyptian pharmaceutical landscape, localized pricing, and available equivalents."
                    )
                    st.markdown(chat_resp.content)
                    st.session_state.messages.append({"role": "assistant", "content": chat_resp.content})
                except Exception as e:
                    st.error(f"Inference exception during contextual response generation: {str(e)}")
