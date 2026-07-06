import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Pharmacy GPT", page_icon=None, layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .logo-container { text-align: center; margin: 10px 0 20px 0; }
    .disclaimer { color: #D32F2F; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

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
    
    # زرار لإعادة المحاولة لو السيرفر كان مقفول وفتحته
    if st.button("🔄 Refresh Connection"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def get_llm():
    try:
        # حددنا الـ base_url صراحة بـ الـ IP المحلي لتفادي مشاكل الـ Windows Localhost
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
    st.error("Ollama connection failed. Ensure the Ollama server is running and the llama3.2:3b model has been pulled via the command line interface.")

# UPDATED: Prompt template explicitly localized for the Egyptian pharmaceutical market
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

def get_ml_prediction(symptoms_text):
    if not symptoms_text:
        return "No symptoms specified for analysis."
    
    normalized_text = symptoms_text.lower()
    trigger_words = ["fever", "cough", "sore throat", "flu", "cold"]
    if any(word in normalized_text for word in trigger_words):
        return "High probability of Viral Respiratory Infection (Cold/Flu)"
    return "Symptoms suggest possible mild infection or allergy. Further clinical evaluation recommended."

if st.button("Analyze & Get Recommendations", type="primary", use_container_width=True):
    if llm is None:
        st.error("Cannot proceed. Ollama server is offline. Try checking your terminal or clicking 'Refresh Connection'.")
    elif not user_symptoms.strip():
        st.warning("Please provide a description of the symptoms before analyzing.")
    else:
        with st.spinner("Analyzing with ML + LLM..."):
            ml_insight = get_ml_prediction(user_symptoms)
            
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
                st.error(f"Execution error: {str(e)}\n\nMake sure you ran 'ollama run llama3.2:3b' in your terminal first.")

if "analysis" in st.session_state:
    st.subheader("Pharmacy GPT Analysis & Recommendations")
    st.markdown(st.session_state.analysis)
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.success("ML Model Insight")
        st.write(st.session_state.ml_insight)
    with c2:
        st.error("Medical Disclaimer")
        st.write("This AI tool is for learning and simulation only. Do not self-medicate based on this output.")

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
