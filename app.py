import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model + tokenizer from Hugging Face Hub
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center; 
            color: #2c3e50; 
            font-size: 42px; 
            font-weight: bold;
        }
        .subtitle {
            text-align: center; 
            font-size: 18px; 
            color: #7f8c8d; 
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 2px solid #dcdde1;
            padding: 12px;
            font-size: 16px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 25px;
            padding: 0.6em 1.5em;
            font-size: 18px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='main-title'>✒️ SpellFixer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fix spelling mistakes instantly with AI ✨</p>", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("#### 📝 Enter your text")
user_input = st.text_area("", height=150, placeholder="Type something with spelling mistakes...")

# Example buttons
examples = [
    "I relly likee this aap",
    "Ths is an exmple with erors",
    "She go to scholl evry day"
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex):
        user_input = ex

# --- Generate Button ---
if st.button("✨ Correct My Text"):
    if user_input.strip():
        with st.spinner("🔍 Checking your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- Output Card ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ✅ Corrected Text")
        st.success(corrected_text)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to correct")

# --- Footer ---
st.markdown("<hr><p style='text-align:center; color:#95a5a6;'></p>", unsafe_allow_html=True)
