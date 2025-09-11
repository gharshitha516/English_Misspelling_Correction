import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model + tokenizer from Hugging Face Hub
@st.cache_resource
def load_model():
    model_name = "harshhitha/3_Misspelling_Correction"  # ✅ corrected
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# App Header
st.markdown(
    """
    <h1 style='text-align: center; color: White;'>✒️ SpellFixer</h1>
    <p style='text-align: center; font-size:18px; color: #555;'>Fix spelling mistakes and typos instantly. Just enter your text to get accurate corrections.</p>
    """,
    unsafe_allow_html=True
)

# Input Section
user_input = st.text_area(
    "",  
    height=150, 
    placeholder="Start typing here."
)

# Generate Button 
if st.button("✨ Correct My Text"):
    if user_input.strip():
        with st.spinner("Checking your text… 🧐"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Output Section 
        st.markdown("### ✅ Corrected Text")
        st.success(corrected_text)
    else:
        st.warning("⚠️ Please enter some text to correct")

# Footer 
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #888; font-size:12px;'>
    </p>
    """,
    unsafe_allow_html=True
)
