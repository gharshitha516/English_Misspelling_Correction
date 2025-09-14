import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.markdown("<h1 style='text-align:center;'>✒️ LexCorrect</h1>", unsafe_allow_html=True)

user_input = st.text_area("", height=150, placeholder="Type something here.")

if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.subheader("✅ Corrected Text")
            st.success(corrected_text)
