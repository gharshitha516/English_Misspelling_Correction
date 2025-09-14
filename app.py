import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load correction model
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe1_Misspelling_Correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Load BERT fill-mask pipeline for word suggestions
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()

st.markdown("<h1 style='text-align:center;'>✒️ SpellFixer Pro</h1>", unsafe_allow_html=True)

user_input = st.text_area("Enter your sentence:", height=150, placeholder="Type with mistakes...")

if st.button("✨ Correct My Text"):
    if user_input.strip():
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Corrected Sentence")
        st.success(corrected_text)

        # Compare word by word
        orig_words = user_input.split()
        corr_words = corrected_text.split()
        final_words = []

        st.subheader("🔄 Word Suggestions")
        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:
                # Mask the corrected word in the sentence
                masked_sentence = corrected_text.split()
                masked_sentence[i] = "[MASK]"
                masked_sentence = " ".join(masked_sentence)

                # Get top predictions from BERT
                suggestions = masker(masked_sentence)[:3]
                options = [corr] + [s['token_str'] for s in suggestions]

                choice = st.selectbox(f"Replace '{corr}' (was '{orig}'):", options=options, index=0)
                final_words.append(choice)
            else:
                final_words.append(corr)

        # Build final sentence
        final_sentence = " ".join(final_words)
        st.subheader("🎯 Final Choice")
        st.success(final_sentence)
