import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import string
import re

# 1. Load the grammar correction model
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# 2. Load BERT fill-mask pipeline for alternative word suggestions
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()

# 3. App Header
st.markdown("<h1 style='text-align:center;'>✒️ SpellFixer Pro</h1>", unsafe_allow_html=True)

# 4. User Input
user_input = st.text_area("Enter your sentence:", height=150, placeholder="Type with mistakes...")

# Helper: Clean word suggestions (remove punctuation, numbers, weird tokens)
def filter_word(token_str):
    token_str = token_str.strip()
    if not token_str:
        return False
    if token_str in string.punctuation:
        return False
    if re.match(r'^[0-9]+$', token_str):
        return False
    if re.match(r'^[^\w]+$', token_str):
        return False
    return True

# 5. Main Button
if st.button("✨ Correct My Text"):
    if user_input.strip():
        # 1: Correct the input sentence 
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Corrected Sentence")
        st.success(corrected_text)

        # 2: Prepare dropdowns only where useful 
        orig_words = user_input.split()
        corr_words = corrected_text.split()

        # Start with corrected words as default final choice
        final_words = corr_words.copy()

        st.subheader("🔄 Word Suggestions (Optional)")
        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:  # only check changed words
                # Mask the corrected word in the sentence
                masked_sentence = corr_words.copy()
                masked_sentence[i] = "[MASK]"
                masked_sentence = " ".join(masked_sentence)

                # Get top predictions from BERT
                suggestions = masker(masked_sentence)[:10]
                valid_options = [
                    s['token_str'] for s in suggestions 
                    if filter_word(s['token_str']) and s['token_str'].lower() != corr.lower()
                ]

                # Only create dropdown if there are real alternatives
                if valid_options:
                    options = [corr] + valid_options
                    options = list(dict.fromkeys(options))  # remove duplicates
                    choice = st.selectbox(
                        f"Replace '{orig}' → '{corr}':",
                        options=options,
                        index=0,
                        key=f"choice_{i}"
                    )
                    final_words[i] = choice  # ✅ update only if user chooses

        # 3: Build final sentence 
        final_sentence = " ".join(final_words)
        st.subheader("🎯 Final Choice")
        st.success(final_sentence)
