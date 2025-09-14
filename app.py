import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import string

# -------------------------------
# Utility: filter out unwanted tokens
# -------------------------------
def filter_word(word: str) -> bool:
    word = word.strip()
    if not word:
        return False
    if word in string.punctuation:
        return False
    if any(ch.isdigit() for ch in word):
        return False
    return True

# -------------------------------
# Load correction model
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# -------------------------------
# Load BERT fill-mask pipeline for suggestions
# -------------------------------
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()

# -------------------------------
# App Frontend
# -------------------------------
st.markdown("<h1 style='text-align:center;'>✒️ SpellFixer Pro</h1>", unsafe_allow_html=True)

user_input = st.text_area("Enter your sentence:", height=150, placeholder="Type with mistakes...")

if st.button("✨ Correct My Text"):
    if user_input.strip():
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -------------------------------
        # Step 1: Show corrected sentence
        # -------------------------------
        st.subheader("✅ Corrected Sentence")
        st.success(corrected_text)

        # -------------------------------
        # Step 2: Word-level suggestions
        # -------------------------------
        orig_words = user_input.split()
        corr_words = corrected_text.split()

        # Start with corrected sentence as default
        final_words = corr_words.copy()

        st.subheader("🔄 Word Suggestions (Optional)")
        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:  # Only suggest replacements where model corrected
                # Mask corrected word
                masked_sentence = corr_words.copy()
                masked_sentence[i] = "[MASK]"
                masked_sentence = " ".join(masked_sentence)

                # Get top predictions
                suggestions = masker(masked_sentence)[:10]
                valid_options = [
                    s['token_str'] for s in suggestions
                    if filter_word(s['token_str']) and s['token_str'].lower() != corr.lower()
                ]

                if valid_options:
                    options = [corr] + valid_options
                    options = list(dict.fromkeys(options))  # remove duplicates
                    choice = st.selectbox(
                        f"Replace '{orig}' → '{corr}':",
                        options=options,
                        index=0,
                        key=f"choice_{i}"
                    )

                    # ✅ Overwrite corrected word with choice
                    final_words[i] = choice

        # -------------------------------
        # Step 3: Final Sentence
        # -------------------------------
        final_sentence = " ".join(final_words)
        st.subheader("🎯 Final Choice")
        st.success(final_sentence)

