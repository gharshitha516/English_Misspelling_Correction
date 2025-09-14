import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
import string

# Helpers
def tokenize_with_punct(text: str):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def is_punct(tok: str):
    return re.fullmatch(r"[^\w\s]", tok) is not None

def detokenize(tokens):
    out = ""
    for tok in tokens:
        if out == "":
            out = tok
        elif is_punct(tok):
            out += tok
        elif out[-1] in "([{\"'":
            out += tok
        else:
            out += " " + tok
    return out.strip()

def filter_suggestion(word: str) -> bool:
    """Remove punctuation, numbers, multi-word junk."""
    word = word.strip()
    if not word:
        return False
    if all(ch in string.punctuation for ch in word):
        return False
    if any(ch.isdigit() for ch in word):
        return False
    if " " in word:
        return False
    return True

# Load correction model
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return mdl, tok

model, tokenizer = load_model()

# Load masker
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()
mask_token = getattr(masker.tokenizer, "mask_token", "[MASK]")

# App
st.markdown("<h1 style='text-align:center;'>✒️ LexCorrect </h1>", unsafe_allow_html=True)

if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = None

user_input = st.text_area("", height=150, placeholder="Type something here.")

if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
            st.session_state.corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# If we already have corrected text
if st.session_state.corrected_text:
    corrected_text = st.session_state.corrected_text

    st.subheader("✅ Corrected Sentence")
    st.success(corrected_text)

    corr_toks = tokenize_with_punct(corrected_text)
    final_toks = corr_toks.copy()

    st.subheader("🔄 Word Suggestions (Optional)")

    for i, word in enumerate(corr_toks):
        if is_punct(word):
            continue

        # Build masked sentence
        masked = corr_toks.copy()
        masked[i] = mask_token
        masked_sentence = detokenize(masked)

        candidates = masker(masked_sentence)[:10]

        valid = []
        for cand in candidates:
            token_str = cand.get("token_str", "").strip(" '\"")
            score = cand.get("score", 0)
            if score < 0.05:
                continue
            if filter_suggestion(token_str):
                valid.append(token_str)

        # dedupe
        seen = set()
        valid = [x for x in valid if not (x in seen or seen.add(x))]

        if valid:
            options = [word] + valid[:3]  # limit to top 3 sensible suggestions
            choice = st.selectbox(
                f"Replace '{word}':",
                options=options,
                index=0,
                key=f"choice_{i}"
            )
            final_toks[i] = choice

    final_sentence = detokenize(final_toks)
    st.subheader("🎯 Final Choice")
    st.success(final_sentence)

