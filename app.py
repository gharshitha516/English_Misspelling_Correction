import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from difflib import SequenceMatcher
import re
import string

# -------------------------------
# Helpers
# -------------------------------
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

def filter_suggestion(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if all(ch in string.punctuation for ch in s):
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if " " in s:
        return False
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9'’-]*", s) is not None

# -------------------------------
# Load correction model
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return mdl, tok

model, tokenizer = load_model()

# -------------------------------
# Load masker
# -------------------------------
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()
mask_token = getattr(masker.tokenizer, "mask_token", "[MASK]")

# -------------------------------
# App
# -------------------------------
st.markdown("<h1 style='text-align:center;'>✒️ SpellFixer Pro</h1>", unsafe_allow_html=True)

# Session state
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = None

user_input = st.text_area("Enter your sentence:", height=150, placeholder="Type with mistakes...")

if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
            st.session_state.corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# If we already have corrected text, show suggestions
# -------------------------------
if st.session_state.corrected_text:
    corrected_text = st.session_state.corrected_text

    st.subheader("✅ Corrected Sentence")
    st.success(corrected_text)

    orig_toks = tokenize_with_punct(user_input)
    corr_toks = tokenize_with_punct(corrected_text)
    final_toks = corr_toks.copy()

    st.subheader("🔄 Word Suggestions (Optional)")

    sm = SequenceMatcher(None, orig_toks, corr_toks)
    choice_index = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                orig_word = orig_toks[i1]
                corr_word = corr_toks[j1]
                if is_punct(corr_word):
                    continue

                masked = corr_toks.copy()
                masked[j1] = mask_token
                masked_sentence = detokenize(masked)

                candidates = masker(masked_sentence)[:12]
                valid = []
                for c in candidates:
                    tok_str = c.get("token_str", "").strip(" '\"")
                    if tok_str and filter_suggestion(tok_str) and tok_str.lower() != corr_word.lower():
                        valid.append(tok_str)
                seen = set()
                valid = [x for x in valid if not (x in seen or seen.add(x))]

                if valid:
                    options = [corr_word] + valid
                    key = f"choice_{choice_index}"
                    default = st.session_state.get(key, corr_word)
                    choice = st.selectbox(
                        f"Replace '{orig_word}' → '{corr_word}':",
                        options=options,
                        index=options.index(default) if default in options else 0,
                        key=key
                    )
                    final_toks[j1] = choice
                    choice_index += 1

    final_sentence = detokenize(final_toks)
    st.subheader("🎯 Final Choice")
    st.success(final_sentence)
