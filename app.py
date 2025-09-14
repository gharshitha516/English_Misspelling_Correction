import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from difflib import SequenceMatcher
import re
import string

# -------------------------------
# Helpers
# -------------------------------
def tokenize_with_punct(text: str):
    """Tokenize text into words and punctuation tokens."""
    # keeps words and individual punctuation as separate tokens
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def is_punct(tok: str):
    return re.fullmatch(r"[^\w\s]", tok) is not None

def detokenize(tokens):
    """Rebuild text from tokens with reasonable spacing around punctuation."""
    out = ""
    for tok in tokens:
        if out == "":
            out = tok
        elif is_punct(tok):
            # no space before punctuation
            out += tok
        elif out[-1] in "([{\"'":  # opening punct, don't add extra space
            out += tok
        else:
            out += " " + tok
    return out.strip()

# stricter filter: only allow single-word alphabetic tokens (allow internal hyphen/apostrophe)
def filter_suggestion(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # no pure punctuation or numbers
    if all(ch in string.punctuation for ch in s):
        return False
    if any(ch.isdigit() for ch in s):
        return False
    # only single token (no spaces)
    if " " in s:
        return False
    # allow letters, apostrophes, hyphens
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9'’-]*", s) is not None

# -------------------------------
# Load correction model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "harshhitha/FTe2_Misspelling"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return mdl, tok

model, tokenizer = load_model()

# -------------------------------
# Load BERT fill-mask pipeline (cached)
# -------------------------------
@st.cache_resource
def load_masker():
    return pipeline("fill-mask", model="bert-base-uncased")

masker = load_masker()
mask_token = getattr(masker.tokenizer, "mask_token", "[MASK]")

# -------------------------------
# UI - minimalistic
# -------------------------------
st.markdown("<h1 style='text-align:center;'>✒️ SpellFixer Pro</h1>", unsafe_allow_html=True)
user_input = st.text_area("Enter your sentence:", height=150, placeholder="Type with mistakes...")

if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting your text…"):
            inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # show corrected sentence
        st.subheader("✅ Corrected Sentence")
        st.success(corrected_text)

        # Tokenize original & corrected (keeps punctuation separate)
        orig_toks = tokenize_with_punct(user_input)
        corr_toks = tokenize_with_punct(corrected_text)

        # We'll build final_tokens from corrected tokens, replacing where user chooses alternatives
        final_toks = corr_toks.copy()

        st.subheader("🔄 Word Suggestions (Optional)")

        # Use SequenceMatcher to align tokens and detect replacements
        sm = SequenceMatcher(None, orig_toks, corr_toks)
        choice_index = 0  # for unique keys

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # no action needed
                continue
            elif tag == "replace":
                # only handle single-token replacements (common case)
                # if replaced span lengths are >1, fallback to leaving corrected tokens as-is
                if (i2 - i1) == 1 and (j2 - j1) == 1:
                    orig_word = orig_toks[i1]
                    corr_word = corr_toks[j1]

                    # skip if either token is punctuation
                    if is_punct(corr_word):
                        continue

                    # build a masked sentence for BERT: replace corr_toks[j1] with mask_token
                    masked = corr_toks.copy()
                    masked[j1] = mask_token
                    masked_sentence = detokenize(masked)

                    # request suggestions
                    candidates = masker(masked_sentence)[:12]  # top 12
                    # extract cleaned token_str and filter
                    valid = []
                    for c in candidates:
                        tok_str = c.get("token_str", "")
                        tok_str = tok_str.strip()
                        # remove leading/trailing quotes or stray symbols
                        tok_str = tok_str.strip(" \t\n'\"")
                        if tok_str and filter_suggestion(tok_str) and tok_str.lower() != corr_word.lower():
                            valid.append(tok_str)

                    # dedupe preserving order
                    seen = set()
                    valid = [x for x in valid if not (x in seen or seen.add(x))]

                    if valid:
                        options = [corr_word] + valid
                        # show dropdown: default index=0 means keep corrected word unless user picks otherwise
                        key = f"choice_{choice_index}"
                        choice = st.selectbox(f"Replace '{orig_word}' → '{corr_word}':", options=options, index=0, key=key)
                        # ensure replacement in final tokens
                        final_toks[j1] = choice
                        choice_index += 1
                    else:
                        # no useful alternatives; keep corrected token
                        continue
                else:
                    # complex replace (multi-token) — skip alternatives and keep corrected tokens
                    continue
            elif tag == "delete":
                # original had tokens deleted in corrected; nothing to show
                continue
            elif tag == "insert":
                # corrected inserted tokens; nothing to do
                continue

        # Build final sentence
        final_sentence = detokenize(final_toks)
        st.subheader("🎯 Final Choice")
        st.success(final_sentence)


