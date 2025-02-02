import streamlit as st

import translation


@st.cache_resource()
def load_model():
    return translation.prepare_model()


test, model_gen_test = load_model()
random_text = ()


def store_random_text():
    random_text = translation.randomize_text(test)
    st.session_state.random_text = random_text
    st.session_state.input_text = random_text[0]
    st.session_state.ground_truth = random_text[1]
    st.session_state.translation_output = ""


st.title("English to Thai Machine Translator")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "random_text" not in st.session_state:
    st.session_state.random_text = ()
if "translation_output" not in st.session_state:
    st.session_state.translation_output = ""

col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

with col2:
    if st.button("Randomize 🎲"):
        store_random_text()
        if st.session_state.input_text.strip():
            st.session_state.translation_output = translation.translate_text(model_gen_test, st.session_state.random_text)


with col1:
    input_text = st.text_area("Input Text (English):", key="input_text", height=120, disabled=True)

st.markdown("---")

if st.session_state.translation_output:
    with st.container():
        st.success("Translation Completed!")
        st.text_area("Ground Truth (Thai):", value=st.session_state.ground_truth, height=120, disabled=True)
        st.text_area("Translated Text using General Attention (Thai):", value=st.session_state.translation_output, height=120, disabled=True)
