import streamlit as st

st.set_page_config(page_title="What is XAI?", page_icon="🧠")
st.title("📖 What is Explainable AI (XAI)?")

st.markdown(
    """
Explainable Artificial Intelligence (XAI) is the ability of AI systems to provide clear and understandable explanations for their actions and decisions. Its central goal is to make the behaviour of these systems understandable to humans.
"""
)

st.image("images/xai.png", caption="XAI Concept")
st.markdown("---")

st.header("🧠 What This System Predicts")
st.markdown(
    """
The transformer-based models in this app predict one of the following categories from text:

- Bipolar Depression
- Atypical Depression
- Psychotic Depression
- Major Depressive Disorder
- Postpartum Depression
- No Depression

⚠️ **Important:** These results are **not medical diagnoses**. They are meant to highlight patterns in language.
"""
)

st.markdown("---")

st.header("🧠 How XAI Explains Its Decision (Integrated Gradients)")
st.markdown(
    """
In this platform, **Integrated Gradients (IG)** is used to show **how much each word contributed** to the model's prediction.

In plain terms:
- Each word gets a numeric "influence score" (the attribution score).
- **Positive scores** (e.g. `+3.1`) mean the word **supports** the predicted class.
- **Negative scores** (e.g. `-1.2`) mean the word **pushes away** from that class.
- The **bigger the absolute number**, the **stronger** the effect.

"""
)

st.subheader("What does a score like +3.057 mean?")
st.markdown(
    """
A score of **`+3.057`** means that this particular word **strongly pushed** the model toward the predicted depression category. Use the sign to know direction (positive = supports, negative = opposes) and the magnitude to know strength.
"""
)

st.markdown("---")

st.subheader("Sample IG bar chart")
st.markdown(
    """ 
Below is a sample of a bar chart output showing the top contributing words and their IG scores.
"""
)
st.image("images/barchart.png", caption="Sample IG bar chart")

st.markdown("---")

st.header("⚠️ Safety & Limitations")
st.markdown(
    """
- This system **does not** provide medical advice.
- Do **not** use predictions or IG explanations to self-diagnose.
- If you or someone you know is experiencing mental health concerns, please consult a qualified healthcare professional or contact emergency services.
"""
)