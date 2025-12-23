import streamlit as st

st.title("📖 What is Explainable AI (XAI)?")

st.markdown("""
Explainable Artificial Intelligence (XAI) is the ability of AI systems to provide clear and understandable explanations for their actions and decisions. Its central goal is to make the behaviour of these systems understandable to humans. 
""")
st.image("images/xai.png", caption="XAI Concept")

st.markdown("---")

st.header("🧠 What Does This System Predict?")

st.markdown("""
The transformer-based models predict one of the following depression type based on text:

- Bipolar Depression
- Atypical Depression
- Psychotic Depression
- Major Depressive Disorder
- Postpartum Depression
- No Depression

⚠️ **Important:** These results are **not medical diagnoses**. They are only meant to highlight patterns in language.
""")

st.markdown("---")

st.header("🧮 How the XAI Explains Its Decision")

st.markdown("""
This system uses a method called **Integrated Gradients**.

It helps explain predictions by showing **how much each word influenced the result**.
Words that strongly express emotions or experiences often have more influence than neutral words.
""")

st.markdown("---")

st.header("🔬 How Does Integrated Gradients Work?")

st.markdown("""
Integrated Gradients works in a simple way:

1. The model starts with a **neutral version** of the text
2. It slowly moves towards the real sentence
3. It checks how the prediction changes along the way
4. Each word receives a **contribution score**

These scores help show which words pushed the prediction **towards** or **away from** a certain category.
""")

#st.image("images/ig_process_placeholder.png", use_column_width=True)

st.markdown("---")

st.header("⚠️ Important Safety Note")

st.markdown("""
- This system does **not** provide medical advice
- Results should not be used for self-diagnosis
- If you are concerned about mental health, please seek support from a qualified professional
""")