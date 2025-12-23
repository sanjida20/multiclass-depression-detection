import streamlit as st

st.set_page_config(
    page_title="Multiclass Depression Detection",
    layout="wide"
)

st.title("🧠 Multiclass Depression Detection Platform")

st.markdown("""
### Welcome 👋

This platform demonstrates a **transformer-based system** for detecting
**multiple subtypes of depression** from social media text data, integrated with **Explainable AI (XAI)** for transparency and reliability.

---

### What is Depression? 🤔

Depression is a common mental disorder involving a **depressed mood** or **loss of pleasure** or interest in activities for **long periods of time**.

### Symptoms of Depression
The symptoms of depression can vary slightly depending on the type and can range from mild to severe. In general, symptoms include:
""")
st.image("images/symptoms.jpg", caption="General symptoms of depression", width=400)

st.markdown("""
### Types of Depression
There are several types of depressive disorders. Clinical depression, or Major Depressive Disorder (MDD), is often just called 'depression'. It’s the most severe type of depression.
""")
st.image("images/types.jpg", caption="Common types of depression", width=600)
st.divider()

st.markdown("""          
### Impact of Depression
- Affects about **332 million** people worldwide **(~4% of the population).**
- Leads to more than **720,000 suicides** globally each year.
- In 2023, around **1 million Malaysian adults (4.6%)** have depression, doubling since 2019.

### Depression Detection Challenges
- Traditional diagnosis depends on patient self-report through **interviews or questionnaires**.
- However, self-reporting is prone to **recall and social desirability biases**.
- Limited access to mental health care due to **shortage of mental health professionals** and **high treatment costs** delay diagnosis and intervention.

### Modern Depression Detection
- **Digital phenotyping**: Monitors behavior via smartphone sensors to predict depressive symptoms.
- **Physiological sensing**: Wearable devices track early indicators of depression.
- **Speech analysis**: Voice and pitch patterns detect emotional states.
- **Social media analysis**: **Natural Language Processing** examines linguistic and behavioral cues in real-time textual data to identify depression.
""")
st.image("images/nlp.jpg", caption="There are 5.66 billion (68.7% of total population) social media user identities in October 2025", width=600)
st.divider()
st.markdown("""
### What is Multiclass Depression Detection? 🤔
- **Multiclass** refers to more than two classes. (e.g., Bipolar, Psychotic, Atypical, Postpartum, Major Depressive Disorder, No Depression). 
- In this project, **multiclass depression detection** refers to analyzing social media text data and predicting five types of depression (**Bipolar depression, Atypical depression, Psychotic depression, Major depressive disorder,** and **Postpartum depression**) or **no depression.**

---

⚠️ **Disclaimer**:  
This system is for **research and educational purposes only** and is **not a clinical diagnostic tool**.
""")
