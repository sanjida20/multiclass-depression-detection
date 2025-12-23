import streamlit as st

st.title("🤖 What is a Transformer-Based Model?")

st.markdown("""
## Transformer-Based Models
- Transformer-based model is a type of **neural network architecture** that excels at processing sequential data.
- Transformer models use **self-attention mechanisms** to understand contextual
relationships between words in a sentence.
- They are the foundation of modern NLP systems.

---

## General vs Domain-Specific Pre-training

### General Pre-trained Models
- Trained on large, generic corpora
- Capture broad linguistic patterns

**Models used in this project:**
- Bidirectional Encoder Representations from Transformers (BERT)
- Robustly Optimized BERT Pretraining Approach (RoBERTa)


### Domain-Specific Pre-trained Models
- Further pre-trained on mental health–related text
- Better at capturing psychological language cues

**Models used in this project:**
- MentalBERT (BERT adapted for mental health text)
- MentalRoBERTa (RoBERTa adapted for mental health text)

---

## Evaluation of Models Used in This Project
The four models were trained and tested using the best hyperparameters identified during hyperparameter tuning. The evaluation metrics used to compare their performance include:
- Accuracy
- Precision
- Recall
- F1-Score
- Area Under the Curve (AUC)
The results are summarized in the table below:
""")
st.image("images/results.png", caption="Comparison of Transformer-Based Models")
st.markdown("""
From the results, we can observe that **MentalBERT achieved the best overall performance**, with the highest Accuracy (0.9424) and AUC (0.9956).
Overall, domain-specific models (MentalBERT and MentalRoBERTa) outperform their general counterparts (BERT and RoBERTa) across all evaluation metrics.
""")