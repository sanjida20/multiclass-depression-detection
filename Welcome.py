import re
from attr import attr
import pandas as pd
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

st.set_page_config(
    page_title="Multiclass Depression Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Tabs spaced out but starting from very left */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: flex-start;
        padding-left: 0;
        padding-right: 0;
    }
    
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 12px 25px;
        margin: 0;
        border-radius: 4px 4px 0 0;
        background-color: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main title - outside tabs so it's always visible
st.title("🧠 Multiclass Depression Detection Platform")

# Create three tabs
tab_about, tab_info, tab_analysis = st.tabs([
    "📋 About", 
    "ℹ️ Information about Depression", 
    "🔍 Multiclass Depression Detector"
])

# TAB 1: ABOUT
with tab_about:
    st.header("Welcome 👋")
    st.markdown("""
    This platform demonstrates a **transformer-based system** for detecting **multiple types of depression** from social media text data. 
    
    The system integrates **Explainable AI (XAI)** technique to provide transparent and reliable predictions, making AI decision-making more interpretable.
    
    ---
    
    ### 📊 What is Multiclass Depression Detection?
    
    - **Multiclass classification** refers to categorizing data into more than two distinct classes
    - This system analyzes social media text to predict **six specific categories**:
    
    1. **Bipolar Depression** - Characterized by alternating periods of depression and mania
    2. **Atypical Depression** - Major Depressive Disorder with atypical features like mood reactivity and hypersomnia
    3. **Psychotic Depression** - Severe depression with psychosis features
    4. **Major Depressive Disorder** - Clinical depression with persistent low mood
    5. **Postpartum Depression** - Clinical depression occurring after childbirth
    6. **No Depression** - No clinical signs of depression
    
    ---
    
    ### 🎯 How to Use This Platform
    
    1. Navigate to the **"🔍 Multiclass Depression Detector"** tab
    2. Enter or paste social media text for analysis
    3. Select your preferred transformer-based predictive model
    4. View the depression type prediction
    5. Explore XAI explanations to understand model decisions
    
    """)

# TAB 2: INFORMATION ABOUT DEPRESSION
with tab_info:
    st.header("📚 Understanding Depression")
    st.markdown("""
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

# TAB 3: Multiclass Depression Detector
with tab_analysis:
    st.header("🔍 Multiclass Depression Detector")
    EXCLUDED_GLOBAL_TOKENS = {"<URL>", "<USER>"}

    @st.cache_resource
    def load_model(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto"
        )
        model.eval()
        return tokenizer, model

    MODEL_PATHS = {
        "MentalBERT": "sanjida20/mentalbert-depression",
        "MentalRoBERTa": "sanjida20/mentalroberta-depression",
        "BERT": "sanjida20/bert-depression",
        "RoBERTa": "sanjida20/roberta-depression",
    }

    TRUE_ID2LABEL = {
        0: "Atypical Depression",
        1: "Bipolar Depression",
        2: "Major Depressive Disorder",
        3: "No Depression",
        4: "Postpartum Depression",
        5: "Psychotic Depression",
    }

    model_name = st.radio(
        "Select Model:",
        options=list(MODEL_PATHS.keys()),
        horizontal=True 
    )
    tokenizer, model = load_model(MODEL_PATHS[model_name])

    # PATCH label mapping
    model.config.id2label = TRUE_ID2LABEL
    model.config.label2id = {v: k for k, v in TRUE_ID2LABEL.items()}

    id2label = model.config.id2label

    @st.cache_resource
    def load_pipeline(_model, _tokenizer):
        return pipeline(
            "text-classification",
            model=_model,
            tokenizer=_tokenizer,
            return_all_scores=True,
            function_to_apply="softmax"
        )

    clf_pipeline = load_pipeline(model, tokenizer)

    # ------------------ Integrated Gradients implementation ------------------
    def merge_tokens_by_offsets(text, tokens, offsets, attributions):
        """
        Merge subword tokens into human words using offset mappings and sum their attributions.
        Removes special tokens (offsets == (0,0)) and punctuation-only tokens.
        Returns lists: words, attrs (numpy array).
        """
        words = []
        attrs = []

        current_word = None
        current_attr = 0.0
        current_end = None

        for tok, off, attr in zip(tokens, offsets, attributions):
            s, e = off
            # skip special tokens (often have offset (0,0))
            if s == 0 and e == 0:
                continue
            # substring from original text
            token_text = text[s:e]
            if current_word is None:
                current_word = token_text
                current_attr = float(attr)
                current_end = e
            else:
                # contiguous/overlapping offsets -> continuation of same word
                if s <= current_end:
                    # append the intervening characters (keeps punctuation/spacing)
                    current_word += text[current_end:e]
                    current_attr += float(attr)
                    current_end = max(current_end, e)
                else:
                    # finalize previous word if not punctuation-only
                    if not re.fullmatch(r"\W+", current_word.strip()):
                        words.append(current_word)
                        attrs.append(current_attr)
                    # start new word
                    current_word = token_text
                    current_attr = float(attr)
                    current_end = e

        # finalize last
        if current_word is not None and not re.fullmatch(r"\W+", current_word.strip()):
            words.append(current_word)
            attrs.append(current_attr)

        return words, np.array(attrs)

    def integrated_gradients(tokenizer, model, text, target_label=None, steps=30, device=None):
        """
        Compute Integrated Gradients attributions for a single input text w.r.t. the target_label.
        Returns (tokens, attributions) where attributions is a 1-D numpy array aligned to tokens.


        This implementation integrates gradients w.r.t. the input embeddings and works with
        Hugging Face transformers models that accept inputs_embeds.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenize
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Determine target label (predicted if not provided)
        with torch.no_grad():
            outputs = model(**{k: v for k, v in enc.items() if k in ["input_ids", "attention_mask"]})
            logits = outputs.logits
            pred_label = int(torch.argmax(logits, dim=1).item())
        if target_label is None:
            target_label = pred_label

        # Get embeddings and baseline
        embed_layer = model.get_input_embeddings()
        embeddings = embed_layer(input_ids) # (1, seq_len, emb_dim)
        
        # Baseline: zeros or embedding of pad token if available
        if tokenizer.pad_token_id is not None:
            baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            baseline_emb = embed_layer(baseline_ids)
        else:
            baseline_emb = torch.zeros_like(embeddings)

        # Scale inputs and accumulate gradients
        embeddings_diff = embeddings - baseline_emb
        grads_sum = torch.zeros_like(embeddings, device=device)

        for alpha in np.linspace(0.0, 1.0, num=steps, endpoint=True):
            interpolated = baseline_emb + alpha * embeddings_diff
            interpolated = interpolated.clone().detach().requires_grad_(True)

            model.zero_grad()
            outputs = model(inputs_embeds=interpolated, attention_mask=attention_mask)
            logit = outputs.logits[0, target_label]

            # Compute gradients
            grad = torch.autograd.grad(logit, interpolated)[0] # shape (1, seq_len, emb_dim)
            grads_sum += grad.detach()

        avg_grads = grads_sum / steps # average gradient

        # Integrated gradients
        attributions = (embeddings_diff * avg_grads).sum(dim=-1).squeeze(0)  # (seq_len,)
        attributions = attributions.detach().cpu().numpy()

        # tokens and offsets
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        offsets = enc["offset_mapping"][0].tolist()

        # Merge subwords using offsets and remove special/punctuation tokens
        merged_words, merged_attrs = merge_tokens_by_offsets(text, tokens, offsets, attributions)

        return merged_words, merged_attrs, target_label

    def render_token_attributions(merged_words, merged_attrs):
        """
        Return an HTML string with merged words colored by attribution (red = positive, blue = negative).
        Each word is shown with a thin border, padding and a browser tooltip (title) that displays the
        numeric attribution score when hovered. Trailing punctuation is removed for display but the
        tooltip shows the original score.
        """
        if len(merged_attrs) == 0:
            return "<i>No tokens to display</i>"

        max_abs = np.max(np.abs(merged_attrs))
        if max_abs == 0:
            max_abs = 1.0

        html_parts = []
        for word, attr in zip(merged_words, merged_attrs):
            # strip leading/trailing punctuation but keep apostrophes (e.g. I've)
            display_word = re.sub(r"^[^\w']+|[^\w']+$", "", word.strip(), flags=re.UNICODE)
            if not display_word:
                continue

            intensity = float(min(1.0, abs(attr) / max_abs))
            if attr > 0:
                # positive -> red-ish
                r = 255
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
                bg_color = f"rgba({r}, {g}, {b}, 0.18)"
                border_color = f"rgba({r}, {int(40*(1-intensity))}, {int(40*(1-intensity))}, 0.6)"
            elif attr < 0:
                # negative -> blue-ish
                r = int(255 * (1 - intensity))
                g = int(255 * (1 - intensity))
                b = 255
                bg_color = f"rgba({r}, {g}, {b}, 0.18)"
                border_color = f"rgba({int(40*(1-intensity))}, {int(40*(1-intensity))}, {b}, 0.6)"
            else:
                bg_color = "transparent"
                border_color = "rgba(0,0,0,0.06)"

            # title tooltip shows the signed attribution with 4 decimal points
            title_txt = f"Attribution: {attr:.4f}"

            safe_word = (display_word.replace(" ", "&nbsp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            
            span = (
                f"<span title=\"{title_txt}\" "
                f"style=\"background:{bg_color}; padding:4px 6px; margin:3px; "
                f"border:1px solid {border_color}; border-radius:6px; display:inline-block; "
                f"font-size:0.95em; line-height:1.2; cursor:default;\">{safe_word}</span>"
            )
            html_parts.append(span)

        # wrap in a container for better spacing
        return "<div style='line-height:1.6; padding:6px 2px;'>" + " ".join(html_parts) + "</div>"

    # New helpers for improved local interpretation
    from collections import defaultdict

    def compute_local_aggregates(merged_words, merged_attrs):
        """
        Aggregate cleaned tokens for local explanation.
        Returns:
        - agg_list: list of dicts sorted by abs(sum_attr) desc, each dict:
            {'word': w, 'sum_attr': s, 'count': c, 'mean_attr': m, 'signed_mean': sm}
        - total_abs: sum of abs of all attributions (float)
        """
        agg = defaultdict(lambda: {"sum": 0.0, "count": 0})
        total_abs = 0.0

        for w, a in zip(merged_words, merged_attrs):
            if not w:
                continue
            w_clean = normalize_explanation_word(re.sub(r"^[^\w']+|[^\w']+$", "", w.strip(), flags=re.UNICODE))
            if not w_clean:
                continue
            # include all tokens for local aggregation (supporting + opposing)
            agg[w_clean]["sum"] += float(a)
            agg[w_clean]["count"] += 1
            total_abs += abs(float(a))

        agg_list = []
        for w, v in agg.items():
            s = v["sum"]
            c = v["count"]
            m = s / c if c else 0.0
            agg_list.append({
                "word": w,
                "sum_attr": s,
                "count": c,
                "mean_attr": m,
                "abs_sum": abs(s)
            })

        # sort by absolute total contribution (descending)
        agg_list = sorted(agg_list, key=lambda x: x["abs_sum"], reverse=True)
        return agg_list, (total_abs if total_abs != 0 else 1.0)


    def interpret_local_ig(agg_list, class_name, top_k=5):
        """
        Short, plain-English summary for the local explanation.
        agg_list: output from compute_local_aggregates
        """
        if not agg_list:
            return "The model did not find any strongly influential words in this text."

        # top supporting words (positive sums)
        top_support = [it for it in agg_list if it["sum_attr"] > 0][:top_k]
        top_support_text = ", ".join([f"“{it['word']}”" for it in top_support[:3]]) if top_support else "no obvious positive keywords"

        summary = (
            f"**Model’s best prediction:** {class_name} *(this is a screening output, not a clinical diagnosis)*.\n\n"
        )
        return summary


    def interpret_local_ig_detailed(agg_list, class_name, total_abs, top_k=10):
        """
        Detailed, human-friendly breakdown.
        - Shows top supporting and top opposing words
        - Reports percent contribution of each top word relative to the whole-text attribution magnitude
        - Gives a gentle 'what to do next' suggestion
        """
        if not agg_list:
            return "No meaningful word contributions were identified for this prediction."

        # split supporting/opposing
        supporting = [it for it in agg_list if it["sum_attr"] > 0]
        opposing = [it for it in agg_list if it["sum_attr"] < 0]

        # sort by absolute contribution
        supporting = sorted(supporting, key=lambda x: abs(x["sum_attr"]), reverse=True)[:top_k]
        opposing = sorted(opposing, key=lambda x: abs(x["sum_attr"]), reverse=True)[:top_k]

        def fmt(item):
            #pct = abs(item["sum_attr"]) / total_abs * 100
            sign = "+" if item["sum_attr"] > 0 else "-"
            return (
                f"**{item['word']}**: {sign}{abs(item['sum_attr']):.3f} "
                f"(occurrences={item['count']})"
            )

        lines = []
        lines.append("")
        if supporting:
            lines.append("**Top supporting words:**")
            for it in supporting:
                lines.append(f"- {fmt(it)}")
        else:
            lines.append("No strong supporting words detected.")

        lines.append("")
        if opposing:
            lines.append("**Top opposing words (these decreased support for the predicted class):**")
            for it in opposing:
                lines.append(f"- {fmt(it)}")
        else:
            lines.append("No strong opposing words detected.")

        # emphasize the most influential word (if present)
        top_overall = max(agg_list, key=lambda x: x["abs_sum"])
        #top_pct = top_overall["abs_sum"] / total_abs * 100
        lines.append("")
        lines.append(
            f"The most influential word is **{top_overall['word']}**, "
            "as it has the highest absolute attribution score in this text."
        )

        # gentle guidance
        lines.append("")
        lines.append(
            "_Note: Attribution scores are relative and are not a clinical diagnosis. "
            "If this output resonates with your experience, consider reaching out to a health professional or support line._"
        )

        return "\n\n".join(lines)


    # ------------------ End Integrated Gradients helpers ------------------

    # File Upload Option
    uploaded_file = st.file_uploader(
        "Upload a file with a single column 'text' (CSV or Excel format)", 
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file with a single column 'text' containing social media textual data. If the file has only one row, it will autofill the text area below for analysis."
    )

    uploaded_data = None

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                try:
                    # Read raw bytes and attempt utf-8 then fallback to latin1
                    raw = uploaded_file.getvalue()
                    try:
                        content = raw.decode('utf-8')
                    except Exception:
                        content = raw.decode('latin1')

                    lines = content.splitlines()

                    # Detect and remove header if it exists (case-insensitive)
                    if len(lines) > 0 and lines[0].strip().lower() == "text":
                        lines = lines[1:]

                    # Create DataFrame with single 'text' column
                    uploaded_data = pd.DataFrame({"text": lines})
                except Exception as e:
                    st.error(f"Error processing uploaded CSV file: {e}")
            elif uploaded_file.name.endswith('.xlsx'):
                uploaded_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")

            if uploaded_data is not None:
                st.write("Uploaded data preview:")
                st.write(uploaded_data.head())

                # Check for required text column
                if "text" not in uploaded_data.columns:
                    st.error("Uploaded file must contain a 'text' column.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

    # Determine default value for manual text area: autofill when exactly 1 row uploaded
    default_text = ""
    show_text_area = True
    if uploaded_data is not None:
        if len(uploaded_data) == 1:
            default_text = str(uploaded_data.iloc[0]["text"])
        elif len(uploaded_data) > 1:
            show_text_area = False # hide text area when multiple rows

    if show_text_area:
        manual_text = st.text_area("Enter text for analysis:", value=default_text, height=150)
    else:
        manual_text = None

    # Build texts_to_predict
    texts_to_predict = []
    single_text_for_xai = None
    if uploaded_data is not None:
        if len(uploaded_data) == 1:
            if manual_text and manual_text.strip():
                texts_to_predict = [manual_text]
                single_text_for_xai = manual_text
            else:
                texts_to_predict = [str(uploaded_data.iloc[0]["text"])]
                single_text_for_xai = str(uploaded_data.iloc[0]["text"])
        else:
            texts_to_predict = uploaded_data["text"].astype(str).tolist()
    else:
        if manual_text and manual_text.strip():
            texts_to_predict = [manual_text]
            single_text_for_xai = manual_text

    # ---- Global Aggregated IG for batch predictions ----
    def integrated_gradients_raw(tokenizer, model, text, target_label=None, steps=20, device=None):
        """
        Similar to integrated_gradients but returns raw (unnormalized) attributions per token as numpy array.
        Now returns (tokens, attributions_raw, target_label, offsets)

        Uses offset mappings so callers can merge subword tokens into words reliably.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Determine target label (predicted if not provided)
        with torch.no_grad():
            outputs = model(**{k: v for k, v in enc.items() if k in ["input_ids", "attention_mask"]})
            logits = outputs.logits
            pred_label = int(torch.argmax(logits, dim=1).item())
        if target_label is None:
            target_label = pred_label

        embed_layer = model.get_input_embeddings()
        embeddings = embed_layer(input_ids)

        if tokenizer.pad_token_id is not None:
            baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            baseline_emb = embed_layer(baseline_ids)
        else:
            baseline_emb = torch.zeros_like(embeddings)

        embeddings_diff = embeddings - baseline_emb
        grads_sum = torch.zeros_like(embeddings, device=device)

        for alpha in np.linspace(0.0, 1.0, num=steps, endpoint=True):
            interpolated = baseline_emb + alpha * embeddings_diff
            interpolated = interpolated.clone().detach().requires_grad_(True)

            model.zero_grad()
            outputs = model(inputs_embeds=interpolated, attention_mask=attention_mask)
            logit = outputs.logits[0, target_label]

            grad = torch.autograd.grad(logit, interpolated)[0]
            grads_sum += grad.detach()

        avg_grads = grads_sum / steps

        attributions = (embeddings_diff * avg_grads).sum(dim=-1).squeeze(0)
        attributions = attributions.detach().cpu().numpy().copy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        offsets = enc["offset_mapping"][0].tolist()

        return tokens, attributions, target_label, offsets

    def aggregate_ig_by_class(texts, tokenizer, model, steps=20, max_samples=100, device=None):
        """
        Compute Integrated Gradients for up to max_samples texts and aggregate mean absolute attributions per *word*
        per predicted class. Uses offsets->merge to reconstruct words from subword tokens and excludes special tokens.
        Returns dict: class_id -> list of (word, mean_abs_attr) sorted descending.
        """
        accum = {}  # class -> {word: [sum_abs, count]}
        n = min(len(texts), max_samples)
        samples = texts[:n]

        for text in samples:
            try:
                tokens, attrs, cls, offsets = integrated_gradients_raw(tokenizer, model, text, steps=steps, device=device)
            except Exception:
                continue

            # Merge subword tokens into words using offsets (leverages your earlier helper)
            merged_words, merged_attrs = merge_tokens_by_offsets(text, tokens, offsets, attrs)

            for word, a in zip(merged_words, merged_attrs):
                if word is None:
                    continue
                w = word.strip()
                if not w:
                    continue

                # Strip leading/trailing punctuation but keep internal apostrophes (e.g. I've)
                w_clean = normalize_explanation_word(w)
                if not w_clean:
                    continue

                # Exclude non-semantic placeholders from GLOBAL explanation
                if w_clean in EXCLUDED_GLOBAL_TOKENS:
                    continue

                # filter punctuation-only (safety)
                if re.fullmatch(r"[^\w']+", w_clean):
                    continue

                val = abs(float(a))
                if cls not in accum:
                    accum[cls] = {}
                if w_clean in accum[cls]:
                    accum[cls][w_clean][0] += val
                    accum[cls][w_clean][1] += 1
                else:
                    accum[cls][w_clean] = [val, 1]

        result = {}
        for cls, toks in accum.items():
            items = [(tok, total / count) for tok, (total, count) in toks.items()]
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            result[cls] = items_sorted
        return result

    def interpret_global_ig(class_name, top_words, n_samples):
        """
        Generate a human-readable explanation for global (aggregated) IG output.
        """
        word_list = ", ".join([f"“{w}”" for w in top_words[:5]])

        if n_samples <= 1:
            return (
                f"The model predicted **{class_name}** mainly because it focused on words such as "
                f"{word_list}. These words contributed positively to the prediction, meaning that "
                f"their presence increased the model’s confidence in this outcome."
            )

        if not top_words:
            return (
                f"The model did not identify any consistently influential words for the class "
                f"**{class_name}** across the dataset."
            )

        return (
            f"Across {n_samples} texts predicted as **{class_name}**, the model consistently "
            f"relied on words such as {word_list}. These words have higher average attribution "
            f"scores, indicating that they are commonly associated with this class in the dataset."
        )

    def normalize_explanation_word(w):
        w = w.strip()

        # Replace URLs
        if w.startswith("http"):
            return "<URL>"

        # Replace usernames
        if w.startswith("@"):
            return "<USER>"

        # Remove leading #
        if w.startswith("#"):
            w = w[1:]

        # Remove trailing punctuation/emojis but keep apostrophes
        w = re.sub(r"[^\w']+$", "", w, flags=re.UNICODE)
        w = re.sub(r"^[^\w']+", "", w, flags=re.UNICODE)

        return w if w else None

    # Prediction
    if st.button("Predict"):
        if not texts_to_predict:
            st.error("No valid text found for prediction.")
        else:
            predictions = []

            for text in texts_to_predict:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()

                # Handle string or int keys safely
                if isinstance(id2label, dict):
                    pred_label = id2label.get(str(pred_class), id2label.get(pred_class))
                else:
                    pred_label = id2label[pred_class]

                predictions.append(pred_label)

            # Output
            if uploaded_data is not None and len(uploaded_data) > 0:
                # attach predictions to a copy of uploaded_data
                result_df = uploaded_data.copy()
                result_df["Prediction"] = predictions

                st.success("Prediction completed!")
                st.write(result_df.head())

                # If dataset has more than 5 rows, offer full CSV download with predictions
                if len(result_df) > 5:
                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download full predictions as CSV",
                        data=csv_bytes,
                        file_name="Predictions.csv",
                        mime="text/csv"
                    )
                
                # ---- Global aggregated IG (batch) ----
                if len(result_df) > 1:
                    with st.spinner("Computing aggregated Integrated Gradients explanation..."):
                        texts_for_agg = result_df["text"].astype(str).tolist()
                        agg = aggregate_ig_by_class(texts_for_agg, tokenizer, model, steps=20, max_samples=100)

                    st.subheader("Aggregated Integrated Gradients Explanation for Input Dataset")
                    st.markdown(f"Using up to {min(100, len(texts_for_agg))} samples to compute aggregated IG per predicted class.")

                    # For each predicted class present in agg, show top tokens
                    for cls, items in agg.items():
                        class_name = id2label.get(cls, str(cls))
                        top_n = min(10, len(items))
                        if top_n == 0:
                            continue
                        top_items = items[:top_n]
                        words = [w for w, v in top_items]
                        vals = [v for w, v in top_items]

                        # compute how many samples in the dataset were predicted as this class
                        class_count = int((result_df["Prediction"] == class_name).sum())

                        # If there's only 1 sample, warn user that this is effectively the local explanation
                        if class_count <= 1:
                            st.info(f"Class **{class_name}** has only {class_count} sample in this dataset. Therefore, this explanation reflects the importance of words in that single sample rather than a general pattern.")

                        # Prepare pairs and sort descending by attribution
                        pairs = list(zip(words, vals))
                        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                        words_sorted = [p[0] for p in pairs_sorted]
                        vals_sorted = [p[1] for p in pairs_sorted]

                        # Dynamic figure height so labels don't overlap
                        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(words_sorted))))

                        ax.barh(words_sorted, vals_sorted, align='center')
                        ax.invert_yaxis()  # largest on top
                        ax.set_xlabel('Mean Integrated Gradients Attribution Score')
                        ax.set_title(f"Top {len(words_sorted)} Words Supporting Class: {class_name}  (n={class_count})")

                        # Annotate numeric values safely (inside OR outside)
                        max_val = max(vals_sorted) if vals_sorted else 1.0

                        # Ensure space on the right for outside labels
                        ax.set_xlim(0, max_val * 1.18)

                        for i, v in enumerate(vals_sorted):
                            # if bar is wide enough, label inside
                            if v >= 0.07 * max_val:
                                ax.text(
                                    v * 0.98,
                                    i,
                                    f"{v:.3f}",
                                    va='center',
                                    ha='right',
                                    fontsize=8,
                                    color='white'
                                )
                            else:
                                # bar is short, label outside
                                ax.text(
                                    v + 0.01 * max_val,
                                    i,
                                    f"{v:.3f}",
                                    va='center',
                                    ha='left',
                                    fontsize=8,
                                    color='black'
                                )

                        plt.tight_layout()
                        st.pyplot(fig)

                        # ---- Global explanation interpretation ----
                        interpretation_text = interpret_global_ig(
                            class_name=class_name,
                            top_words=words,
                            n_samples=class_count
                        )

                        st.markdown("**Interpreting the Predictions**")
                        st.write(interpretation_text)

                        st.divider()

                    st.caption("*Note: Attribution scores are relative and do not represent probabilities or clinical diagnoses.*")
                    
            else:
                # manual input case
                st.success(f"Prediction: **{predictions[0]}**")

            # If there is exactly one text being predicted, compute Integrated Gradients explanation
            if single_text_for_xai is not None:
                try:
                    # Fixed plain-English summary per class (single-input)
                    summary_card = {
                        "Major Depressive Disorder": "Language patterns commonly linked to persistent low mood, loss of interest, and hopelessness.",
                        "Postpartum Depression": "Language patterns often related to new-parent stress, low mood, and difficulty coping after childbirth.",
                        "Psychotic Depression": "Language may show unusual beliefs, disorganized thoughts, or perceptual disturbances; this is a sensitive flag.",
                        "Bipolar Depression": "Language showing mood swings or periods of both elevated and low mood.",
                        "Atypical Depression": "Language often associated with increased sleep, increased appetite, and mood reactivity.",
                        "No Depression": "The text does not show clear linguistic signals associated with depressive states according to the model."
                    }

                    # Use the first prediction as the single prediction label if available
                    single_pred_label = None
                    try:
                        if 'predictions' in locals() and len(predictions) > 0:
                            single_pred_label = predictions[0]
                    except Exception:
                        single_pred_label = None

                    # Fallback: try to infer from target_label after IG runs (kept as fallback)
                    summary_text = summary_card.get(single_pred_label, None)
                    if summary_text:
                        st.info(f"Summary: **{summary_text}**")
                    else:
                        # If predictions not present for some reason, skip summary for now (we'll display label after IG)
                        pass
                    
                    with st.spinner("Computing Integrated Gradients explanation…"):
                        merged_words, merged_attrs, target_label = integrated_gradients(tokenizer, model, single_text_for_xai, steps=30)

                    # Display predicted label for clarity
                    pred_label_name = id2label.get(target_label, str(target_label))
                    st.subheader(f"Integrated Gradients Explanation for Input Text")

                    # Render colored merged words with bordered spans and hover tooltip
                    html = render_token_attributions(merged_words, merged_attrs)
                    st.markdown(html, unsafe_allow_html=True)


                    # Small caption about tooltip
                    st.caption("Hover over each highlighted word to see its attribution score. Positive values increase confidence for the predicted class; negative values decrease it.")

                    # Bar chart of top positive attributions (supporting words)
                    pos_pairs = [(w, a) for w, a in zip(merged_words, merged_attrs) if a > 0]
                    pos_pairs_sorted = sorted(pos_pairs, key=lambda x: x[1], reverse=True)
                    top_n = min(10, len(pos_pairs_sorted))
                    top_pairs = pos_pairs_sorted[:top_n]

                    # ---- Local explanation interpretation ----
                    #top_words_local = [w for w, _ in top_pairs]
                    #interpretation_text = interpret_local_ig(top_words_local, pred_label_name)

                    if len(top_pairs) > 0:
                        words_top = [p[0] for p in top_pairs]
                        vals_top = [p[1] for p in top_pairs]

                        # Prepare local pairs sorted by attribution
                        # Aggregate duplicate cleaned words (positive attributions only) for local chart
                        from collections import defaultdict

                        # Build aggregated positive attributions keyed by cleaned word
                        agg_pos = defaultdict(float)
                        counts = defaultdict(int)

                        for w, a in zip(merged_words, merged_attrs):
                            # clean token to a human-friendly word (reuse your helper)
                            w_clean = normalize_explanation_word(re.sub(r"^[^\w']+|[^\w']+$", "", w.strip(), flags=re.UNICODE))
                            if not w_clean:
                                continue
                            if w_clean in EXCLUDED_GLOBAL_TOKENS:
                                continue
                            # only consider positive contributions for supporting-words chart
                            if float(a) > 0:
                                agg_pos[w_clean] += float(a)
                                counts[w_clean] += 1

                        # Convert to sorted list (descending by summed attribution)
                        pairs_local_sorted = sorted(agg_pos.items(), key=lambda x: x[1], reverse=True)

                        # Pick top-N to display (same as before)
                        top_n = min(10, len(pairs_local_sorted))
                        if top_n == 0:
                            st.info("No positive supporting tokens found to plot.")
                        else:
                            # Unzip top pairs
                            words_local_sorted, vals_local_sorted = zip(*pairs_local_sorted[:top_n])

                            # Convert to lists (safe for indexing)
                            words_local_sorted = list(words_local_sorted)
                            vals_local_sorted = list(vals_local_sorted)

                            # Plot horizontal bars
                            fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(words_local_sorted))))
                            ax.barh(words_local_sorted, vals_local_sorted, align='center')
                            ax.invert_yaxis()
                            ax.set_xlabel('Integrated Gradients Attribution Score')
                            ax.set_title(f"Top {len(words_local_sorted)} Words Supporting Prediction")

                            # numeric annotation: inside if wide enough, else outside (and expand xlim to fit)
                            max_local = max(vals_local_sorted) if vals_local_sorted else 1.0
                            # ensure some right padding for outside labels
                            ax.set_xlim(0, max_local * 1.18)

                            for i, v in enumerate(vals_local_sorted):
                                # threshold for "wide enough" — tune as needed (7% of max here)
                                if v >= 0.07 * max_local:
                                    # place inside bar, right-aligned, white text
                                    ax.text(v * 0.98, i, f"{v:.3f}", va='center', ha='right', fontsize=8, color='white')
                                else:
                                    # place just outside bar, left-aligned, black text (guaranteed inside axes because of set_xlim)
                                    ax.text(v + 0.01 * max_local, i, f"{v:.3f}", va='center', ha='left', fontsize=8, color='black')

                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.info("No positive supporting tokens found to plot.")

                    st.subheader("**Interpreting the Prediction**")
                    agg_list, total_abs = compute_local_aggregates(merged_words, merged_attrs)

                    # Short summary (friendly)
                    interpretation_text = interpret_local_ig(agg_list, pred_label_name, top_k=5)
                    st.write(interpretation_text)
                    detailed_text = interpret_local_ig_detailed(agg_list, pred_label_name, total_abs, top_k=5)
                    st.markdown(detailed_text)

                    # Full attribution table + download (for single-input local explanation)
                    # Prepare cleaned display words and the corresponding scores
                    display_words = []
                    display_scores = []
                    for w, a in zip(merged_words, merged_attrs):
                        disp = re.sub(r"^[^\w']+|[^\w']+$", "", w.strip(), flags=re.UNICODE)
                        if not disp:
                            continue
                        display_words.append(disp)
                        display_scores.append(float(a))

                    if display_words:
                        df_attrs = pd.DataFrame({"word": display_words, "attribution_score": display_scores})
                        df_attrs = df_attrs.sort_values(by='attribution_score', ascending=False).reset_index(drop=True)

                        st.subheader("Full Attribution Scores for All Words")
                        st.markdown("**Below is the full list showing how each word contributed to the predicted depression type for this text:**")
                        # show in an interactive dataframe view
                        st.dataframe(df_attrs)

                except Exception as e:
                    st.error(f"Error computing Integrated Gradients: {e}")


# Footer - visible across all tabs
st.divider()
st.caption("""
⚠️ **Disclaimer**: This platform is for research and educational purposes only. Not a clinical diagnostic tool. 
If you or someone you know is experiencing mental health concerns, please consult a qualified healthcare professional or contact emergency services.

""")


