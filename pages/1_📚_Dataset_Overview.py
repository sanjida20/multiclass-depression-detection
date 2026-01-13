import streamlit as st
from textwrap import dedent

st.set_page_config(page_title="1 — Dataset Overview")

st.title("📚 Overview of the Multiclass Depression Tweets Dataset")

st.markdown("""
This page briefly describes the dataset used to train the multiclass depression detection models.

The goal is to help non-technical users understand where the data came from, what each label means, and how dataset patterns can affect model predictions.
""")

# Short facts
st.header("Quick facts")
st.markdown(
    dedent(
        """
        - **Source:** Public dataset, originally scraped from X (Twitter) using Apify. *(Muhammad Osama Nusrat et al., 2023)* 
        - **Size:** The dataset contains **~14,996** rows with two columns: *Tweets* and *Labels*.
        - **Target classes:** Bipolar, Atypical, Psychotic, Major Depressive, Postpartum, No Depression.
        - **Features:** Raw social media text (tweets) and annotated labels.
        - **Annotation:** Manual, context-aware labeling by the research team.
        """
    )
)

st.image("images/dataset.png", caption="Dataset snippet", width=600)
st.markdown("---")

# Dataset collection and annotation
st.header("How the Dataset was Created")
st.markdown(
    dedent(
        """
        - Since no public dataset existed for these depression subtypes in tweets, the dataset was **constructed from scratch**. *(Muhammad Osama Nusrat et al., 2023)*
        - Researchers **created lexicons for each subtype, verified them with a psychiatrist**, and scraped tweets using Apify.
        - The scraped tweets were **manually annotated**. Importantly, a tweet was labeled only if the context made it clear the author was describing their own condition.

        **Example lexicons used**
        - *Major depressive disorder*: "I have a major depressive disorder", "suffering from major depression", "major depressive episode"
        - *Bipolar*: "I have bipolar depression", "bipolar mood disorder", "bipolar"
        - *Atypical*: "atypical major depression", "hypersomnia", "feeling worthless"
        - *Psychotic*: "psychotic depression", "I have psychosis"
        - *Postpartum*: "postbirth depression", "I have postpartum depression"
        """
    )
)

st.markdown("---")

# Inclusion and exclusion
st.header("Inclusion & Exclusion Criteria (How Tweets were Selected)")
st.markdown(
    dedent(
        """
        **Inclusion**
        - Tweets where users self-report currently or historically being depressed.
        - Tweets containing verified lexicons such as: "I have bipolar depression", "I am suffering from atypical depression", "I have a major depressive disorder", etc.

        **Exclusion**
        - Non-public or paywalled research sources.
        - Non-English tweets.
        - Spammy or hashtag-only tweets, retweets, repetitive or incomplete tweets.
        """
    )
)

st.markdown("---")

# Wordcloud placeholders
st.header("Wordclouds by Class")
st.markdown("These wordclouds show the most common words in each class from the dataset.")

cols = st.columns(3)
with cols[0]:
    st.image("images/atypical.png", caption="Atypical - e.g. hypersomnia, sleep, tired")
with cols[1]:
    st.image("images/bipolar.png", caption="Bipolar - e.g. bipolar, mood, episode")
with cols[2]:
    st.image("images/major.png", caption="Major Depressive - e.g. hopeless, hard, life")

cols2 = st.columns(3)
with cols2[0]:
    st.image("images/postpartum.png", caption="Postpartum - e.g. baby, mother, overwhelmed")
with cols2[1]:
    st.image("images/psychotic.png", caption="Psychotic - e.g. psychotic, mind, trapped")
with cols2[2]:
    st.image("images/no.png", caption="No Depression - e.g. good, love, lol")

st.markdown("---")

# How dataset patterns affect predictions
st.header("How Dataset Characteristics can Affect Model Predictions?")
st.markdown(
    dedent(
        """
        - **Keyword bias:** Since tweets were collected using lexicons (e.g. "I have X"), models can become sensitive to explicit mentions of conditions. This helps precision for explicit posts but may miss implicit or ambiguous descriptions.
        - **Symptom overlap:** Words like "sleep", "tired", or "anxiety" appear across classes (and sometimes in non-depressed posts). This can cause confusion between related classes (e.g., atypical vs. major depression).
        - **Comorbidity signals:** Terms such as "anxiety" or "PTSD" in the major-depressive subset indicate comorbidity; the model may learn to associate these with certain labels even though they are not certain indicators.
        - **Context dependence:** Tweets that describe feelings without explicitly naming a condition are often ambiguous; the model may predict incorrectly when the author is describing general distress rather than a particular disorder.
        - **Platform differences:** Platform-specific language differs in style, length, or vocabulary compared to X/Twitter. Domain shift may reduce performance.

        **Practical takeaway:** The dataset is high-quality (manual annotation) but still limited by the lexicon-based scraping strategy and platform-specific language. Use XAI explanations to check whether the model relies on meaningful words or on accidental artifacts.
        """
    )
)
