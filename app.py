"""Streamlit UI for the Text Emotion Classifier.  Run: streamlit run app.py"""
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from predict import EmotionPredictor
from train_model import load_data, model_exists, train

st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="wide")

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}


@st.cache_resource
def get_predictor():
    return EmotionPredictor()


def emo(label):
    return f"{EMOJI.get(label, '🙂')} {label.capitalize()}"


st.title("🧠 Text Emotion Classifier")
st.caption("Detects the emotion behind a sentence using TF-IDF + machine learning.")

tab_predict, tab_batch, tab_perf, tab_train = st.tabs(
    ["🔮 Predict", "📄 Batch (CSV)", "📊 Model performance", "🛠️ Train / Retrain"]
)

# ---------------- Train tab ----------------
with tab_train:
    st.subheader("Train the model")
    st.write("Upload your `train.txt` (format: `text;emotion` per line, no header).")
    up = st.file_uploader("Training file", type=["txt", "csv"], key="train_up")
    if up is not None:
        df_train = load_data(io.BytesIO(up.getvalue()))
        st.write(f"Loaded **{len(df_train):,}** rows")
        st.dataframe(df_train.head(), use_container_width=True)
        if st.button("🚀 Train model", type="primary"):
            box = st.empty()
            with st.spinner("Training..."):
                train(df_train, progress=lambda m: box.info(m))
            get_predictor.clear()
            st.success("Model trained and saved! Go to the Predict tab.")

if not model_exists():
    for t in (tab_predict, tab_batch, tab_perf):
        with t:
            st.warning("No trained model found. Open the **Train / Retrain** tab and upload `train.txt` "
                       "(or run `python train_model.py --data train.txt`).")
    st.stop()

predictor = get_predictor()

# ---------------- Predict tab ----------------
with tab_predict:
    examples = {
        "— choose an example —": "",
        "I can't believe I finally got the job, I'm thrilled!": "I can't believe I finally got the job, I'm thrilled!",
        "I feel so lonely and empty tonight": "I feel so lonely and empty tonight",
        "I am furious that they lied to me": "I am furious that they lied to me",
        "I'm terrified of what might happen tomorrow": "I'm terrified of what might happen tomorrow",
    }
    choice = st.selectbox("Try an example", list(examples.keys()))
    text = st.text_area("Enter your text", value=examples[choice], height=120,
                        placeholder="Type how you feel...")
    if st.button("Analyze emotion", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            label, scores = predictor.predict(text)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted emotion", emo(label))
                st.metric("Confidence", f"{scores[label]*100:.1f}%")
            with c2:
                sdf = (pd.Series(scores).sort_values(ascending=False) * 100).rename("Probability (%)")
                st.bar_chart(sdf)

# ---------------- Batch tab ----------------
with tab_batch:
    st.write("Upload a CSV with a column named **text** to classify many rows at once.")
    bf = st.file_uploader("CSV file", type=["csv"], key="batch_up")
    if bf is not None:
        bdf = pd.read_csv(bf)
        if "text" not in bdf.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            preds = [predictor.predict(t) for t in bdf["text"].astype(str)]
            bdf["predicted_emotion"] = [p[0] for p in preds]
            bdf["confidence"] = [round(p[1][p[0]], 4) for p in preds]
            st.dataframe(bdf, use_container_width=True)
            st.download_button("⬇️ Download results", bdf.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ---------------- Performance tab ----------------
with tab_perf:
    m = predictor.metrics
    st.success(f"Best model in use: **{m['best_model']}**  (trained on {m['n_samples']:,} samples)")
    a, b = st.columns(2)
    with a:
        st.markdown("**Accuracy by model**")
        st.bar_chart(pd.Series(m["accuracies"]).rename("Accuracy"))
        st.markdown("**Class distribution**")
        st.bar_chart(pd.Series(m["class_counts"]).rename("Samples"))
    with b:
        st.markdown("**Confusion matrix (20% test split)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=m["labels"], yticklabels=m["labels"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    st.markdown("**Classification report**")
    rep = pd.DataFrame(m["report"]).T.round(3)
    st.dataframe(rep, use_container_width=True)
