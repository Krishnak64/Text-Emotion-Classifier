# 🧠 Text Emotion Analyzer

A machine learning web app that detects the emotion behind a piece of text — **joy, sadness, anger, fear, love or surprise** — built with Python, scikit-learn and Streamlit.

### 🚀 [Live Demo → text-emotion-analyzer-str.streamlit.app](https://text-emotion-analyzer-str.streamlit.app/)

---

## ✨ Features

- 🔮 **Single prediction** – type a sentence and get the predicted emotion, confidence and a probability chart
- 📄 **Batch prediction** – upload a CSV with a `text` column and download the results
- 📊 **Model performance** – accuracy comparison, class distribution, confusion matrix and classification report
- 🛠️ **Train / Retrain in the UI** – upload `train.txt` and train the model straight from the browser

## 🧪 How it works

1. **Preprocessing** – lowercase → remove punctuation → remove numbers → remove emojis → remove stopwords
2. **Vectorization** – Bag of Words / TF-IDF
3. **Models compared** – Naive Bayes (BoW), Naive Bayes (TF-IDF), Logistic Regression (TF-IDF)
4. **Best model** (by test accuracy) is saved as `model.pkl` + `vectorizer.pkl`

## 📁 Project structure

```
emotion_app/
├── app.py                  # Streamlit UI
├── train_model.py          # Trains models, saves best to models/
├── predict.py              # Loads saved model and predicts
├── text_preprocessing.py   # Text cleaning pipeline
├── requirements.txt
├── train.txt               # Dataset (text;emotion) – add your own
└── models/                 # Generated: model.pkl, vectorizer.pkl, metrics.json
```

## ⚙️ Run locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (or do it from the app's "Train / Retrain" tab)
python train_model.py --data train.txt

# 4. Launch the app
streamlit run app.py
```

## 📦 Dataset format

One sample per line, text and emotion separated by a semicolon, no header:

```
i didnt feel humiliated;sadness
i am feeling grouchy;anger
```

## ☁️ Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own copy:

1. Push this project to GitHub (include the `models/` folder, or train from the app after deploying)
2. Go to Streamlit Community Cloud → **New app**
3. Select your repo, branch and `app.py` as the main file
4. Click **Deploy**

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib · Seaborn · Streamlit

## 🔭 Future improvements

- Keep negation words ("not", "no") during preprocessing
- Handle class imbalance with `class_weight="balanced"`
- Add n-gram features and hyperparameter tuning
- Try deep learning models (LSTM / BERT)

