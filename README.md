# Text Emotion Classifier

Project files
- `text_preprocessing.py` – cleaning pipeline (lowercase, punctuation, numbers, emojis, stopwords)
- `train_model.py` – trains Naive Bayes (BoW), Naive Bayes (TF-IDF), Logistic Regression (TF-IDF); saves the best to `models/`
- `predict.py` – loads the saved model and predicts
- `app.py` – Streamlit UI
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
# put your train.txt (text;emotion) in this folder, then:
python train_model.py --data train.txt      # optional: you can also train from the UI
streamlit run app.py
```
