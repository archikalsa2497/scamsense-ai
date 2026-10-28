import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
try:
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label_num"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Model retrained and saved successfully!")
except Exception as e:
    print(f"Error downloading online dataset: {e}")
