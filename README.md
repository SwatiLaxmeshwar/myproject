import os
import streamlit as st
import PyPDF2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# ----- Step 1: Load or Train the Model -----
@st.cache_resource

def train_or_load_model():
    if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
    else:
        # Sample training data
        data = [
            ("This is a resume with work experience and skills.", "Resume"),
            ("Invoice number 2345, due date, and total amount.", "Invoice"),
            ("This contract is entered into by and between parties.", "Contract"),
            ("Abstract, introduction, and methodology of research.", "Research Paper"),
            ("Receipt for your payment of $23.00.", "Receipt"),
        ] * 10  # Duplicate to simulate a small dataset

        df = pd.DataFrame(data, columns=["text", "label"])
        X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        clf = LogisticRegression()
        X_train_vec = vectorizer.fit_transform(X_train)
        clf.fit(X_train_vec, y_train)

        # Save model
        joblib.dump(clf, "model.joblib")
        joblib.dump(vectorizer, "vectorizer.joblib")

        model = clf

        # Print classification report (optional)
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        print(classification_report(y_test, y_pred))

    return model, vectorizer


# ----- Step 2: PDF/Text Processing -----
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# ----- Step 3: Streamlit UI -----
st.set_page_config(page_title="SmartScanner AI", layout="centered")
st.title("📄 SmartScanner – Document Type Classifier")

model, vectorizer = train_or_load_model()

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    if text:
        st.subheader("Extracted Text (First 500 chars)")
        st.text(text[:500] + ("..." if len(text) > 500 else ""))

        # Predict
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]
        confidence = max(model.predict_proba(X_input)[0])

        st.success(f"Predicted Document Type: **{prediction}**")
        st.info(f"Confidence Score: {confidence:.2f}")
    else:
        st.warning("Couldn't extract text from file.")


  
