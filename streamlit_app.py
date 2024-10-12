import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')
# Load the trained model and TF-IDF vectorizer
with open('voting_classifier_model.pkl', 'rb') as file:
    voting_clf = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)


# Preprocess function (same as in training)
def preprocess_text(text):
    import regex as re
    import string
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    stop_words = stopwords.words('english')
    ps = PorterStemmer()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokenized_words = word_tokenize(text.lower())  # Lowercase and tokenize
    tokenized_words = [ps.stem(word) for word in tokenized_words if word not in stop_words]  # Remove stopwords and stem
    return " ".join(tokenized_words)


# Streamlit app layout
st.title("Consumer Complaint Classification App")

st.write("""
This app classifies consumer complaints into one of the following categories:
- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Retail Banking
""")

# Input text from user
user_input = st.text_area("Enter the complaint narrative:")

# Function to plot confidence scores
def plot_confidence_scores(confidence_scores):
    categories = list(confidence_scores.keys())
    scores = list(confidence_scores.values())

    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=categories, palette='viridis')
    plt.xlabel("Confidence Score")
    plt.ylabel("Complaint Category")
    plt.title("Confidence Scores by Category")
    st.pyplot(plt)


if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        input_tfidf = tfidf.transform([processed_text]).toarray()  # Transform input to TF-IDF

        # Get probability predictions
        prediction_probs = voting_clf.predict_proba(input_tfidf)[0]

        # Map predictions to category names
        product_categories = ['Credit Card', 'Credit Reporting', 'Debt Collection', 'Mortgages and Loans', 'Retail Banking']

        # Get the predicted category with the highest confidence
        predicted_category = product_categories[np.argmax(prediction_probs)]

        # Create a dictionary of category and confidence scores
        confidence_scores = dict(zip(product_categories, prediction_probs))

        # Display the predicted category and confidence scores
        st.success(f"The complaint belongs to the category: **{predicted_category}**")
        st.write("Confidence scores for each category:")
        st.write(confidence_scores)

        # Plot the confidence scores
        plot_confidence_scores(confidence_scores)
    else:
        st.error("Please enter a complaint narrative.")
