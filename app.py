python
Copy code
import streamlit as st
import requests
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Streamlit page configuration
st.set_page_config(page_title="JobMatcher Pro", layout="centered")
st.title("🔍 JobMatcher Pro")
st.write("Find jobs that match your skills and interests!")

# Step 1: Function to process text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Step 2: Fetch jobs from Adzuna API
def fetch_ai_jobs(app_id, app_key, location="Calgary", min_salary=0, max_salary=None, full_time=1, keywords="AI"):
    base_url = "http://api.adzuna.com/v1/api/jobs/ca/search/1"
    params = {
        "app_id": c439c644,
        "app_key": 
01163d7f667bc6da157e36a54a5e2abd,
        "results_per_page": 50,
        "what": keywords,
        "where": location,
        "salary_min": min_salary,
        "content-type": "application/json"
    }
    if max_salary is not None:
        params["salary_max"] = max_salary
    if full_time is not None:
        params["full_time"] = full_time
    response = requests.get(base_url, params=params)
    return response.json()['results'] if response.status_code == 200 else []

# Step 3: Vectorize job descriptions and user query
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts), vectorizer

# Step 4: Get Top 10 Recommendations with Feedback Scoring
def get_recommendations(jobs, user_query, feedback_weights=None):
    processed_query = preprocess_text(user_query)
    preprocessed_descriptions = [preprocess_text(job['description']) for job in jobs]
    texts_vectorized, vectorizer = vectorize_texts(preprocessed_descriptions + [processed_query])
    user_query_vector = texts_vectorized[-1]
    job_vectors = texts_vectorized[:-1]
    cos_similarities = cosine_similarity(job_vectors, user_query_vector).flatten()
    scaler = MinMaxScaler()
    cos_similarities_scaled = scaler.fit_transform(cos_similarities.reshape(-1, 1)).flatten()

    if feedback_weights is not None:
        weighted_similarities = cos_similarities_scaled * feedback_weights
    else:
        weighted_similarities = cos_similarities_scaled

    top_matches = weighted_similarities.argsort()[-10:][::-1]
    return [jobs[i] for i in top_matches]

# Streamlit App Layout
st.subheader("Enter Your Job Search Criteria")
keywords = st.text_input("Keywords or Fields of Interest", placeholder="e.g., Data Science, Machine Learning, AI")

st.subheader("Upload Your Resume")
uploaded_file = st.file_uploader("Upload a PDF file of your resume", type="pdf")

if uploaded_file and keywords:
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    resume_text = ""
    for page in range(pdf_reader.getNumPages()):
        resume_text += pdf_reader.getPage(page).extractText()
 # Fetch jobs based on keywords
    app_id = "your_app_id"   # Replace with your Adzuna app_id
    app_key = "your_app_key" # Replace with your Adzuna app_key
    jobs = fetch_ai_jobs(app_id, app_key, keywords=keywords)

    if jobs:
        st.success(f"Found {len(jobs)} job listings matching '{keywords}'.")

        # Initial feedback on first 3 jobs
        st.subheader("Rate Initial Job Suggestions")
        user_feedback = []
        for i, job in enumerate(jobs[:3]):
            st.write(f"**Job #{i+1}**")
            st.write(f"**Title:** {job['title']}")
            st.write(f"**Location:** {', '.join(job['location']['area'])}")
            st.write(f"**Description:** {job['description'][:250]}...")
            st.write(f"[Apply here]({job['redirect_url']})")
            rating = st.slider(f"Rate this job (1-5)", 1, 5, 3)
            user_feedback.append(rating)

        # Normalize feedback and get top recommendations
        feedback_weights = np.array(user_feedback) / np.max(user_feedback) if user_feedback else None
        recommendations = get_recommendations(jobs, resume_text, feedback_weights)

        # Display Top 10 Recommendations
        st.subheader("Top Job Matches Based on Your Profile")
        for job in recommendations:
            st.write("----")
            st.write(f"**Title:** {job['title']}")
            st.write(f"**Location:** {', '.join(job['location']['area'])}")
            st.write(f"**Description:** {job['description'][:250]}...")
            st.write(f"[Apply here]({job['redirect_url']})")
    else:
        st.error("No jobs found with the specified criteria.")
else:
    st.info("Please enter your job search criteria and upload your resume.")
