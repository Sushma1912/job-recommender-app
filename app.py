<<<<<<< HEAD
# ==========================================================
# AI Job Recommender System - Local Streamlit App
# ==========================================================
# Author: Sushma
# Description: Upload your resume PDF → Get job recommendations + Apply links
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF text extraction

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
)

# ----------------------------------------------------------
# BANNER SECTION
# ----------------------------------------------------------
st.markdown("""
    <div style="
        background-color: #000000;
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 0 25px #00bfff;
    ">
        <h1 style="color:#00bfff; font-size:50px; font-family:'Trebuchet MS', sans-serif;">
            Job Recommender
        </h1>
        <p style="color:#cccccc; font-size:20px;">
            Upload your resume to instantly get job recommendations based on your skills and experience.
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD MODEL (cached)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

# ----------------------------------------------------------
# SAMPLE JOB DATASET
# ----------------------------------------------------------
jobs = pd.DataFrame([
    {
        "title": "Data Scientist",
        "company": "Google",
        "description": "Analyze big data, build ML models using TensorFlow, and visualize data insights.",
        "link": "https://careers.google.com/jobs/results/"
    },
    {
        "title": "AI Engineer",
        "company": "Microsoft",
        "description": "Develop AI models using Azure ML and deep learning frameworks such as PyTorch.",
        "link": "https://careers.microsoft.com/us/en/search-results"
    },
    {
        "title": "Data Analyst",
        "company": "Amazon",
        "description": "Perform SQL analytics, create dashboards, and analyze business performance.",
        "link": "https://www.amazon.jobs/en/"
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Meta",
        "description": "Build scalable ML pipelines, NLP solutions, and recommendation systems.",
        "link": "https://www.metacareers.com/jobs"
    },
    {
        "title": "Software Developer",
        "company": "IBM",
        "description": "Design cloud-based solutions, develop APIs, and collaborate in agile teams.",
        "link": "https://www.ibm.com/careers"
    }
])

# ----------------------------------------------------------
# JOB EMBEDDINGS
# ----------------------------------------------------------
@st.cache_data
def compute_job_embeddings(jobs_df):
    return model.encode(jobs_df["description"].tolist())

job_embeddings = compute_job_embeddings(jobs)
index = faiss.IndexFlatL2(job_embeddings.shape[1])
index.add(np.array(job_embeddings))

# ----------------------------------------------------------
# PDF TEXT EXTRACTION FUNCTION
# ----------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded resume PDF."""
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- Upload Resume  
- View Job Matches  
- Click Apply Links  
""")
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Sushma**")

# ----------------------------------------------------------
# MAIN SECTION: FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Analyzing your resume... ⏳"):
        resume_text = extract_text_from_pdf(uploaded_file)
        user_vector = model.encode([resume_text])
        distances, indices = index.search(np.array(user_vector), 3)

    st.success("✅ Here are your top job recommendations:")

    for i in indices[0]:
        job = jobs.iloc[i]
        st.markdown(f"""
        <div style="background-color:#111111; padding:20px; border-radius:15px; margin-bottom:20px;
        border:1px solid #00bfff; box-shadow:0 0 10px #00bfff;">
            <h3 style="color:#00bfff;">{job['title']} — {job['company']}</h3>
            <p style="color:#cccccc;">{job['description']}</p>
            <a href="{job['link']}" target="_blank" style="text-decoration:none;">
                <button style="background-color:#00bfff; color:white; padding:10px 18px;
                border:none; border-radius:8px; cursor:pointer;">Apply Now</button>
            </a>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Upload your resume PDF to start getting job recommendations.")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("""
<hr style="border: 1px solid #00bfff;">
<p style="text-align:center; color:gray;">
AI Job Recommender | Built by <b>Sushma</b> | © 2025
</p>
""", unsafe_allow_html=True)
=======
# ==========================================================
# AI Job Recommender System - Local Streamlit App
# ==========================================================
# Author: Sushma
# Description: Upload your resume PDF → Get job recommendations + Apply links
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF text extraction

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
)

# ----------------------------------------------------------
# BANNER SECTION
# ----------------------------------------------------------
st.markdown("""
    <div style="
        background-color: #000000;
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 0 25px #00bfff;
    ">
        <h1 style="color:#00bfff; font-size:50px; font-family:'Trebuchet MS', sans-serif;">
            Job Recommender
        </h1>
        <p style="color:#cccccc; font-size:20px;">
            Upload your resume to instantly get job recommendations based on your skills and experience.
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD MODEL (cached)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

# ----------------------------------------------------------
# SAMPLE JOB DATASET
# ----------------------------------------------------------
jobs = pd.DataFrame([
    {
        "title": "Data Scientist",
        "company": "Google",
        "description": "Analyze big data, build ML models using TensorFlow, and visualize data insights.",
        "link": "https://careers.google.com/jobs/results/"
    },
    {
        "title": "AI Engineer",
        "company": "Microsoft",
        "description": "Develop AI models using Azure ML and deep learning frameworks such as PyTorch.",
        "link": "https://careers.microsoft.com/us/en/search-results"
    },
    {
        "title": "Data Analyst",
        "company": "Amazon",
        "description": "Perform SQL analytics, create dashboards, and analyze business performance.",
        "link": "https://www.amazon.jobs/en/"
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Meta",
        "description": "Build scalable ML pipelines, NLP solutions, and recommendation systems.",
        "link": "https://www.metacareers.com/jobs"
    },
    {
        "title": "Software Developer",
        "company": "IBM",
        "description": "Design cloud-based solutions, develop APIs, and collaborate in agile teams.",
        "link": "https://www.ibm.com/careers"
    }
])

# ----------------------------------------------------------
# JOB EMBEDDINGS
# ----------------------------------------------------------
@st.cache_data
def compute_job_embeddings(jobs_df):
    return model.encode(jobs_df["description"].tolist())

job_embeddings = compute_job_embeddings(jobs)
index = faiss.IndexFlatL2(job_embeddings.shape[1])
index.add(np.array(job_embeddings))

# ----------------------------------------------------------
# PDF TEXT EXTRACTION FUNCTION
# ----------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded resume PDF."""
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- Upload Resume  
- View Job Matches  
- Click Apply Links  
""")
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Sushma**")

# ----------------------------------------------------------
# MAIN SECTION: FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Analyzing your resume... ⏳"):
        resume_text = extract_text_from_pdf(uploaded_file)
        user_vector = model.encode([resume_text])
        distances, indices = index.search(np.array(user_vector), 3)

    st.success("✅ Here are your top job recommendations:")

    for i in indices[0]:
        job = jobs.iloc[i]
        st.markdown(f"""
        <div style="background-color:#111111; padding:20px; border-radius:15px; margin-bottom:20px;
        border:1px solid #00bfff; box-shadow:0 0 10px #00bfff;">
            <h3 style="color:#00bfff;">{job['title']} — {job['company']}</h3>
            <p style="color:#cccccc;">{job['description']}</p>
            <a href="{job['link']}" target="_blank" style="text-decoration:none;">
                <button style="background-color:#00bfff; color:white; padding:10px 18px;
                border:none; border-radius:8px; cursor:pointer;">Apply Now</button>
            </a>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Upload your resume PDF to start getting job recommendations.")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("""
<hr style="border: 1px solid #00bfff;">
<p style="text-align:center; color:gray;">
AI Job Recommender | Built by <b>Sushma</b> | © 2025
</p>
""", unsafe_allow_html=True)
>>>>>>> adc6eeb82cdb9cbdfb6eb33e405d3e42be40ebad
