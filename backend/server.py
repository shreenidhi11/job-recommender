from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
import io
from io import BytesIO
import re
import json
from pdfminer.high_level import extract_text
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pdfplumber
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter 

app = FastAPI()
# Allow CORS for specific origins (your React frontend)
origins = [
    "http://localhost:3000",  # React app running on port 3001
    "http://127.0.0.1:3000",  # Another possible React URL
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load skills from JSON file
def load_skills_from_json(json_file):
    with open(json_file, "r") as f:
        skills_data = json.load(f)
    return skills_data.get("skills", [])  # Ensure "skills" is the key in JSON

def preprocess_text(text):
    text = re.sub(r'(gpa|skills|linkedin\.com|github\.com|relevant courses|work experience|projects)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b', '', text)  
    text = re.sub(r'\b(?:\+?[0-9]{1,4}[ -]?)?[0-9]{7,10}\b', '', text) 
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path):

    print("inside the extracting the text from pdf now, inside job matching")

    #extract text from the pdf file given as input
    text = ""
    with pdfplumber.open(BytesIO(pdf_path.file.read())) as pdf:
    # with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return preprocess_text(text)


# Extract skills from resume
def extract_skills_from_resumes(resume_text, skills_list):
    skills = []
    
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, resume_text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills


def parse_resume_skills(file: UploadFile) -> List:
    """Extract contents from the resume (Placeholder)."""
    # Now lets extract the skills from the resume
    print("file object inside the parse_resume_skills", file)
    # Load skills from JSON
    skills_json_path = "unique_skills.json"  # Ensure the JSON file is in the correct directory
    skills_list = load_skills_from_json(skills_json_path)

    # Path to the resume PDF
    # resume_path = "/Users/shreenidhi/Desktop/Shreenidhi_Acharya_SDE1.pdf"
    # resume_path = '/Users/shreenidhi/Downloads/capstone_presentation/testing_clusters_resumes/fullstack_cloud/new4.pdf'

    resume_text = extract_text_from_pdf(file)
    print(resume_text)

    # Extract skills from resume
    extracted_skills_from_resume = extract_skills_from_resumes(resume_text, skills_list)

    # removing integers from the output
    extracted_skills_from_resume = [item for item in extracted_skills_from_resume if not item.isdigit()]

    # if extracted_skills_from_resume:
    #     print("Skills found from the resume:", extracted_skills_from_resume)
    # else:
    #     print("No matching skills found")

    return extracted_skills_from_resume, resume_text
    
def find_matching_clusters(resume_text) -> tuple:
    """Find best matching clusters for the given resume (Placeholder)."""
    # step 2: get matching job titles as per the model suggestion
    # Running the model to find the related job titles for the given resume
    # checking the resume for database cluster
    print("Inside finding the matching clusters- step 2")
    print(resume_text)

    # Load the datasets for tfidf
    technical_jobs_df_tfidf_clusters = pd.read_csv('/Users/shreenidhi/technical_jobs_cleaned_with_tfidf_clusters.csv',on_bad_lines='skip')

    # Model - tfidf with cosine similarity
    nltk.download('punkt')
    nlp = spacy.load("en_core_web_sm")  # Load spaCy model

    # pdf_path = "/Users/shreenidhi/Downloads/capstone_presentation/resume_pdfs/Resume_Sameer_Yadav.pdf"
    # pdf_path = "/Users/shreenidhi/Downloads/capstone_presentation/testing_clusters_resumes/fullstack_cloud/new4.pdf"

    # clean_text = extract_text_from_pdf(resume_text)

    # vectorizer
    vectorizer_tf = TfidfVectorizer()

    # fit the vectorizer on the combined_data column 
    vectorizer_tf.fit(technical_jobs_df_tfidf_clusters['combined_data'])

    # transform the 'combined_data' column into vectors
    combined_data_vectors = vectorizer_tf.transform(technical_jobs_df_tfidf_clusters['combined_data'])

    # transform the user's resume into a vector using the same vectorizer as above
    user_resume_vector = vectorizer_tf.transform([resume_text])

    # calculate cosine similarity
    similarity_scores = cosine_similarity(user_resume_vector, combined_data_vectors).flatten()

    # get the top 3 unique clusters
    top_3_indices = similarity_scores.argsort()[::-1] 
    top_3_clusters = []
    seen_clusters = set()

    for idx in top_3_indices:
        cluster = technical_jobs_df_tfidf_clusters.iloc[idx]['tf_idf_cluster']
        if cluster not in seen_clusters:
            top_3_clusters.append(cluster)
            seen_clusters.add(cluster)
        if len(top_3_clusters) == 3: 
            break

    print("Top 3 Predicted Clusters:", top_3_clusters)

    # do the matching
    all_matched_jobs_3 = technical_jobs_df_tfidf_clusters[
        technical_jobs_df_tfidf_clusters['tf_idf_cluster'].isin(top_3_clusters)
    ][['job_title', 'tf_idf_cluster']].drop_duplicates()

    print("Recommended Job Titles within Top 3 Clusters:\n", all_matched_jobs_3)

    job_title_list = all_matched_jobs_3['job_title'].tolist()


    # now get all the job skills for the matched job cluster
    # skill extraction remaining from the resultant job titles cluster

    print(top_3_clusters)
    filtered_cluster_df = technical_jobs_df_tfidf_clusters[technical_jobs_df_tfidf_clusters['tf_idf_cluster'].isin(top_3_clusters)]
    filtered_cluster_df['job_skills']

    skills_list = filtered_cluster_df["job_skills"].tolist()

    # skills_list = filtered_cluster_df["job_skills"].apply(lambda x: x.split(", ")).tolist()
    skills_list = filtered_cluster_df["job_skills"].apply(lambda x: str(x).split(", ")).tolist()



    # Flatten the list and get unique skills
    all_skills = [skill.strip() for sublist in skills_list for skill in sublist]  # Remove extra spaces
    unique_skills = set(all_skills)  # Convert to set to ensure uniqueness

    # Convert back to list if needed
    unique_skills_list = list(unique_skills)

    # Count the frequency of all skills
    major_skill_counts = Counter(all_skills)

    # Get all unique skills, no limitation to top 30
    all_skills_unique = list(major_skill_counts.keys())  # Using all unique skills from the counts


    # flattening all skills into a single list
    all_skills = [
        skill.strip() 
        for skills in filtered_cluster_df["job_skills"].dropna()
        for skill in skills.split(", ")
    ]

    major_skill_counts = Counter(all_skills)

    # top 100 most common skills
    top_100_skills = major_skill_counts.most_common(100)

    # store only the skill name
    skill_name = [skills[0] for skills in top_100_skills]

    return job_title_list, skill_name


        
def perform_skill_gap_analysis(resume_skills, matched_cluster_skills) -> Dict:
    resume_skills = set(resume_skills)
    matched_job_skills = set(matched_cluster_skills)
    # matched_job_skills = set(all_skills_unique)

    missing_skills = matched_job_skills.difference(resume_skills)

    print("Skill Gap:", missing_skills)
    return list(missing_skills)

@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...)):

    print(file)
    print("---------------------")
    """Main API endpoint to analyze resume and return job recommendations with skill gaps."""
    # first parse the resume and fetch the skills from the resume
    extract_skills_from_resume, cleaned_resume_text = parse_resume_skills(file)

    print("Printing the extracted skills from the resume")
    print(extract_skills_from_resume)
    # finding_matching_job_title_and_clusters should return the list of job titles and cluster list

    # find the clusters that match the resume and return the job titles and cluster matched skills
    job_titles, cluster_matched_skills = find_matching_clusters(cleaned_resume_text)
    print(job_titles, cluster_matched_skills)

    skill_gaps = perform_skill_gap_analysis(extract_skills_from_resume, cluster_matched_skills)
    print(skill_gaps)
    
    return {"recommended_jobs": job_titles[0:4], "skill_gaps": skill_gaps[0:101]}
