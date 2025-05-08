import streamlit as st
import pdfplumber
import csv
from langchain.llms import HuggingFaceHub

# Replace this token with your Hugging Face token
HUGGINGFACE_API_TOKEN = "your_hf_token_here"

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")
st.title("AI Resume Screener")

# Compatible HF model (does not require TGI or inference endpoint)
model_id = "google/flan-t5-base"

# Load the Hugging Face LLM
def get_llm_hf_inference(model_id="google/flan-t5-base", max_new_tokens=128, temperature=0.1):
    return HuggingFaceHub(
        repo_id=model_id,
        model_kwargs={"max_new_tokens": max_new_tokens, "temperature": temperature},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

# Extract text from PDF
def pdf_to_text(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Save results to CSV
def update_csv(results):
    with open('results.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Resume Name", "Missing Keywords", "Confidence Score", "Comments", "Suitability"])
        csv_writer.writerows(results)

# Function to get response from the model
def get_response(system_message, user_text, max_new_tokens=256):
    hf = get_llm_hf_inference(max_new_tokens=max_new_tokens, temperature=0.1)
    prompt = f"{system_message}\n\n{user_text}\n"
    return hf(prompt)

# Streamlit UI elements
uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description:")
mandatory_keywords = st.text_input("Enter Mandatory Keywords (comma-separated):")

if st.button("Analyze Resumes"):
    if not uploaded_files or not job_description or not mandatory_keywords:
        st.error("Please provide resumes, job description, and mandatory keywords.")
    else:
        results = []
        keywords_list = [kw.strip().lower() for kw in mandatory_keywords.split(',')]
        
        for uploaded_file in uploaded_files:
            resume_text = pdf_to_text(uploaded_file).lower()
            missing_keywords = [kw for kw in keywords_list if kw not in resume_text]
            
            system_message = "You are a recruitment AI that evaluates resumes based on job descriptions and mandatory keywords."
            user_text = (
                f"Job Description: {job_description}\n"
                f"Mandatory Keywords: {mandatory_keywords}\n"
                f"Resume: {resume_text}\n"
                f"Identify missing keywords, give a confidence score (0-100), and suggest improvements."
            )
            
            try:
                response = get_response(system_message, user_text)
            except Exception as e:
                response = f"Model Error: {str(e)}"
            
            confidence_score = 100 - (len(missing_keywords) * 10)
            confidence_score = max(0, confidence_score)
            
            suitability = "Suitable" if confidence_score >= 70 else "Maybe Suitable" if confidence_score >= 50 else "Not Suitable"
            
            results.append([uploaded_file.name, ", ".join(missing_keywords), confidence_score, response, suitability])
        
        st.session_state.results = results
        st.success("Analysis Complete!")
        for result in results:
            st.write(f"**{result[0]}** - {result[4]} (Confidence: {result[2]}%)")
            if result[1]:
                st.write(f"🔴 Missing Keywords: {result[1]}")
            st.write(result[3])

if "results" in st.session_state and st.session_state.results:
    update_csv(st.session_state.results)
    with open("results.csv", "rb") as file:
        st.download_button(label="Download CSV", data=file, file_name="results.csv", mime="text/csv")
