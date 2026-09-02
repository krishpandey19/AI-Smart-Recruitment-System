# 🤖 AI Smart Recruitment System

An intelligent web-based recruitment system that helps recruiters find relevant candidates from a pool of resumes.

The idea behind this project is simple: instead of manually going through every resume for a job opening, the system analyzes the job requirements and resumes and gives recruiters a ranked list of candidates.

Built with **Python, Django, NLP, and Machine Learning**.

---

## 📌 Why I Built This

Recruiters can receive hundreds of resumes for a single job opening. Going through each resume manually can be time-consuming and inconsistent.

I wanted to build a system that could automate the initial screening process by:

- Extracting useful information from resumes
- Comparing resumes with job descriptions
- Filtering candidates based on basic requirements
- Calculating a similarity score
- Ranking candidates based on their relevance

This project also gave me hands-on experience with **Django, NLP, text processing, TF-IDF, and machine learning**.

---

## ✨ Features

### 👤 Candidate

- Create an account
- Browse available job openings
- View job details
- Apply for jobs
- Upload resume
- Track the application process

### 🧑‍💼 Recruiter

- Create and manage job openings
- Define job requirements
- View applications
- Analyze candidate resumes
- Get ranked candidates
- Shortlist relevant candidates

### 🧠 Resume Ranking

The system processes the resume and job description using NLP techniques and calculates their similarity.

The current ranking pipeline includes:

- Text extraction from resumes
- Text cleaning and preprocessing
- Tokenization
- Stop-word removal
- TF-IDF based text representation
- Similarity calculation
- Candidate ranking

---

## 🏗️ How It Works

The overall workflow looks like this:

```text
Recruiter Creates Job
        ↓
Job Description + Requirements
        ↓
Candidate Applies
        ↓
Resume Uploaded
        ↓
Resume Text Extraction
        ↓
Text Preprocessing
        ↓
TF-IDF Representation
        ↓
Similarity Calculation
        ↓
Candidate Ranking
        ↓
Recruiter Reviews Candidates
