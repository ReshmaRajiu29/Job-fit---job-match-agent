import os
from urllib import response

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to the .env file."
    )
    st.stop()

groq_client = Groq(
    api_key=GROQ_API_KEY
)

class CandidateProfile(BaseModel):
    candidate_summary: str
    years_of_experience: float
    skills: list[str]
    education: list[str]
    key_projects: list[str]

class JobRequirements(BaseModel):
    role_title: str
    required_skills: list[str]
    preferred_skills: list[str]
    experience_required: str
    responsibilities: list[str]

class JobMatchResult(BaseModel):
    match_score: int
    matching_skills: list[str]
    missing_skills: list[str]
    not_evidenced_skills: list[str]
    experience_match: str    
    recommendations: list[str]
    interview_questions: list[str]

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    extracted_pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_pages.append(page_text)

    resume_text = "\n".join(extracted_pages)

    return resume_text

def extract_candidate_profile(
    resume_text,
    client,
    model_name
):
    prompt = f"""
You are a resume analysis assistant.

Extract the candidate profile from the resume below.

Resume:
{resume_text}

Return ONLY valid JSON using this structure:

{{
  "candidate_summary": "short professional summary",
  "years_of_experience": 0,
  "skills": ["skill"],
  "education": ["education"],
  "key_projects": ["project"]
}}

Rules:
- Use only information present in the resume
- Do not invent experience, skills, education, or projects
- years_of_experience must be a number
- If a field is not available, return an empty list
- Keep candidate_summary short and factual
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    result_text = (
        response.choices[0]
        .message.content
        .strip()
    )

    if result_text.startswith("```"):
        result_text = result_text.removeprefix("```json")
        result_text = result_text.removeprefix("```")
        result_text = result_text.removesuffix("```")
        result_text = result_text.strip()


    profile = CandidateProfile.model_validate_json(
        result_text
    )

    return profile

def extract_job_requirements(
    job_description,
    client,
    model_name
):
    prompt = f"""
You are a job description analysis assistant.

Extract the job requirements from the job description below.

Job Description:
{job_description}

Return ONLY valid JSON using this structure:

{{
  "role_title": "job title",
  "required_skills": ["skill"],
  "preferred_skills": ["skill"],
  "experience_required": "experience requirement",
  "responsibilities": ["responsibility"]
}}

Rules:
- Use only information present in the job description
- Do not invent requirements
- Put mandatory skills in required_skills
- Put optional or preferred skills in preferred_skills
- Keep the output concise
- If a field is not available, return an empty list or empty string
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1500
    )

    result_text = (
        response.choices[0]
        .message.content
        .strip()
    )

    if result_text.startswith("```"):
        result_text = result_text.removeprefix("```json")
        result_text = result_text.removeprefix("```")
        result_text = result_text.removesuffix("```")
        result_text = result_text.strip()

    job_requirements = (
        JobRequirements.model_validate_json(
            result_text
        )
    )

    return job_requirements

def compare_candidate_with_job(
    candidate_profile,
    job_requirements,
    client,
    model_name
):
    prompt = f"""
You are an AI job matching assistant.

Compare the candidate profile with the job requirements.

Candidate Profile:
{candidate_profile.model_dump_json(indent=2)}

Job Requirements:
{job_requirements.model_dump_json(indent=2)}

Return ONLY valid JSON using this structure:

{{
  "match_score": 0,
  "matching_skills": ["skill"],
  "missing_skills": ["skill"],
  "not_evidenced_skills": ["skill"],
  "experience_match": "short explanation"
  "recommendations": ["recommendation"],
  "interview_questions": ["question"]
}}

Rules:
- match_score must be between 0 and 100
- use only information present in the candidate profile
- do not invent candidate skills or experience
- matching_skills should contain requirements clearly supported by the resume
- missing_skills should contain important technical or role skills not supported by the resume
- not_evidenced_skills should contain requirements such as communication or soft skills that may exist but are not demonstrated in the resume
- consider required skills more important than preferred skills
- consider the experience requirement when calculating the score
- keep experience_match short and factual
- generate 3 practical recommendations
- recommendations must be based on the identified gaps
- do not suggest adding experience the candidate does not have
- generate exactly 5 interview questions
- interview questions should be relevant to this job description
- keep every recommendation to one short sentence
- keep every interview question concise
- return complete valid JSON only
- do not include markdown code fences
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    result_text = (
        response.choices[0]
        .message.content
        .strip()
    )

    if result_text.startswith("```"):
        result_text = result_text.removeprefix(
            "```json"
        )
        result_text = result_text.removeprefix(
            "```"
        )
        result_text = result_text.removesuffix(
            "```"
        )
        result_text = result_text.strip()

    match_result = (
        JobMatchResult.model_validate_json(
            result_text
        )
    )

    return match_result

st.set_page_config(
    page_title="AI Job Search Agent",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Job Search Agent")

st.write(
    "Upload your resume and paste a job description "
    "to analyse how well your profile matches the role."
)

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the job description",
    height=300,
    placeholder=(
        "Paste the complete job description here..."
    )
)

analyse_button = st.button(
    "Analyse Job Match",
    type="primary"
)

if analyse_button:
    if uploaded_resume is None:
        st.warning(
            "Please upload your resume."
        )
        st.stop()

    if not job_description.strip():
        st.warning(
            "Please paste a job description."
        )
        st.stop()

    resume_text = extract_text_from_pdf(
        uploaded_resume
    )

    if not resume_text.strip():
        st.error(
            "No readable text was found in the resume. "
            "Please upload a text-based PDF."
        )
        st.stop()

    try:
        with st.spinner(
            "Analysing candidate profile..."
        ):
            candidate_profile = (
                extract_candidate_profile(
                    resume_text,
                    groq_client,
                    GROQ_MODEL
                )
            )

    except Exception as error:
        st.error(
            "Unable to analyse the resume."
        )
        st.caption(str(error))
        st.stop()

    try:
        with st.spinner(
            "Analysing job requirements..."
        ):
            job_requirements = (
                extract_job_requirements(
                    job_description,
                    groq_client,
                    GROQ_MODEL
                )
            )

    except Exception as error:
        st.error(
            "Unable to analyse the job description."
        )
        st.caption(str(error))
        st.stop()

    try:
        with st.spinner(
            "Calculating job match..."
        ):
            match_result = (
                compare_candidate_with_job(
                    candidate_profile,
                    job_requirements,
                    groq_client,
                    GROQ_MODEL
                )
            )

    except Exception as error:
        st.error(
            "Unable to calculate the job match."
        )
        st.caption(str(error))
        st.stop()

    st.subheader("Candidate Profile")

    st.write(
        candidate_profile.candidate_summary
    )

    st.write(
        f"**Experience:** "
        f"{candidate_profile.years_of_experience} years"
    )

    st.write("**Skills:**")

    st.write(
        ", ".join(
            candidate_profile.skills
        )
    )

    st.subheader("Job Requirements")

    st.write(
        f"**Role:** "
        f"{job_requirements.role_title}"
    )

    st.write("**Required Skills:**")

    st.write(
        ", ".join(
            job_requirements.required_skills
        )
    )

    if job_requirements.preferred_skills:
        st.write("**Preferred Skills:**")

        st.write(
            ", ".join(
                job_requirements.preferred_skills
            )
        )

    st.write(
        f"**Experience Required:** "
        f"{job_requirements.experience_required}"
    )

    st.subheader("Job Match Analysis")

    st.metric(
        "Match Score",
        f"{match_result.match_score}%"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Matching Skills")

        for skill in match_result.matching_skills:
            st.write(f"✅ {skill}")

    with col2:
        st.write("### Missing Skills")

        for skill in match_result.missing_skills:
            st.write(f"❌ {skill}")

    if match_result.not_evidenced_skills:
        st.write("### Not Evidenced in Resume")

        for skill in (
            match_result.not_evidenced_skills
        ):
            st.write(f"⚠️ {skill}")

        st.write("### Experience Match")

    st.write(
        match_result.experience_match
    )

    st.write("### Recommendations")

    for recommendation in match_result.recommendations:
        st.write(f"• {recommendation}")

    st.write("### Interview Questions")

    for index, question in enumerate(
        match_result.interview_questions,
        start=1
    ):
        st.write(f"{index}. {question}")