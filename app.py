import joblib
import streamlit as st
import pandas as pd
import numpy as np

best_model = joblib.load("best_xgb_model.pkl")
freq_map_train = joblib.load("freq_map_train.pkl")
country_map_train = joblib.load("country_freq_train.pkl")
level_map_train = joblib.load("level_map.pkl")
scaler_train = joblib.load("scaler.pkl")
mlb_encoder_train = joblib.load("mlb_encoder.pkl")
final_feature_train = joblib.load("final_model_features.pkl")

st.title("💼 Salary Prediction App")
st.markdown("Enter job details and get an estimated salary!")

# Sample lists (replace with your full lists)
job_titles = list(freq_map_train.keys())
countries = list(country_map_train.keys())
schedules = ['Full-Time', 'Contractor', 'Part-Time', 'Intership', 'Other']
exp_levels = ['Junior', 'Mid', 'Senior']
skills = sorted(mlb_encoder_train.classes_)

with st.form("user_input_form"):
    job_title = st.selectbox("Job Title", job_titles)
    job_country = st.selectbox("Job Country", countries)
    experience = st.radio("Experience Level", exp_levels)
    schedule = st.selectbox("Schedule Type", schedules)
    selected_skills = st.multiselect("Skills", skills)
    
    remote = st.checkbox("Remote Work")
    insurance = st.checkbox("Health Insurance Provided")
    degree = st.checkbox("Degree Mentioned in Job Post")

    submitted = st.form_submit_button("🚀 Predict Salary")

def preprocess_input(user_input_dict):
    # Convert to DataFrame
    #df = pd.DataFrame([user_input_dict])
    df = pd.json_normalize(user_input_dict)
    # Ensure job_skills is treated as a proper list column

    # Multi-label encoding (skills)
    import re
    import ast

    def sanitize_skills(skill_str):
        if isinstance(skill_str, list):
            # Already a list — sanitize each item
            return [str(skill).strip().lower() for skill in skill_str]

        if not isinstance(skill_str, str):
            return []  # handle None or unexpected types

        # Remove brackets if present (e.g. "[Excel, SQL]")
        skill_str = skill_str.strip().replace("[", "").replace("]", "")

        # Add quotes around each word-like token
        skill_str = re.sub(r"([\w\-+]+)", r"'\1'", skill_str)

        # Extract all quoted tokens
        tokens = re.findall(r"'[^']+'", skill_str)

        # Convert to lowercase list
        return [token.strip("'").lower() for token in tokens]
    
    df['job_skills'] = df['job_skills'].fillna("[]").apply(sanitize_skills)

    # Multi-label encoding
    skills_matrix = mlb_encoder_train.transform(df['job_skills'])
    skills_df = pd.DataFrame(skills_matrix, columns=mlb_encoder_train.classes_)
    skills_df = skills_df.reindex(columns=mlb_encoder_train.classes_, fill_value=0)
    df = pd.concat([df, skills_df], axis=1)
    df.drop(columns=['job_skills'], inplace=True)
    df.drop(columns=', ', errors='ignore', inplace=True)

    
    # Frequency encoding
    df['job_title_short'] = df['job_title_short'].map(freq_map_train).fillna(0)
    df['job_country'] = df['job_country'].map(country_map_train).fillna(0)

    # Ordinal encoding
    df['Experience_Level'] = df['Experience_Level'].map(level_map_train).fillna(-1).astype(int)
    # One-hot encoding alignment
    
    # One-hot encode job schedule type
    schedule_dummies = pd.get_dummies(df['job_schedule_type_cleaned'], prefix='job_schedule_type_cleaned', drop_first=False)

    # Only include columns that were present during training
    expected_schedule_cols = [col for col in final_feature_train if col.startswith("job_schedule_type_cleaned_")]
    schedule_dummies = schedule_dummies.reindex(columns=expected_schedule_cols, fill_value=0)

    # Drop original and attach encoded schedule columns
    df = pd.concat([df.drop(columns=['job_schedule_type_cleaned'], errors='ignore'), schedule_dummies], axis=1)
    df = df.reindex(columns=final_feature_train, fill_value=0)

    # Binary flags
    binary_cols = ['job_work_from_home', 'job_no_degree_mention', 'job_health_insurance']
    df[binary_cols] = df[binary_cols].astype(int)

    # Scaling
    scale_cols = ['job_title_short','job_country']
    df[scale_cols] = scaler_train.transform(df[scale_cols])

    # Final selected columns
    return df 

if submitted:
    user_dict = {
        "job_title_short": job_title,
        "job_work_from_home": remote,
        "job_no_degree_mention": degree,
        "job_health_insurance": insurance,
        "job_country": job_country,
        "job_skills": selected_skills,
        "Experience_Level": experience,
        "job_schedule_type_cleaned": schedule
    }

    try:
        processed_input = preprocess_input(user_dict)
        log_salary = best_model.predict(processed_input)
        salary = int(np.expm1(log_salary)[0])
        st.success(f"💰 Estimated Salary: ${salary:,}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

