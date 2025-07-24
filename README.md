# 💼 Salary Predictor App — Data Nerds Edition

This Streamlit-powered web app helps companies estimate the average salary they can offer for a data-related role, based on job specifics like:

- **Job Title** (e.g. Data Scientist, Data Engineer, Analyst)
- **Job Country**
- **Experience Level**: Junior | Mid | Senior
- **Schedule Type**: Full-Time | Part-Time | Contractor | Internship | Other
- **Required Skills**: Choose from 200+ technologies/tools
- **Remote Work Policy** ✅
- **Health Insurance Provided** ✅
- **Degree Mentioned in Job Post** ✅

💡 Behind the scenes, the app uses a trained XGBoost regression model with multi-label encoding, frequency encoding, one-hot encoding, and robust scaling — all wrapped in a clean input pipeline.

---

## 🚀 How to Use

1. Clone or fork this repository
2. Run the app locally:
   ```bash
   streamlit run app.py
--

## 🚀 Live Demo

Want to test the salary prediction tool in real-time?

👉 [Launch the app here](https://employeesalpred.streamlit.app)

No installation required — just visit and explore.


