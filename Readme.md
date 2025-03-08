# 🩺 Smart Women’s Health Assistant

## Overview
The **Smart Women’s Health Assistant** is a Streamlit-based web application that provides personalized lifestyle tips and medical advice based on women's health profiles. It leverages machine learning techniques such as Hierarchical Clustering and Gaussian Mixture Models (GMM) to group users with similar health conditions and offers tailored recommendations.

## Features
- **Data Preprocessing & Feature Engineering**: Categorical data encoding and dimensionality reduction using PCA.
- **Clustering**: Grouping health profiles using Hierarchical Clustering and GMM.
- **Interactive UI**: User-friendly interface for inputting health details and receiving personalized advice.
- **AI Chatbot**: Provides one-line explanations and health tips based on user queries.

## Setup Instructions

### 1. Create a Virtual Environment
```bash
pip install virtualenv
python -m venv health_env
health_env\Scripts\activate
```

### 2. Install Required Packages
```bash
pip install pandas numpy seaborn matplotlib streamlit joblib scikit-learn scipy
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Track Installed Packages
To track installed packages, generate a `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

## Usage
1. **Load Dataset**: The application loads the `Womens_Health.csv` dataset.
2. **User Inputs**: Users select their age group and primary symptom.
3. **Cluster Assignment**: The system assigns the user to a health cluster based on their inputs.
4. **AI Chatbot**: Users can ask health-related questions, and the chatbot provides personalized advice.

## Dataset
The dataset `Womens_Health.csv` contains the following columns:
- Disease
- Symptoms
- Diagnosis & Tests
- Symptom Severity
- Duration of Treatment
- Medical History
- Current Medications
- Risk Factors
- Environmental Factors
- Physical Activity Levels
- Dietary Habits
- Age Group
- Occupation and Lifestyle
- Herbal/Alternative Remedies
- Ayurvedic Herbs
- Formulation
- Herbal Tea (Recommended)
- Doshas
- Constitution/Prakriti
- Diet and Lifestyle Recommendations
- Yoga & Physical Therapy
- Medical Intervention
- Prevention
- Complications
- Recovery & Maintenance Tips

## Example User Inputs and Responses
- **User Input**: "What Ayurvedic herbs should I take?"
  - **AI Response**: "Suggested herbs: Ashoka, Shatavari, and Guggulu for hormonal balance."
- **User Input**: "What diet should I follow?"
  - **AI Response**: "Follow a warm, anti-inflammatory diet including turmeric, ginger, and ghee to balance Pitta dosha."

## Conclusion
The **Smart Women’s Health Assistant** integrates machine learning and Ayurvedic expertise to provide personalized health guidance. It ensures holistic healthcare recommendations, quick and interactive AI chatbot responses, and a science-backed approach to women's health.

## License
This project is licensed under the MIT License.
