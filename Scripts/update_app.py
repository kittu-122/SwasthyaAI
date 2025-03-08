import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import requests

# Load dataset
df = pd.read_csv("E:/Women Health advisor - Ayurveda/Womens_Health.csv", encoding='latin-1')
df.rename(columns={"Constitution/Prakriti": "Prakriti"}, inplace=True)

# Selecting relevant features for clustering
features = [
    "Symptoms", "Symptom Severity", "Age Group", "Medical History", 
    "Current Medications", "Physical Activity Levels", "Dietary Habits", 
    "Occupation and Lifestyle", "Doshas", "Prakriti"
]
df_selected = df[features].copy()

# Encode categorical variables
encoders = {col: LabelEncoder() for col in df_selected.columns if df_selected[col].dtype == 'object'}
for col, encoder in encoders.items():
    df_selected[col] = encoder.fit_transform(df_selected[col].astype(str))

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Hierarchical Clustering
dist_matrix = linkage(df_scaled, method='ward')
df["Cluster_HC"] = fcluster(dist_matrix, 5, criterion='maxclust')

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=5, random_state=42)
df["Cluster_GMM"] = gmm.fit_predict(df_scaled)

# Train Nearest Neighbors for chatbot recommendations
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
knn.fit(df_scaled)

# Save models
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(gmm, "gmm.pkl")
joblib.dump(knn, "knn.pkl")

# Set up Streamlit App
st.set_page_config(page_title="SwasthyaAI", page_icon="🩺", layout="wide")

# Sidebar: User Input
st.sidebar.title("🩺 SwasthyaAI: Ayurveda Health Assistant")
st.sidebar.header("🔍 Provide Your Health Details")

symptoms = st.sidebar.text_input("Enter Symptoms (comma-separated)")
symptom_severity = st.sidebar.slider("Symptom Severity", 1, 10, 5)
age_group = st.sidebar.selectbox("Select Age Group", df["Age Group"].unique())
medical_history = st.sidebar.text_input("Enter Medical History")
current_medications = st.sidebar.text_input("Enter Current Medications")
physical_activity = st.sidebar.selectbox("Select Physical Activity Levels", df["Physical Activity Levels"].unique())
dietary_habits = st.sidebar.selectbox("Select Dietary Habits", df["Dietary Habits"].unique())
occupation_lifestyle = st.sidebar.selectbox("Select Occupation and Lifestyle", df["Occupation and Lifestyle"].unique())
doshas = st.sidebar.selectbox("Select Doshas", df["Doshas"].unique())
prakriti = st.sidebar.selectbox("Select Prakriti", df["Prakriti"].unique())

# Welcome Section
st.title("🌟 SwasthyaAI: Smart Ayurveda for Women’s Wellness")
st.write("Welcome to **SwasthyaAI** – your AI-powered Ayurveda health assistant! 🌿💡")
st.success("Provide your health details to receive **personalized Ayurvedic insights**.")

# Chat Interface Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Done", "content": "___"}]

# Process Input and Provide Health Insights
if st.sidebar.button("🩺 Get Health Insights"):
    user_data = pd.DataFrame([[symptoms, symptom_severity, age_group, medical_history, current_medications,
                               physical_activity, dietary_habits, occupation_lifestyle, doshas, prakriti]], 
                              columns=features)
    
    # Encode user input
    for col in user_data.columns:
        if col in encoders and user_data[col].values[0] in encoders[col].classes_:
            user_data[col] = encoders[col].transform(user_data[col])
        else:
            user_data[col] = -1  # Handle unknown categories

    # Standardize input data
    user_data_scaled = scaler.transform(user_data)
    
    # Predict cluster
    cluster_gmm = gmm.predict(user_data_scaled)[0]
    
    # Find closest existing profile
    _, index = knn.kneighbors(user_data_scaled)
    matched_profile = df.iloc[index[0][0]]
    
    # Display Personalized Recommendations
    st.subheader("🌿 Personalized Ayurvedic Recommendations")
    st.markdown(f"**🩸 Disease:** {matched_profile['Disease']}")
    st.markdown(f"**📊 Diagnosis & Tests:** {matched_profile['Diagnosis & Tests']}")
    st.markdown(f"**⏳ Duration of Treatment:** {matched_profile['Duration of Treatment']}")
    st.markdown(f"**⚠️ Risk Factors:** {matched_profile['Risk Factors']}")
    st.markdown(f"**🌍 Environmental Factors:** {matched_profile['Environmental Factors']}")
    st.markdown(f"**🍀 Herbal Remedies:** {matched_profile['Herbal/Alternative Remedies']}")
    st.markdown(f"**🌿 Ayurvedic Herbs:** {matched_profile['Ayurvedic Herbs']}")
    st.markdown(f"**🥄 Recommended Herbal Formulation:** {matched_profile['Formulation']}")
    st.markdown(f"**☕ Recommended Herbal Tea:** {matched_profile['Herbal Tea(Recommended)']}")
    st.markdown(f"**🍽️ Diet & Lifestyle Tips:** {matched_profile['Diet and Lifestyle Recommendations']}")
    st.markdown(f"**🧘 Yoga & Physical Therapy:** {matched_profile['Yoga & Physical Therapy']}")
    st.markdown(f"**💊 Medical Intervention:** {matched_profile['Medical Intervention']}")
    st.markdown(f"**🛡️ Preventive Measures:** {matched_profile['Prevention']}")
    st.markdown(f"**❗ Complications to Watch For:** {matched_profile['Complications']}")
    st.markdown(f"**🧘 Recovery & Maintenance Tips:** {matched_profile['Recovery & Maintenance Tips']}")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "https://api.gemini.com/v1/complete",
                headers={"Authorization": f"Bearer {'your_gemini_api_key'}"},
                json={"prompt": user_query, "max_tokens": 150}
            )
            ai_response = response.json().get("output_text", "Sorry, unable to process your request.")
            st.write(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Clear Chat History
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
