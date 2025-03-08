import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv("E:/Women Health advisor - Ayurveda/Womens_Health.csv", encoding='ISO-8859-1')

# Encode categorical features
encoder = LabelEncoder()
for col in ["Symptoms", "Doshas", "Ayurvedic Herbs", "Age Group", "Dietary Habits"]:
    df[col] = encoder.fit_transform(df[col])

# Create a mapping for symptoms to display text instead of numerical values
symptom_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Dimensionality Reduction using PCA
features = ["Symptoms", "Doshas", "Age Group"]
X = df[features]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df["PCA_1"], df["PCA_2"] = X_pca[:, 0], X_pca[:, 1]

# Apply Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Cluster_HC"] = hierarchical.fit_predict(X_pca)

gmm = GaussianMixture(n_components=3, random_state=42)
df["Cluster_GMM"] = gmm.fit_predict(X_pca)

# Streamlit UI
st.set_page_config(page_title="Smart Women’s Health Assistant", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #4b47cc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Smart Women’s Health Assistant")
st.subheader("Get AI-driven personalized lifestyle and medical guidance tailored to your health profile.")

# User Inputs
st.markdown("### Enter Your Details")
age_group = st.selectbox("Select Age Group", ["18-25", "26-35", "36-45", "46+"])
symptoms = st.multiselect("Select Your Symptoms", df["Symptoms"].unique())
other_symptom = st.text_input("Enter other symptoms if not listed")

# Combine symptoms
if other_symptom:
    symptoms.append(other_symptom)

# Find Cluster
encoded_symptoms = [encoder.transform([symptom])[0] for symptom in symptoms if symptom in encoder.classes_]
user_data = df[(df["Age Group"] == age_group) & (df["Symptoms"].isin(encoded_symptoms))]
if not user_data.empty:
    cluster = user_data["Cluster_HC"].values[0]
    st.write(f"🔍 You belong to **Cluster {cluster}**")

    # AI Chatbot with One-Line Explanations
    st.subheader("💬 AI Health Assistant")
    user_question = st.text_input("Ask me about your health (e.g., diet, stress, yoga)")

    response_dict = {
        "Symptoms": f"{user_data['Symptoms'].values[0]} is a key indicator of health imbalance.",
        "Diagnosis & Tests": f"Suggested tests: {user_data['Diagnosis & Tests'].values[0]}.",
        "Symptom Severity": f"Severity level: {user_data['Symptom Severity'].values[0]}.",
        "Duration of Treatment": f"Expected treatment duration: {user_data['Duration of Treatment'].values[0]}.",
        "Medical History": f"Previous conditions affecting health: {user_data['Medical History'].values[0]}.",
        "Current Medications": f"Ongoing medications include {user_data['Current Medications'].values[0]}.",
        "Risk Factors": f"Risk factors: {user_data['Risk Factors'].values[0]}.",
        "Environmental Factors": f"Impact of surroundings: {user_data['Environmental Factors'].values[0]}.",
        "Physical Activity Levels": f"Suggested physical activity: {user_data['Physical Activity Levels'].values[0]}.",
        "Dietary Habits": f"Recommended diet: {user_data['Dietary Habits'].values[0]}.",
        "Herbal/Alternative Remedies": f"Alternative therapy: {user_data['Herbal/Alternative Remedies'].values[0]}.",
        "Ayurvedic Herbs": f"Suggested herbs: {user_data['Ayurvedic Herbs'].values[0]}.",
        "Formulation": f"Herbal formulation: {user_data['Formulation'].values[0]}.",
        "Herbal Tea": f"Best herbal tea: {user_data['Herbal Tea(Recommended)'].values[0]}.",
        "Doshas": f"Dosha imbalance: {user_data['Doshas'].values[0]}.",
        "Constitution/Prakriti": f"Body constitution type: {user_data['Constitution/Prakriti'].values[0]}.",
        "Diet and Lifestyle Recommendations": f"Lifestyle changes: {user_data['Diet and Lifestyle Recommendations'].values[0]}.",
        "Yoga & Physical Therapy": f"Best yoga practice: {user_data['Yoga & Physical Therapy'].values[0]}.",
        "Medical Intervention": f"Medical approach: {user_data['Medical Intervention'].values[0]}.",
        "Prevention": f"Preventive measures: {user_data['Prevention'].values[0]}.",
        "Complications": f"Possible complications: {user_data['Complications'].values[0]}.",
    }

    if st.button("Get My Personalized Advice"):
        response = response_dict.get(user_question, "Please ask a specific health-related question.")
        st.write("🤖 AI Advice:", response)
else:
    st.write("⚠️ No matching data found. Try selecting other symptoms.")
