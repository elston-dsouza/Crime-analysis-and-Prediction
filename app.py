import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv("data/crime_dataset_india.csv")
    return df

df = load_data()

# Load trained ML models
crime_domain_model = joblib.load("models/crime_domain.pkl")
case_closure_model = joblib.load("models/case_closure.pkl")

st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide")
st.title("🔎 Crime Analytics & Prediction Dashboard")

# Data Analysis
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Case Closure Distribution")
    fig, ax = plt.subplots(figsize=(4,4))
    df["Case Closed"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax,
        colors=["#ff9999","#66b3ff"],
        startangle=90,
        explode=[0.05,0]
    )
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
    st.subheader("Victim Gender Distribution")
    fig, ax = plt.subplots(figsize=(5,4))
    df["Victim Gender"].value_counts().plot(
        kind="bar",
        color="skyblue",
        edgecolor="black",
        ax=ax
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    st.pyplot(fig)

st.subheader("Top 5 Cities with Most Crimes")
top5_cities = df["City"].value_counts().head(5)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=top5_cities.index, y=top5_cities.values, palette="pastel", ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("")
st.pyplot(fig)

st.subheader("Top 5 Crime Types")
top5_crimes = df["Crime Description"].value_counts().head(5)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=top5_crimes.index, y=top5_crimes.values, palette="muted", ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(100), height=300)

# Predictions
st.header("ML Predictions")

# 1. Crime Domain Prediction
st.subheader("Predict Crime Domain")
with st.form(key="crime_domain_form"):
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", df["City"].dropna().unique())
        age = st.slider("Victim Age", 0, 100, 30)
        gender = st.selectbox("Victim Gender", df["Victim Gender"].dropna().unique())
    with col2:
        weapon = st.selectbox("Weapon Used", df["Weapon Used"].dropna().unique())
        crime_desc = st.selectbox("Crime Description", df["Crime Description"].dropna().unique())
    submit1 = st.form_submit_button("Predict Crime Domain")

if submit1:
    crime_domain_input = pd.DataFrame([[city, age, gender, weapon, crime_desc]],
                                      columns=['City', 'Victim Age', 'Victim Gender', 'Weapon Used', 'Crime Description'])
    pred_domain = crime_domain_model.predict(crime_domain_input)[0]
    st.success(f"Predicted Crime Domain: **{pred_domain}**")

# 2. Case Closure Prediction
st.subheader("Predict Case Closure")
with st.form(key="case_closure_form"):
    col1, col2 = st.columns(2)

    with col1:
        city2 = st.selectbox("City", df["City"].dropna().unique(), key="city2")
        age2 = st.slider("Victim Age", 0, 100, 25, key="age2")
        gender2 = st.selectbox("Victim Gender", df["Victim Gender"].dropna().unique(), key="gender2")

    with col2:
        weapon2 = st.selectbox("Weapon Used", df["Weapon Used"].dropna().unique(), key="weapon2")
        domain2 = st.selectbox("Crime Domain", df["Crime Domain"].dropna().unique(), key="domain2")
        crime_desc2 = st.selectbox("Crime Description", df["Crime Description"].dropna().unique(), key="crime_desc2")
        police = st.slider("Police Deployed", 0, 50, 10)

    submit2 = st.form_submit_button("Predict Case Closure")

if submit2:
    case_input = pd.DataFrame([[city2, crime_desc2, age2, gender2, weapon2, domain2, police]],
                              columns=['City', 'Crime Description', 'Victim Age', 'Victim Gender', 'Weapon Used', 'Crime Domain', 'Police Deployed'])
    st.write("Input columns:", case_input.columns.tolist())
    
    pred_case = case_closure_model.predict(case_input)[0]
    st.success(f"Prediction: **{'Closed' if pred_case == 1 else 'Open'}**")
    

#Footer
st.markdown("---")
st.caption("SafeState: Crime Analytics & Prediction")
