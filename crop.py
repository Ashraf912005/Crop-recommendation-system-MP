import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model and Columns
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_model()

# -------------------------------
# Load Dataset for Options
# -------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("Maharashtra_crop_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

df = load_dataset()

available_districts = df["district"].unique()
available_soiltypes = df["soiltype"].unique()
available_seasons = df["season"].unique()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌾 Crop Recommendation System (Maharashtra)")
st.write("Provide the details below to get a recommended crop for your region and soil conditions.")

# --- Input Form ---
with st.form("crop_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", sorted(available_districts))
        soiltype = st.selectbox("Soil Type", sorted(available_soiltypes))
        season = st.selectbox("Season", sorted(available_seasons))
    with col2:
        avgrainfall_mm = st.number_input("Average Rainfall (mm)", min_value=0.0, step=1.0)
        avgtemp_c = st.number_input("Average Temperature (°C)", min_value=0.0, step=0.1)
        avghumidity = st.number_input("Average Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    with col3:
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, step=1.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("🔍 Predict Crop")

# -------------------------------
# Prediction Logic
# -------------------------------
if submitted:
    try:
        # Create user input DataFrame
        user_data = pd.DataFrame([{
            "district": district,
            "soiltype": soiltype,
            "season": season,
            "avgrainfall_mm": avgrainfall_mm,
            "avgtemp_c": avgtemp_c,
            "avghumidity_%": avghumidity,
            "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen,
            "phosphorus_kg_ha": phosphorus,
            "potassium_kg_ha": potassium
        }])

        # One-hot encode
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)

        # Align columns with training data
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(user_data)[0]

        # Display result
        st.success(f"✅ **Recommended Crop:** {prediction}")

        with st.expander("📋 View Input Summary"):
            st.json({
                "District": district,
                "Soil Type": soiltype,
                "Season": season,
                "Average Rainfall (mm)": avgrainfall_mm,
                "Temperature (°C)": avgtemp_c,
                "Humidity (%)": avghumidity,
                "Soil pH": soil_ph,
                "Nitrogen (kg/ha)": nitrogen,
                "Phosphorus (kg/ha)": phosphorus,
                "Potassium (kg/ha)": potassium
            })

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
