import streamlit as st
import requests
from model import predict_lifts
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "open-mistral-nemo"

CSV_PATH = "database/openpowerlifting-2024-01-06-4c732975.csv"

st.set_page_config(page_title="Powerlifting AI Coach", layout="wide")

st.title("Coach IA Powerlifting")

st.write("Entrez vos informations et recevez un **programme complet généré par Mistral**, "
         "basé sur un modèle XGBoost entraîné sur OpenPowerlifting.")

col1, col2, col3 = st.columns(3)

sex = col1.selectbox("Sexe :", ["M", "F"])
age = col2.number_input("Âge :", value=20)
bodyweight = col3.number_input("Poids (kg) :", value=75.0)

api_key = MISTRAL_API_KEY


if st.button("Générer mon programme complet"):
    if not api_key:
        st.error("Vous devez fournir une clé API Mistral.")
        st.stop()

    with st.spinner("Calcul des prédictions + envoi à Mistral..."):

        preds = predict_lifts(sex, age, bodyweight)

        prompt = f"""
Tu es un coach expert en powerlifting.
Voici le profil de l'athlète :

- Sexe : {sex}
- Âge : {age}
- Poids : {bodyweight} kg

Voici ses performances estimées selon un modèle XGBoost basé sur 3M lignes OpenPowerlifting :
- Squat estimé : {preds["Best3SquatKg"]:.1f} kg
- Bench estimé : {preds["Best3BenchKg"]:.1f} kg
- Deadlift estimé : {preds["Best3DeadliftKg"]:.1f} kg

Fais un **programme complet**, clair, structuré (3-5 jours/semaine) :
- exercices
- séries
- répétitions
- intensité (% ou RPE)
- progression semaine par semaine
- conseils techniques

Ne renvoie PAS de texte superflu, juste un programme propre et détaillé.
"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }

        res = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers)
        output = res.json()

        st.subheader("Programme généré par Mistral :")
        st.write(output["choices"][0]["message"]["content"])