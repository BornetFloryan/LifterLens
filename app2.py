import streamlit as st
import requests
from model import predict_lifts
import os
from dotenv import load_dotenv
import json

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "open-mistral-nemo"

CSV_PATH = "database/openpowerlifting-2024-01-06-4c732975.csv"

st.set_page_config(page_title="Powerlifting AI Coach", layout="wide")

st.title("Coach IA Powerlifting")

st.write("Entrez vos informations et recevez un **programme complet généré par Mistral**, "
         "basé sur un modèle XGBoost entraîné sur OpenPowerlifting.")

valeur = st.text_input("Exprimez ce que vous voulez, en précisant votre sexe, âge et poids de corps ainsi que le nombre de jours")

api_key = MISTRAL_API_KEY


if st.button("Générer mon programme complet"):
    if not api_key:
        st.error("Vous devez fournir une clé API Mistral.")
        st.stop()

    with st.spinner("Calcul des prédictions + envoi à Mistral..."):

        prompt = f"""
        {valeur}
        Renvoie uniquement un JSON brut contenant :
        - sex
        - age
        - bodyweight
        - nbrDays
        Exemple :
        {{"sex": "M", "age": 20, "bodyweight": 75, "nbrDays": 5}}
        Aucun texte autour, seulement le JSON.
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

        raw_json = output["choices"][0]["message"]["content"]

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            st.error("Erreur : Mistral n'a pas renvoyé un JSON valide.")
            st.write("Contenu reçu :", raw_json)
            st.stop()

        preds = predict_lifts(
            data["sex"],
            data["age"],
            data["bodyweight"]
        )

        prompt = f"""
Tu es un coach expert en powerlifting.
Voici le profil de l'athlète :

- Sexe : {data["sex"]}
- Âge : {data["age"]}
- Poids : {data["bodyweight"]} kg
- Nombre de jours : {data["nbrDays"]}

Voici ses performances estimées selon un modèle XGBoost basé sur OpenPowerlifting :
- Squat estimé : {preds["Best3SquatKg"]:.1f} kg
- Bench estimé : {preds["Best3BenchKg"]:.1f} kg
- Deadlift estimé : {preds["Best3DeadliftKg"]:.1f} kg

Génère un **programme complet**, clair, structuré pour la durée de {data["nbrDays"]} jours :
- exercices
- séries
- répétitions
- intensité (% ou RPE)
- progression jours par jours
- conseils techniques

Ne renvoie que le programme, propre, lisible, sans texte superflu.
"""

        payload = {
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }

        res = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers)
        output = res.json()


        st.subheader("Programme généré par Mistral :")
        st.write(output["choices"][0]["message"]["content"])
