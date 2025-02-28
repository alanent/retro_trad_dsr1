import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import time
import re
import json
import os
import logging
import io
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from flask import Flask
import threading

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Initialisation du client Azure pour la traduction
client = ChatCompletionsClient(
    endpoint=os.getenv("DSR1_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("DSR1_KEY"))
)

def predict(br, max_tries=3):
    """Effectue la traduction du breton vers le français via Azure AI avec max 3 tentatives."""
    tries = 0
    logger.info("Tentative de traduction via DeepSeek-R1...")
    while tries < max_tries:
        try:
            response = client.complete(
                messages=[
                    SystemMessage(content=""" 
                        You are an advanced machine translation system specializing in Breton-to-French translation.
                        Your task is to accurately translate Breton text into fluent and natural French while preserving the original meaning, nuances, and cultural context.
                        Ensure that the translations are grammatically correct and stylistically appropriate for the given text.
                        Translate only this text from Breton to French and return the response in JSON format as follows: {"translation": ""}.
                        No explanations, no extra words—only the JSON response.
                    """),
                    UserMessage(content=br)
                ],
                max_tokens=2048,
                model='DeepSeek-R1-aakkp'
            )

            result = response.choices[0].message.content
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            try:
                json_result = json.loads(result)
                if "translation" in json_result:
                    return json_result
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON.")
        except Exception as e:
            logger.error(f"Erreur lors de la traduction (tentative {tries+1}/{max_tries}): {e}")
            time.sleep(10)
        tries += 1
    return {"translation": "api_error"}

def main():
    try:
        logger.info("Démarrage du traitement...")

        # Connexion à Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("STORAGE_CONNECTION_STRING"))

        # Téléchargement des credentials Firebase depuis Azure Blob
        blob_client = blob_service_client.get_blob_client(container="data", blob="troer-dataset-firebase-adminsdk-fbsvc-58d8f446f7.json")
        json_data = blob_client.download_blob().readall()
        data = json.loads(json_data)

        # Initialisation de Firebase
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Connexion à Firebase réussie.")

        # Téléchargement du fichier CSV contenant les données à traduire
        blob_client = blob_service_client.get_blob_client(container="data", blob="br_mono.csv")
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(blob_data))  # Chargement du CSV dans un DataFrame
        logger.info(f"Données chargées : {df.head()}")

        # Traitement des traductions
        for index, row in df.iterrows():
            br_text = row['br']  # Colonne contenant le texte breton

            # Vérifier si la phrase est déjà enregistrée dans Firestore
            existing_docs = db.collection("to_validate").where("br", "==", br_text).stream()
            if any(existing_docs):
                logger.info(f"Les données existent déjà : {br_text}")
                continue

            # Traduction
            result = predict(br_text)
            if result.get("translation") and result.get("translation") != "api_error":
                fr_translation = result["translation"]

                # Enregistrement dans Firestore
                try:
                    to_validate_ref = db.collection("to_validate").document()
                    to_validate_ref.set({
                        'br': br_text,
                        'fr': fr_translation,
                        'source': "dsr1",
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    logger.info(f"Traduction ajoutée : {fr_translation}")

                    # Mise à jour du compteur dans Firestore
                    stats_ref = db.collection("stats").document("global")
                    stats_ref.update({'to_validate': firestore.Increment(1)})
                except Exception as e:
                    logger.error(f"Erreur Firestore : {e}")
            else:
                logger.error(f"Échec de traduction pour l'entrée {index} après 3 tentatives.")
                # Ajouter à la collection to_retry
                try:
                    to_retry_ref = db.collection("to_retry").document()
                    to_retry_ref.set({
                        'br': br_text,
                        'attempts': 3,
                        'last_error': "api_error" if result.get("translation") == "api_error" else "Traduction invalide",
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    logger.info(f"Entrée ajoutée à to_retry : {br_text}")
                    
                    # Mise à jour du compteur to_retry dans Firestore (si nécessaire)
                    stats_ref = db.collection("stats").document("global")
                    stats_ref.update({'to_retry': firestore.Increment(1)})
                except Exception as e:
                    logger.error(f"Erreur lors de l'ajout à to_retry : {e}")

        logger.info("Traitement terminé.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script : {e}")

# Création de l'application Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Service en cours d'exécution", 200

def start_background_task():
    # Lancer le traitement dans un thread séparé
    processing_thread = threading.Thread(target=main)
    processing_thread.start()

if __name__ == "__main__":
    # Démarrer la tâche de traitement en arrière-plan
    start_background_task()
    # Démarrer le serveur Flask sur le port défini ou 8000 par défaut
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)