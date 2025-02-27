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

def predict(br, max_tries=10):
    """ Effectue la traduction du breton vers le français via Azure AI. """
    tries = 0
    logger.info("trying request deepseekr1...")
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
            logger.error(f"Erreur lors de la traduction : {e}")

        tries += 1
        time.sleep(60)

    return {"translation": None}

def main():
    try:
        logger.info("Démarrage du script...")

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

            # Traduction
            result = predict(br_text)

            if result.get("translation"):
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
                    logger.info(f"Traduction ajoutée: {fr_translation}")

                    # Mise à jour du compteur dans Firestore
                    stats_ref = db.collection("stats").document("global")
                    stats_ref.update({'to_validate': firestore.Increment(1)})
                except Exception as e:
                    logger.error(f"Erreur Firestore : {e}")
            else:
                logger.error(f"Aucune traduction valide pour l'entrée {index}.")

        logger.info("Traitement terminé.")

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script : {e}")

if __name__ == "__main__":
    main()
