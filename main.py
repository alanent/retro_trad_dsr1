from fastapi import FastAPI
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import time
import re
import json
from azure.storage.blob import BlobServiceClient
import logging
import os
from dotenv import load_dotenv
import os
import io
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import re
load_dotenv()

app = FastAPI()

# Configuration explicite du logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




@app.on_event("startup")
async def startup_event():

    global client
    client = ChatCompletionsClient(endpoint=os.getenv("DSR1_ENDPOINT"), credential=AzureKeyCredential(os.getenv("DSR1_KEY")))

    def predict(br, max_tries=2):

        tries = 0
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
                    model='DeepSeek-R1-soxol'
                )

                result = response.choices[0].message.content
                # Retirer les parties <think>...</think> si présentes
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                result = result.strip()
                print(result)

                # Vérifier que le résultat est un JSON valide
                try:
                    json_result = json.loads(result)
                    # S'assurer que le champ 'translation' est présent
                    if "translation" in json_result:
                        return json_result
                    else:
                        print("Le JSON retourné ne contient pas le champ 'translation'.")
                except json.JSONDecodeError:
                    print("Erreur de décodage JSON.")

            except Exception as e:
                print(f"Erreur lors de la traduction : {e}")

            tries += 1
            print(f"Tentative {tries} échouée, attente de 2 secondes avant de réessayer...")
            time.sleep(2)

        # Si toutes les tentatives échouent, renvoyer un JSON indiquant l'échec
        return {"translation": None}



    try:
        logger.info("Démarrage de l'application...")
        # Connexion à Azure Blob
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("STORAGE_CONNECTION_STRING"))
        blob_client = blob_service_client.get_blob_client(container="data", blob="troer-dataset-firebase-adminsdk-fbsvc-58d8f446f7.json")
        
        # Téléchargement du fichier JSON
        json_data = blob_client.download_blob().readall()
        data = json.loads(json_data)
        
        # Initialisation de Firebase
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)
        
        # Connexion à Firestore
        db = firestore.client()
        
        logger.info("Connexion à Firebase réussie et à Firestore établie.")
    except Exception as e:
        logger.error(f"Erreur de connexion à Firebase : {e}")

    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("STORAGE_CONNECTION_STRING"))
    blob_client = blob_service_client.get_blob_client(container="data", blob="br_mono.csv")
    # Téléchargement du fichier et conversion en fichier-like avec io.BytesIO
    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(blob_data))  # Utilisation de BytesIO pour convertir en fichier-like
    logger.info(df.head())




    # Traitement ligne par ligne du DataFrame 
    for index, row in df.iterrows():
        # For Serverless API or Managed Compute endpoints
        
        br_text = row['br']  # Assurez-vous que la colonne 'br' existe dans votre CSV
        
        # Appel à la fonction de prédiction pour traduire le texte
        result = predict(br_text)
        
        if result.get("translation"):
            fr_translation = result["translation"]
            
            # Enregistrement dans la collection 'to_validate'
            try:
                to_validate_ref = db.collection("to_validate").document()
                to_validate_ref.set({
                    'br': br_text,
                    'fr': fr_translation,
                    'source': "dsr1",
                    'timestamp': firestore.SERVER_TIMESTAMP  # Vous pouvez ajouter un timestamp pour suivi
                })
                
                logger.info(fr_translation)

                # Mise à jour du compteur dans la collection 'stats'
                stats_ref = db.collection("stats").document("global")
                stats_ref.update({
                    'to_validate': firestore.Increment(1)  # Incrémente le compteur
                })
                
                logger.info(f"Traduction ajoutée à la collection 'to_validate' pour l'entrée {index}.")
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout des traductions dans Firestore : {e}")
        else:
            logger.error(f"Aucune traduction valide trouvée pour l'entrée {index}.")