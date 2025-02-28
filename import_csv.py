import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import time
import json
import os
import logging
import io
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from flask import Flask
import threading
import argparse

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

def import_csv_to_firebase(csv_path, source_name, is_blob=False):
    """
    Importe des traductions depuis un fichier CSV vers Firebase.
    
    Args:
        csv_path (str): Chemin vers le fichier CSV ou nom du blob dans Azure Storage
        source_name (str): Nom de la source à enregistrer dans Firestore
        is_blob (bool): Si True, charge le CSV depuis Azure Blob Storage, sinon depuis le système de fichiers local
    """
    try:
        logger.info(f"Début de l'importation des traductions depuis {csv_path}...")
        
        # Chargement du CSV
        if is_blob:
            # Connexion à Azure Blob Storage
            blob_service_client = BlobServiceClient.from_connection_string(os.getenv("STORAGE_CONNECTION_STRING"))
            
            # Téléchargement des credentials Firebase depuis Azure Blob
            cred_blob_client = blob_service_client.get_blob_client(container="data", blob="troer-dataset-firebase-adminsdk-fbsvc-58d8f446f7.json")
            json_data = cred_blob_client.download_blob().readall()
            cred_data = json.loads(json_data)
            
            # Téléchargement du fichier CSV depuis Azure Blob
            csv_blob_client = blob_service_client.get_blob_client(container="data", blob=csv_path)
            csv_data = csv_blob_client.download_blob().readall()
            df = pd.read_csv(io.BytesIO(csv_data))
        else:
            # Chargement du fichier CSV local
            df = pd.read_csv(csv_path)
            
            # Chargement des credentials Firebase depuis le fichier local
            cred_file_path = os.getenv("FIREBASE_CREDENTIALS", "troer-dataset-firebase-adminsdk-fbsvc-58d8f446f7.json")
            with open(cred_file_path, 'r') as f:
                cred_data = json.load(f)
        
        # S'assurer que le DataFrame a les colonnes requises
        required_columns = ['br', 'fr']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Le CSV doit contenir les colonnes {required_columns}")
            return
        
        logger.info(f"Données chargées : {len(df)} lignes")
        
        # Initialisation de Firebase
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(cred_data)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        logger.info("Connexion à Firebase réussie.")
        
        # Compteurs pour le suivi
        total_count = len(df)
        added_count = 0
        existing_count = 0
        error_count = 0
        
        # Traitement des traductions
        batch_size = 500
        current_batch = 0
        
        while current_batch * batch_size < total_count:
            # Créer un lot pour les opérations Firestore (maximum 500 opérations par lot)
            batch = db.batch()
            batch_docs = 0
            
            start_idx = current_batch * batch_size
            end_idx = min((current_batch + 1) * batch_size, total_count)
            
            for index in range(start_idx, end_idx):
                row = df.iloc[index]
                br_text = row['br']
                fr_text = row['fr']
                
                # Vérifier si la phrase est déjà enregistrée dans Firestore
                existing_docs = list(db.collection("to_validate").where("br", "==", br_text).limit(1).stream())
                
                if existing_docs:
                    existing_count += 1
                    if index % 100 == 0:
                        logger.info(f"Progression: {index}/{total_count}, existants: {existing_count}")
                    continue
                
                try:
                    # Créer un document dans le lot
                    doc_ref = db.collection("to_validate").document()
                    batch.set(doc_ref, {
                        'br': br_text,
                        'fr': fr_text,
                        'source': source_name,
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    batch_docs += 1
                    added_count += 1
                except Exception as e:
                    logger.error(f"Erreur lors de la préparation du document {index}: {e}")
                    error_count += 1
            
            # Exécuter le lot si des documents ont été ajoutés
            if batch_docs > 0:
                try:
                    batch.commit()
                    logger.info(f"Lot {current_batch+1} commité: {batch_docs} documents")
                    
                    # Mise à jour du compteur global dans Firestore
                    stats_ref = db.collection("stats").document("global")
                    stats_ref.update({'to_validate': firestore.Increment(batch_docs)})
                except Exception as e:
                    logger.error(f"Erreur lors du commit du lot {current_batch+1}: {e}")
                    error_count += batch_docs
                    added_count -= batch_docs
            
            current_batch += 1
            
        logger.info(f"Importation terminée. Total: {total_count}, Ajoutés: {added_count}, Existants: {existing_count}, Erreurs: {error_count}")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")

# Création de l'application Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Service d'importation CSV vers Firebase", 200

def start_background_task(csv_path, source_name, is_blob):
    # Lancer le traitement dans un thread séparé
    processing_thread = threading.Thread(target=import_csv_to_firebase, args=(csv_path, source_name, is_blob))
    processing_thread.start()

if __name__ == "__main__":
    # Configurer les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Importer des traductions depuis un CSV vers Firebase")
    parser.add_argument("--csv_path", required=True, help="Chemin vers le fichier CSV ou nom du blob")
    parser.add_argument("--source_name", required=True, help="Nom de la source à enregistrer")
    parser.add_argument("--is_blob", action="store_true", help="Si spécifié, charge le CSV depuis Azure Blob Storage")
    parser.add_argument("--web", action="store_true", help="Si spécifié, lance le serveur Flask")
    
    args = parser.parse_args()
    
    if args.web:
        # Démarrer la tâche de traitement en arrière-plan
        start_background_task(args.csv_path, args.source_name, args.is_blob)
        # Démarrer le serveur Flask sur le port défini ou 8000 par défaut
        port = int(os.environ.get("PORT", 8000))
        app.run(host="0.0.0.0", port=port)
    else:
        # Exécuter directement la fonction d'importation
        import_csv_to_firebase(args.csv_path, args.source_name, args.is_blob)