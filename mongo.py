
from pymongo import MongoClient
import logging
from datetime import datetime

logger = logging.getLogger("MongoDB")


class MongoDB:
    def __init__(self):
        try:
            self.client = MongoClient(
                "mongodb://localhost:27017",
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client.contentguard
            self.moderation_collection = self.db.contents
            self.generation_collection = self.db.generated_images

            # Test de connexion
            self.client.admin.command('ping')
            logger.info("Connexion MongoDB réussie – Base: contentguard")
            logger.info("Collections disponibles : contents, generated_images")
        except Exception as e:
            logger.error(f"Échec connexion MongoDB : {e}")
            raise

    def save_moderation_result(self, data: dict):
        """Sauvegarde un résultat de modération d'image uploadée"""
        doc = {
            "filename": data["filename"],
            "is_fake": data["is_fake"],
            "confidence": data["confidence"],
            "prob_real": data["prob_real"],
            "prob_fake": data["prob_fake"],
            "status": data["status"],
            "timestamp": data["timestamp"]
        }
        try:
            result = self.moderation_collection.insert_one(doc)
            logger.info(f"Modération sauvegardée → _id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Erreur sauvegarde modération : {e}")
            raise

    def save_generation_result(self, task_id: str, prompt: str, filename: str):
        """
        Sauvegarde chaque image générée par Stable Diffusion
        Collection : generated_images
        """
        doc = {
            "task_id": task_id,
            "prompt": prompt.strip(),
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        }
        try:
            result = self.generation_collection.insert_one(doc)
            logger.info(f"Génération sauvegardée dans MongoDB → _id: {result.inserted_id} | {filename}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde génération dans MongoDB : {e}")

    def get_stats(self):
        """Retourne les statistiques pour le dashboard"""
        try:
            total = self.moderation_collection.count_documents({})
            approved = self.moderation_collection.count_documents({"status": "approved"})
            rejected = self.moderation_collection.count_documents({"status": "rejected"})
            generated = self.generation_collection.count_documents({})
            return {
                "total": int(total),
                "approved": int(approved),
                "rejected": int(rejected),
                "generated": int(generated)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats : {e}")
            return {"total": 0, "approved": 0, "rejected": 0, "generated": 0}


mongodb = MongoDB()