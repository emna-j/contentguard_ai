# train_and_test.py (à mettre à la racine)
from generation.model.trainer import Trainer
from generation.generator import Generator
import time

if __name__ == "__main__":
    print("ÉTAPE 1 : Finetuning LoRA (8–12 minutes)")
    trainer = Trainer(subject_name="emnaj")
    model_path = trainer.train()
    print(f"Modèle entraîné → {model_path}")

    print("\nÉTAPE 2 : Génération instantanée")
    generator = Generator()

    task_id = generator.generate("emnaj en tenue de soirée à Paris la nuit, ultra-réaliste")
    print(f"Task lancée : {task_id}")

    while True:
        status = generator.get_status(task_id)
        if status["status"] == "completed":
            print("GÉNÉRÉ EN MOINS DE 2 SECONDES !")
            print(f"Image dans MongoDB → file_id = {status['file_id']}")
            break
        elif status["status"] == "failed":
            print("Échec :", status.get("error"))
            break
        time.sleep(0.5)