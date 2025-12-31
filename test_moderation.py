from moderator.moderation_engine import engine
import time
from pathlib import Path

print("Démarrage de la modération complète avec les 4 concurrences...\n")

engine.start_processing()

fake_path = Path("training/datasets/deepfake_mini/fake")
real_path = Path("training/datasets/deepfake_mini/real")

fake_img = next(fake_path.glob("*.jpg"), None) or next(fake_path.glob("*.png"))
real_img = next(real_path.glob("*.jpg"), None) or next(real_path.glob("*.png"))

if fake_img and real_img:
    engine.submit_image(str(fake_img), priority=1)
    engine.submit_image(str(real_img), priority=2)
    print(f"Images envoyées :\n  - {fake_img.name}\n  - {real_img.name}")
else:
    print("Attention : Aucune image trouvée dans fake/ ou real/")

print("\nAttente des résultats (15 secondes max)...\n")
time.sleep(15)

print("\nTEST TERMINÉ – Si tu as vu [RÉSULTAT] plus haut → TOUT FONCTIONNE PARFAITEMENT !")