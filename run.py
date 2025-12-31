import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
import os
from datetime import datetime
import threading
import subprocess

from database.mongo import mongodb

app = FastAPI(title="ContentGuard AI")


@app.on_event("startup")
async def startup():
    await mongodb.connect()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Simulation modération
    await asyncio.sleep(0.8)
    is_bad = any(x in file.filename.lower() for x in ["deepfake", "hate", "violence"])
    status = "rejected" if is_bad else "approved"

    await mongodb.save_result({
        "filename": file.filename,
        "status": status,
        "deepfake": is_bad,
        "uploaded_at": datetime.now().isoformat()
    })

    return {"status": status, "message": "Deepfake détecté !" if is_bad else "Contenu sûr"}


if __name__ == "__main__":
    # Thread 1 : API
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": 8000}, daemon=True).start()

    # Thread 2 : Dashboard Streamlit
    subprocess.Popen(["streamlit", "run", "frontend/dashboard.py", "--server.port=8501"])

    print("ContentGuard AI lancé !")
    print("→ API : http://localhost:8000/docs")
    print("→ Dashboard : http://localhost:8501")
    print("→ Ray : http://127.0.0.1:8265")

    import time

    while True:
        time.sleep(1)