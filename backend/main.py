from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uvicorn
import uuid
from datetime import datetime
from typing import List
from models import VerificationResult, UserCreate
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent  # Путь к handwriting-identification/
sys.path.append(str(root_path))

from personalized_writer_id import register_person, identify
from BD import get_user_info, register_user, add_request, get_request, create_tables

app = FastAPI(title="ScriptVerify API", version="1.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_DIR = "uploads"
REGISTER_DIR = os.path.join(UPLOAD_DIR, "register")  # Для базы фотографий
VERIFY_DIR = os.path.join(UPLOAD_DIR, "verify")      # Для проверочных фото

# Создание всех необходимых директорий при старте
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.post("/verify")
async def verify_handwriting(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Поддерживаются только изображения")
    
    # Сохраняем файл
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(VERIFY_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    id_user, confidence = identify(file_path)
    confidence = round(float(confidence), 2)
    user_info = get_user_info(id_user)

    if not user_info:
        return JSONResponse(
            status_code=404,
            content={"message": "User not found"}
        )
    
    req_id = f"req_{str(uuid.uuid4())}"
    
    requests_data = {
        "user_id": id_user,
        "request_id": req_id,
        "full_name": user_info["full_name"],
        "confidence": confidence

    }
    try:
        add_request(requests_data)
    except Exception as e:
        return str(e)
    return {
        "success": True,
        "requestId": req_id,
        "result": [
            {
                "user_id": id_user,
                "user_info": [
                    user_info
                ],
                "score": float(confidence),
            }
        ]
    }

@app.post("/users", status_code=201)
async def create_user(
    full_name: str = Form(...),
    user_id: str = Form(...),
    samples: List[UploadFile] = File(...),
    notes: str = Form("")
):
    
    saved_paths = []
    for i, photo in enumerate(samples):
        if not photo.content_type.startswith("image/"):
            raise HTTPException(400, "Файл должен быть изображением")

        # Генерация имени файла
        ext = photo.filename.split('.')[-1]
        filename = f"{user_id}_{i+1}.{ext}"
        file_path = os.path.join(REGISTER_DIR, filename)
        
        # Сохранение файла 
        with open(file_path, "wb") as buffer:
            buffer.write(await photo.read())
        
        saved_paths.append(f"{file_path}")

    new_user = {
        "full_name": full_name,
        "user_id": user_id,
        "notes": notes,
        "photos": saved_paths
    }

    register_person(user_id, saved_paths)

    try:
        register_user(new_user)
    except Exception as e:
        return str(e)
    return {
        "success": True,
        "user_id": user_id,
        "samplesProcessed": len(samples)
    }

@app.get("/results/{request_id}")
async def get_verification_results(request_id: str):
    """
    Получение результатов верификации по ID запроса
    
    - **request_id**: Уникальный идентификатор запроса
    - Возвращает статус и результаты верификации
    """
    try:
        result = get_request(request_id)  # Предполагаем, что функция переименована
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Request not found"
            )
        
        return {
            "success": True,
            "status": 'completed',
            "results": [
                {
                    "userId": result['user_id'],
                    "name": result["full_name"],
                    "confidence": result['confidence']
                }
            ],
            "completedAt": result.get("completed_at")  # Используем get для избежания KeyError
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)