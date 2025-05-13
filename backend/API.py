from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Form, status
from typing import List, Optional
import uuid
import os
from datetime import datetime
from pydantic import BaseModel
from models import ApiKeyCreate, ApiKeyInfo
import psycopg2
from psycopg2 import sql, errors
from BD import DB_CONFIG, add_request, get_request, get_user_info, register_user
import sys 
from pathlib import Path

root_path = Path(__file__).parent.parent  # Путь к handwriting-identification/
sys.path.append(str(root_path))

from personalized_writer_id import identify_with_calibration, register_person
from utils import validate_api_key
import uvicorn

# Конфигурация
API_KEYS_DB = "api_keys.db"
UPLOAD_DIR = "uploads"
REGISTER_DIR = os.path.join(UPLOAD_DIR, "register")  # Для базы фотографий
VERIFY_DIR = os.path.join(UPLOAD_DIR, "verify")      # Для проверочных фото

router = APIRouter(prefix="/v1", tags=["API Keys"])

@router.post("/generate", response_model=ApiKeyInfo)
async def generate_api_key(key_data: ApiKeyCreate):
    new_key = str(uuid.uuid4())
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                query = sql.SQL("""
                    INSERT INTO api_keys (key, owner, created_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    RETURNING created_at
                """)
                
                cursor.execute(query, (new_key, key_data.owner))
                created_at = cursor.fetchone()[0]
                conn.commit()
                
    except errors.IntegrityError as e:
        # Ошибка уникальности ключа (крайне маловероятно для UUID)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Key generation conflict, try again"
        )
        
    except errors.DatabaseError as e:
        # Общие ошибки базы данных
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed"
        )
        
    except Exception as e:
        # Все остальные исключения
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    
    return {
        "key": new_key,
        "owner": key_data.owner,
        "created_at": created_at,
        "is_active": True
    }

@router.delete("/delete/{key}")
async def revoke_api_key(key: str):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:

                cursor.execute("""
                    UPDATE api_keys
                    SET is_active = FALSE
                    WHERE key = %s
                """, (key,))
                
                if cursor.rowcount == 0:
                    conn.close()
                    raise HTTPException(404, "Key not found")
                
                conn.commit()
                return {"message": "Key revoked"}
        
    except errors.DatabaseError as e:
        # Общие ошибки базы данных
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed"
        )
        
    except Exception as e:
        # Все остальные исключения
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    

@router.post("/verify")
async def external_verify_handwriting(
    file: UploadFile = File(...),
    api_key: str = Depends(validate_api_key)
):
    # Оригинальная логика из /verify
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Поддерживаются только изображения")

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(VERIFY_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    id_user, confidence = identify(file_path)
    confidence = round(float(confidence), 2)
    user_info = get_user_info(id_user)
    print
    if not user_info:
        return {
            'status_code': 404,
            'content' :{
                "message": "User not found"
                }
        }
    
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
                "user_info": [user_info],
                "score": float(confidence),
            }
        ]
    }

@router.post("/users", status_code=201)
async def external_create_user(
    full_name: str = Form(...),
    user_id: str = Form(...),
    samples: List[UploadFile] = File(...),
    notes: str = Form(""),
    api_key: str = Depends(validate_api_key)
):
    saved_paths = []
    for i, photo in enumerate(samples):
        if not photo.content_type.startswith("image/"):
            raise HTTPException(400, "Файл должен быть изображением")

        ext = photo.filename.split('.')[-1]
        filename = f"{user_id}_{i+1}.{ext}"
        file_path = os.path.join(REGISTER_DIR, filename)
        
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

@router.get("/results/{request_id}")
async def external_get_verification_results(
    request_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        result = get_request(request_id)
        if not result:
            raise HTTPException(404, "Request not found")
        
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
            "completedAt": result.get("completed_at")
        }
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    