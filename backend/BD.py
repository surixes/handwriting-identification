import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
import uuid
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),  # Конвертируем в число
    "client_encoding": os.getenv("DB_CLIENT_ENCODING", "utf8")
}

def create_tables():
    """Создание таблиц в базе данных"""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    full_name TEXT NOT NULL,
                    user_id TEXT UNIQUE NOT NULL,
                    notes TEXT,
                    photos JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    confidence REAL,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            conn.commit()

def register_user(user_data):
    """
    Регистрация нового пользователя
    :param user_data: словарь с данными пользователя
    :return: ID зарегистрированного пользователя
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                query = sql.SQL("""
                    INSERT INTO users (full_name, user_id, notes, photos)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """)
                cursor.execute(query, (
                    user_data['full_name'],
                    user_data['user_id'],
                    user_data.get('notes', ''),
                    Json(user_data['photos'])
                ))
                user_uuid = cursor.fetchone()[0]
                conn.commit()
                return user_uuid
    except psycopg2.IntegrityError as e:
        raise ValueError("User ID already exists") from e

def add_request(request_data):
    """
    Регистрация нового пользователя
    :param user_data: словарь с данными пользователя
    :return: ID зарегистрированного пользователя
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                query = sql.SQL("""
                    INSERT INTO requests (request_id, user_id, full_name, confidence)
                    VALUES (%s, %s, %s, %s);
                """)
                cursor.execute(query, (
                    request_data['request_id'],
                    request_data['user_id'],
                    request_data['full_name'],
                    request_data['confidence']
                ))
                conn.commit()
                return "OK"
    except psycopg2.IntegrityError as e:
        raise ValueError("User ID already exists")

def get_user_info(user_id):
    """
    Получение информации о пользователе по ID
    :param user_id: идентификатор пользователя
    :return: словарь с данными пользователя или None
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT full_name, notes 
                FROM users 
                WHERE user_id = %s;
            """, (user_id,))
            result = cursor.fetchone()
            if result:
                return {
                    'full_name': result[0],
                    'notes': result[1]
                }
            return None

def get_request(request_id):
    """
    Получение информации о пользователе по ID
    :param user_id: идентификатор пользователя
    :return: словарь с данными пользователя или None
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, full_name, confidence, completed_at
                FROM requests 
                WHERE request_id = %s;
            """, (request_id,))
            result = cursor.fetchone()
            if result:
                return {
                    'user_id': result[0],
                    'full_name': result[1],
                    'confidence': result[2],
                    'completed_at': result[3]
                }
            return None
        
create_tables()