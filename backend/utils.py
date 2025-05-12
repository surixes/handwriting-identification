from datetime import datetime
import os
import psycopg2
from fastapi import HTTPException, Header, status
from BD import DB_CONFIG
from models import FeedbackRequest
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

async def validate_api_key(api_key: str = Header(..., alias="X-API-Key")):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:

                cursor.execute("""
                    SELECT key, is_active 
                    FROM api_keys 
                    WHERE key = %s
                """, (api_key,))
                
                result = cursor.fetchone()

                if not result or not result[1]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid or inactive API key"
                    )
                return api_key
                
    except psycopg2.OperationalError as e:
        # Ошибки подключения к БД
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection error"
        )
        
    except psycopg2.DatabaseError as e:
        # Общие ошибки базы данных
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )
        
    except Exception as e:
        if "Invalid or inactive API key" in str(e):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or inactive API key"
                )
        else:
            # Непредвиденные ошибки
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )

def send_email(feedback: FeedbackRequest):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")

    # Создание HTML-сообщения
    msg = MIMEMultipart('alternative')
    msg["From"] = smtp_username
    msg["To"] = recipient_email
    msg["Subject"] = f"📬 Новое сообщение: {feedback.subject}"

    # HTML-шаблон с CSS-стилями
    html = f"""
    <html>
      <head>
        <style>
          .container {{ 
            max-width: 600px;
            margin: 20px auto;
            padding: 30px;
            border: 1px solid #eee;
            border-radius: 10px;
            font-family: Arial, sans-serif;
          }}
          .header {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
          }}
          .content {{
            margin: 20px 0;
            line-height: 1.6;
          }}
          .highlight {{
            color: #3498db;
            font-weight: bold;
          }}
          .message-box {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
          }}
          .footer {{
            margin-top: 20px;
            font-size: 0.9em;
            color: #7f8c8d;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1 class="header">✉️ Новое сообщение с сайта</h1>
          
          <div class="content">
            <p><span class="highlight">От:</span> {feedback.name} ({feedback.email})</p>
            <p><span class="highlight">Тема:</span> {feedback.subject}</p>
            
            <div class="message-box">
              <h3>📝 Сообщение:</h3>
              <pre style="white-space: pre-wrap;">{feedback.message}</pre>
            </div>
          </div>
          
          <div class="footer">
            <p>Дата отправки: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
          </div>
        </div>
      </body>
    </html>
    """
    body = f"""
        Новое сообщение от {feedback.name} ({feedback.email}):

        Тема: {feedback.subject}
        Сообщение:
        {feedback.message}

        Согласие на обработку: {'да' if feedback.consent else 'нет'}
        Дата отправки: {datetime.now().strftime("%d.%m.%Y %H:%M")}
        """


    # Прикрепляем HTML и plain-text версии
    part_text = MIMEText(body, "plain")  # Можно сохранить оригинальный текст
    part_html = MIMEText(html, "html")
    
    msg.attach(part_text)
    msg.attach(part_html)

    # Отправка через SMTP
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, recipient_email, msg.as_string())