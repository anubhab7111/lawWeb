"""
Thin SMTP email sender. If SMTP_HOST is unset (the local-dev default), sends
are logged instead of attempted — so notification_dispatch.py works
end-to-end without real credentials.
"""

import smtplib
from email.message import EmailMessage

from app.config import get_settings


def send_email(to_address: str, subject: str, body: str) -> bool:
    settings = get_settings()
    if not settings.smtp_host:
        print(f"[EmailClient] SMTP not configured — would send to {to_address}: {subject}")
        return False

    message = EmailMessage()
    message["From"] = settings.smtp_from_address
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            server.starttls()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"[EmailClient] send failed for {to_address}: {e}")
        return False
