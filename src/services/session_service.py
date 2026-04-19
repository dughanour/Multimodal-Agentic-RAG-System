from src.models.db_connection import SessionLocal
from src.models.chat_models import ChatSession, ChatMessage
from datetime import datetime, timezone


class SessionService:
    """Handles all database operations for chat sessions and messages."""

    def _get_db(self):
        """Get a new database session."""
        return SessionLocal()
    
    def create_session(self, title: str="New Chat")-> dict:
        """Create a new chat session."""
        db = self._get_db()
        try:
            session = ChatSession(title=title)
            db.add(session)
            db.commit()
            db.refresh(session)
            return {
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at,
            }
        finally:
            db.close()
    
    def list_sessions(self)-> list:
        """List all sessions, newest first."""
        db = self._get_db()
        try:
            sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            return [
                {
                    "id": str(session.id),
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat()
                }
                for session in sessions
            ]
        finally:
            db.close()
    
    def get_messages(self, session_id: str)-> list:
        """Get all messages for a session, ordered by time."""
        db = self._get_db()
        try:
            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            return [
                {
                    "id": str(message.id),
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at.isoformat()
                }
                for message in messages
            ]
        finally:
            db.close()
    
    def add_message(self, session_id: str, role: str, content: str)-> dict:
        """Save a new message and update the session's updated_at time."""
        db = self._get_db()
        try:
            message = ChatMessage(session_id=session_id, role=role, content=content)
            db.add(message)

            # Update the session's updated_at so it moves to top of sidebar
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(message)
            return {
                "id": str(message.id),
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat()
            }
        finally:
            db.close()
    
    def delete_session(self, session_id: str)-> bool:
        """Delete a session and all its messages."""
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return False
            
            db.delete(session)
            db.commit()
            return True
        finally:
            db.close()
    
    def update_session_title(self, session_id: str, title: str)-> bool:
        """Update the title of a session."""
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return False
            
            session.title = title
            db.commit()
            return True
        finally:
            db.close()
            
    
    # Singleton instance
session_service = SessionService()

def get_session_service():
    """Dependency injector for FastAPI."""
    return session_service