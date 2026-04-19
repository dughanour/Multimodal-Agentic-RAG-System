from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.services.session_service import get_session_service

router = APIRouter()

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"

class UpdateSessionRequest(BaseModel):
    title: str


@router.post("/sessions")
async def create_session(payload: CreateSessionRequest = CreateSessionRequest()):
    """Create a new chat session. Called when user clicks 'New Chat'."""
    try:
        service = get_session_service()
        session = service.create_session(payload.title)
        return session

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/sessions")
async def list_sessions():
    """List all chat sessions. Called when user clicks on the 'Sessions' sidebar."""
    try:
        service = get_session_service()
        sessions = service.list_sessions()
        return sessions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.get(("/sessions/{session_id}/messages"))
async def get_session_messages(session_id: str):
    """Get all messages for a specific session. Called when user clicks an old chat."""
    try:
        service = get_session_service()
        messages = service.get_messages(session_id)
        return messages
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    try:
        service = get_session_service()
        deleted  = service.delete_session(session_id)
        if not deleted :
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {"message": f"Session {session_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, payload: UpdateSessionRequest):
    """Update the title of a session."""
    try:
        service = get_session_service()
        updated = service.update_session_title(session_id, payload.title)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {"message": f"Session {session_id} updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")










