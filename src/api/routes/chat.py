from fastapi import APIRouter, UploadFile, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from src.services.document_service import DocumentService, get_document_service
from src.services.chat_service import ChatService, get_chat_service
from src.agents.tools import configure_retriever_mode
from src.agents.supervisor import configure_supervisor
from src.services.session_service import get_session_service
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
import json
import asyncio 

router = APIRouter()

class SupervisorInstructions(BaseModel):
    instructions: str

class ScrapeRequest(BaseModel):
    url: str

@router.post("/supervisor/instructions", status_code=204)
async def set_supervisor_instructions(payload: SupervisorInstructions):
    """
    Sets the custom instructions for the supervisor agent.
    These instructions will be applied to all subsequent chat sessions.
    """
    try:
        configure_supervisor(payload.instructions)
        print(f"Supervisor instructions updated: {payload.instructions}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", summary="Upload and process a document")
async def upload_document(
    file: UploadFile,
    strategy: str = Query("docling", enum=["standard", "summarize", "docling"]),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Endpoint to upload a document for processing and embedding.
    The file is processed in the background.
    - **strategy**: 'standard' or 'summarize', 'docling'.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    try:
        # The service handles saving the file and processing it
        result = await document_service.process_and_embed_document(file, strategy=strategy)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"message": f"File '{file.filename}' is being processed with strategy '{strategy}'.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/scrape", summary="Scrape a URL and process its content")
async def scrape_url(
    payload: ScrapeRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Endpoint to scrape a URL for processing and embedding.
    The content is processed in the background.
    """
    try:
        # The service handles scraping, processing, and embedding
        result = await document_service.scrape_and_embed_url(payload.url)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"message": f"URL '{payload.url}' is being processed.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.websocket("/ws/chat/{chat_id}")
async def websocket_chat(
    websocket: WebSocket,
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Handles the real-time chat communication over WebSocket.
    """
    await websocket.accept()

    # --- 1. Define the heartbeat function ---
    async def send_heartbeat():
        try:
            while True:
                # Wait for 20 seconds
                await asyncio.sleep(20)
                # Send a heartbeat message to the client
                await websocket.send_json({"type": "ping"})
                print(f"[Heartbeat] Sent ping to {chat_id}")
        except asyncio.CancelledError:
            # This happens when we cancel the task (connection closed)
            pass
        except Exception as e:
            print(f"[Heartbeat] Error sending ping: {e}")
        
    # --- 2. Start the heartbeat as a background task ---
    heartbeat_task = asyncio.create_task(send_heartbeat())
            


    try:
        while True:
            try:
                # Receive message from the client
                data = await websocket.receive_text()
                message_data = json.loads(data)
            
                # --- Get data from the client payload ---
                user_message = message_data.get("content")
                use_strict_context = message_data.get("use_strict_context", False)  # Default: Strict Mode ON
                temperature = message_data.get("temperature", 0.0) # Default to 0.2 if not provided

                # --- Configure agent components for this specific call ---
                configure_retriever_mode(use_strict_context)

                if not user_message:
                    continue

                # Save user message to database
                service = get_session_service()
                service.add_message(chat_id, "user", user_message)

                # Get the runnable agent from the chat service
                runnable = chat_service.get_runnable()

                # Load conversation history from database for context
                history = service.get_messages(chat_id)
                # Convert history to LangChain format
                lang_messages = []
                for msg in history[:-1]:
                    if msg["role"] == "user":
                        lang_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lang_messages.append(AIMessage(content=msg["content"]))
                
                # Add the new user message to the history
                lang_messages.append(HumanMessage(content=user_message))
                input_data = {"messages": lang_messages}
            
                # Define the configuration for this specific run
                run_config = {
                    "configurable": {
                        "thread_id": chat_id,
                        "temperature": temperature  # Pass the temperature here
                    },
                    "recursion_limit": 150
                }
                print(f"Executing chat for {chat_id} with temperature={temperature}")

                full_response = ""
                async for chunk in runnable.astream(
                    input_data,
                    config=run_config,
                ):
                # Stream back chunks of the response as they are generated
                # The format of the chunk will depend on your graph's output
                # We're looking for the final answer from the supervisor
                    if "supervisor" in chunk:
                        supervisor_update = chunk["supervisor"]
                    
                        # --- NEW: Check for and send the routing decision ---
                        if "next" in supervisor_update and supervisor_update["next"] != "FINISH" and supervisor_update.get("next") is not None:
                            next_agent = supervisor_update["next"]
                            ## For tracing and debugging purpose
                            print(f"[CHAT_ID: {chat_id}] Sending routing: {next_agent}")
                            await websocket.send_json({"type": "routing", "data": f"Routing to {next_agent}..."})

                        # Check for final answer (existing logic)
                        final_answer_chunk = supervisor_update.get("messages", [])
                        if final_answer_chunk:
                            content = final_answer_chunk[-1].content
                            content_to_send = content.replace("Final Answer:", "").strip()
                            print(f"content_to_send//{content_to_send}")
                            if content_to_send:
                                full_response += content_to_send
                                ## For tracing and debugging purpose
                                print(f"[CHAT_ID: {chat_id}] Sending stream chunk...")
                                await websocket.send_json({"type": "stream", "data": content_to_send})

                # Save AI response to database
                if full_response:
                    service.add_message(chat_id, "assistant", full_response)
                    
                    # Auto-generate title based on the conversation
                    session_messages = service.get_messages(chat_id)
                    if len(session_messages) <= 2:
                        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
                        service.update_session_title(chat_id, title)
                        await websocket.send_json({"type": "title_update", "data": title})

                print(f"[CHAT_ID: {chat_id}] Sending stream end.")
                # Send a final message to indicate the end of the stream
                await websocket.send_json({"type": "stream_end"})
                print(f"[CHAT_ID: {chat_id}] Stream end sent successfully.")

            # --- CATCH ERRORS *INSIDE* THE LOOP ---
            except WebSocketDisconnect as e:
                # This will catch a disconnect on EITHER receive OR send
                print(f"!!! [CHAT_ID: {chat_id}] WebSocket disconnected during operation: {e.code}")
                # Break the inner loop to end this session
                break
            except Exception as e:
                # Catch other errors, like JSON parsing or send errors
                print(f"!!! [CHAT_ID: {chat_id}] An error occurred inside the chat loop: {e}")
                # You might want to send an error message to the client if the socket is still open
                try:
                    await websocket.send_json({"type": "error", "data": f"An internal error occurred: {e}"})
                except Exception as send_e:
                    print(f"!!! [CHAT_ID: {chat_id}] Failed to send error message, connection likely dead: {send_e}")
                # Break the loop to be safe
                break


    # --- OUTER ERROR HANDLING ---
    except WebSocketDisconnect:
        # This catches the initial disconnect if it happens on `accept` or `receive_text`
        print(f"Client disconnected from chat_id: {chat_id}")
    except Exception as e:
        # This catches errors outside the loop (e.g., on `accept`)
        print(f"An unexpected error occurred in websocket for chat_id {chat_id}: {e}")
        try:
            await websocket.close(code=1011, reason=f"An internal error occurred: {e}")
        except:
            pass

    # --- 4. CRITICAL CLEANUP (THE FINALLY BLOCK) ---
    finally:
        # This block ALWAYS runs, no matter how the loop ended.
        # We MUST cancel the heartbeat task here.
        print(f"[Cleanup] Cleaning up resources for {chat_id}")
        heartbeat_task.cancel()
        try:
            # Wait for the task to acknowledge cancellation
            await heartbeat_task
        except asyncio.CancelledError:
            # This is expected, we just cancelled it
            pass
