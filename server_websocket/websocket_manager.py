import asyncio
from datetime import datetime
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect
import json

class WebsocketConnectionManager:
    def __init__(self, message_handlers):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
        self.message_handlers = message_handlers
        self.listen_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, expires_at: datetime):
        await websocket.accept()
        async with self.lock:
            self.active_connections[client_id] = websocket
            self.session_data[client_id] = {
                "requests": 0,
                "connected_at": datetime.utcnow(),
                "expires_at": expires_at
            }
            print(f"Client {client_id} connected.")
            # Start listening to messages for this client
            task = asyncio.create_task(self.listen_to_client(websocket, client_id))
            self.listen_tasks[client_id] = task

    async def disconnect(self, client_id: str):
        async with self.lock:
            websocket = self.active_connections.get(client_id)
            if websocket:
                await websocket.close()
                del self.active_connections[client_id]
                print(f"Client {client_id} disconnected.")
            if client_id in self.session_data:
                del self.session_data[client_id]
            # Cancel the listening task
            task = self.listen_tasks.get(client_id)
            if task:
                task.cancel()
                del self.listen_tasks[client_id]

    async def increment_request_count(self, client_id: str):
        async with self.lock:
            if client_id in self.session_data:
                self.session_data[client_id]['requests'] += 1

    async def listen_to_client(self, websocket: WebSocket, client_id: str):
        try:
            while True:
                data = await websocket.receive_text()
                await self.increment_request_count(client_id)
                message = json.loads(data)
                # print(f"Received message from {client_id}: {message}")

                action = message.get("action")
                payload = message.get("payload")

                handler = self.message_handlers.get(action)
                if handler:
                    try:
                        await handler(websocket, payload)
                    except Exception as handler_exc:
                        error_message = json.dumps({"error": f"Handler error: {str(handler_exc)}"})
                        await websocket.send_text(error_message)
                        print(f"Error in handler for action '{action}': {handler_exc}")
                else:
                    # Handle unknown actions
                    error_message = json.dumps({"error": f"Unknown action: {action}"})
                    await websocket.send_text(error_message)
        except WebSocketDisconnect:
            print(f"WebSocketDisconnect: Client {client_id} disconnected.")
            await self.disconnect(client_id)
        except asyncio.CancelledError:
            print(f"Listener task for {client_id} cancelled.")
        except Exception as e:
            print(f"Error in listening to client {client_id}: {e}")
            await self.disconnect(client_id)
            await websocket.close(code=1011, reason="Internal server error")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_json({"message": message})

    async def send_message(self, message: any):
        for websocket in self.active_connections.values():
            await websocket.send_json(message)
