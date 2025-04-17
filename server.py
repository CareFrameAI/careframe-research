from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request
import asyncio
import json
import uuid  # Import the uuid module

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified WebSocket manager
class ConnectionManager:
    SYSTEM_TOPIC = "system"  # Add the system topic constant

    def __init__(self):
        self.active_connections: dict = {}
        self.topics: dict = {}  # Store topic subscriptions
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = {
            "websocket": websocket,
            "subscriptions": set()  # Track topics this client is subscribed to
        }
        print(f"Client {client_id} connected")
        await self.subscribe(client_id, self.SYSTEM_TOPIC)  # Auto-subscribe to system topic
        # Add small delay to ensure proper initialization
        await asyncio.sleep(0.1)
        await self.send_client_list()  # Send client list
        # Send an initial system message for this client
        await self.broadcast_system_message({
            "type": "client_connected",
            "clientId": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            # Remove client from all their subscribed topics
            for topic in self.active_connections[client_id]["subscriptions"]:
                if topic in self.topics:
                    self.topics[topic].discard(client_id)
                    if not self.topics[topic]:  # If topic has no subscribers
                        del self.topics[topic]
            
            await self.active_connections[client_id]["websocket"].close()
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected")
            await self.send_client_list() # Send client list
            
    async def subscribe(self, client_id: str, topic: str):
        """Subscribe a client to a topic"""
        if client_id in self.active_connections:
            if topic not in self.topics:
                self.topics[topic] = set()
            self.topics[topic].add(client_id)
            self.active_connections[client_id]["subscriptions"].add(topic)
            print(f"Client {client_id} subscribed to {topic}")
            
    async def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe a client from a topic"""
        if topic in self.topics and client_id in self.topics[topic]:
            self.topics[topic].discard(client_id)
            self.active_connections[client_id]["subscriptions"].discard(topic)
            if not self.topics[topic]:  # If topic has no subscribers
                del self.topics[topic]
                
    async def publish(self, message: dict, publisher_id: str):
        """Publish a message to all subscribers of a topic"""
        topic = message.get("topic")
        if not topic:
            return
        
        # Validate message format
        if not isinstance(message.get("data"), (str, dict, list)):
            raise ValueError("Message data must be string, dict, or list")
        
        # Add size limit
        message_size = len(str(message))
        if message_size > 20 * 1024 * 1024:  # 20MB limit
            raise ValueError("Message size exceeds 20MB limit")
        
        if topic in self.topics:
            for subscriber_id in self.topics[topic]:
                if subscriber_id != publisher_id:
                    try:
                        await self.active_connections[subscriber_id]["websocket"].send_json({
                            "action": "message",
                            "payload": message
                        })
                    except Exception as e:
                        print(f"Error sending to {subscriber_id}: {e}")
                        # Remove failed connection
                        await self.disconnect(subscriber_id)

    async def send_client_list(self):
        """Sends the list of connected clients to all connected clients."""
        client_ids = list(self.active_connections.keys())
        # Send *only* to subscribers of the SYSTEM_TOPIC
        if self.SYSTEM_TOPIC in self.topics:
            for client_id in self.topics[self.SYSTEM_TOPIC]:
                try:
                    await self.active_connections[client_id]["websocket"].send_json({
                        "action": "client_list",
                        "payload": {"clients": client_ids}
                    })
                except Exception as e:
                    print(f"Error sending client list to {client_id}: {e}")

    async def broadcast_system_message(self, message_data: dict):
        """Broadcasts a message to the system topic."""
        message = {
            "topic": self.SYSTEM_TOPIC,
            "data": message_data,
            "context": "system"  # You can add context if needed
        }
        await self.publish(message, "system") # System is publisher

# Create a single instance of connection manager
manager = ConnectionManager()

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# {
#   "action": "publish",
#   "payload": {
#     "topic": "example",
#     "data": "your data",
#     "context": "your context"
#   }
# }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with pub/sub support"""
    client_id = str(uuid.uuid4()) # Generate unique ID
    try:
        await manager.connect(websocket, client_id)
        # Example: Broadcast a system message when a client connects
        await manager.broadcast_system_message({"message": f"Client {client_id} connected"})
        
        while True:
            try:
                data = await websocket.receive_json()
                print(f"Received message from {client_id}: {data}")
                
                action = data.get("action", "")
                payload = data.get("payload", {})
                
                if action == "subscribe":
                    topic = payload.get("topic")
                    if topic:
                        await manager.subscribe(client_id, topic)
                        response = {
                            "action": "subscribe_ack",
                            "payload": {"topic": topic}
                        }
                        await websocket.send_json(response)
                
                elif action == "unsubscribe":
                    topic = payload.get("topic")
                    if topic:
                        await manager.unsubscribe(client_id, topic)
                        response = {
                            "action": "unsubscribe_ack",
                            "payload": {"topic": topic}
                        }
                        await websocket.send_json(response)
                
                elif action == "publish":
                    await manager.publish(payload, client_id)
                    response = {
                        "action": "publish_ack",
                        "payload": {"message": "Message published"}
                    }
                    await websocket.send_json(response)
                
                elif action == "ping":
                    await websocket.send_json({
                        "action": "pong",
                        "payload": {"message": "pong"}
                    })
                
            except WebSocketDisconnect:
                await manager.disconnect(client_id)
                break
                
    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
        await manager.disconnect(client_id)
        
    finally:
        if client_id in manager.active_connections:
            await manager.disconnect(client_id)


