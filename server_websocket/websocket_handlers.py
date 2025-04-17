from typing import Any, Callable, Dict
import functools

# =====================
# Websocket Handlers
# =====================
def message_handler(action: str):
    """
    Decorator to register a function as a handler for a specific action.
    """
    def decorator(func: Callable[..., Any]):
        message_handlers[action] = func
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator

message_handlers: Dict[str, Callable[..., Any]] = {}
sessions = {}
# privacy_module = PrivacyModule()
# privacy_module.close()


@message_handler("ping")
async def handle_ping(websocket: Any, payload: Any):
    """Handler for 'ping' action."""
    await websocket.send_json({"action": "ping", "payload": {"message": "pong"}})
