from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from collections import defaultdict
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_attempts: int = 100, block_time: int = 60 * 5):
        """
        Initialize the RateLimitMiddleware.

        :param app: The FastAPI application.
        :param max_attempts: Maximum number of allowed requests within the block_time window.
        :param block_time: Time window in seconds for rate limiting.
        """
        super().__init__(app)
        self.max_attempts = max_attempts
        self.block_time = block_time
        self.attempts = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        now = time.time()

        # Clean up old attempts
        self.attempts[ip] = [timestamp for timestamp in self.attempts[ip] if now - timestamp < self.block_time]

        if len(self.attempts[ip]) >= self.max_attempts:
            return Response(
                content="Too many requests. Please try again later.",
                status_code=429
            )

        # Record the current attempt
        self.attempts[ip].append(now)

        response = await call_next(request)
        return response
