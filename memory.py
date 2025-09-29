# utils/memory.py

class SessionMemory:
    def __init__(self, window: int = 8):
        # store memory in a dictionary {session_id: [messages]}
        self.window = window
        self._store = {}

    async def append_message(self, session_id: str, message: dict):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(message)
        # keep only last `window` messages
        self._store[session_id] = self._store[session_id][-self.window:]

    async def get_history(self, session_id: str):
        return self._store.get(session_id, [])

    async def clear(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]
