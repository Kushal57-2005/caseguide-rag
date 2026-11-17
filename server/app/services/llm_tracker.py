import threading
from typing import List, Dict, Any


class LLMCallTracker:
    """
    Thread-safe tracker to verify and record LLM completions invocations.
    Allows automated tests to prove 0 LLM calls occur during fallback.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.call_count = 0
        self.call_log: List[Dict[str, Any]] = []

    def record_call(self, caller: str, model: str = "", prompt_preview: str = ""):
        with self._lock:
            self.call_count += 1
            self.call_log.append({
                "caller": caller,
                "model": model,
                "prompt_preview": prompt_preview[:120] if prompt_preview else ""
            })

    def get_count(self) -> int:
        with self._lock:
            return self.call_count

    def get_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.call_log)

    def reset(self):
        with self._lock:
            self.call_count = 0
            self.call_log.clear()


llm_tracker = LLMCallTracker()
