from __future__ import annotations

from task_agent.models import TaskState
from task_agent.parsing import classify_task
from task_agent.runtime import TaskRuntime
from task_agent.storage import TaskStateStore


class TaskService:
    def __init__(self, store: TaskStateStore | None = None, runtime: TaskRuntime | None = None):
        self.store = store or TaskStateStore()
        self.runtime = runtime or TaskRuntime()

    def start_task(self, query: str, session_id: str = "default") -> dict:
        state = TaskState(goal=query, session_id=session_id, task_type=classify_task(query))
        state = self.runtime.bootstrap(state)
        state = self.runtime.advance(state)
        self.store.save(state)
        return state.to_response()

    def continue_task(self, task_id: str, user_input: str) -> dict:
        state = self._load_or_failure(task_id)
        if state is None:
            return self._failure_response(task_id, f"Unknown task_id: {task_id}")
        state = self.runtime.continue_with_input(state, user_input)
        state = self.runtime.advance(state)
        self.store.save(state)
        return state.to_response()

    def confirm_task_action(self, task_id: str, confirmation_id: str, approved: bool) -> dict:
        state = self._load_or_failure(task_id)
        if state is None:
            return self._failure_response(task_id, f"Unknown task_id: {task_id}")
        try:
            state = self.runtime.apply_confirmation(state, confirmation_id, approved)
        except KeyError:
            return self._confirmation_failure_response(state, confirmation_id)
        self.store.save(state)
        return state.to_response()

    def get_task_status(self, task_id: str) -> dict:
        state = self._load_or_failure(task_id)
        if state is None:
            return self._failure_response(task_id, f"Unknown task_id: {task_id}")
        return state.to_response()

    def _load_or_failure(self, task_id: str) -> TaskState | None:
        try:
            return self.store.load(task_id)
        except KeyError:
            return None

    def _failure_response(self, task_id: str, message: str) -> dict:
        return {
            "task_id": task_id,
            "session_id": "",
            "goal": "",
            "task_type": "faq",
            "task_status": "failed",
            "active_agent": "supervisor",
            "plan": [],
            "current_step": 0,
            "collected_info": {},
            "tool_outputs": {},
            "pending_questions": [],
            "pending_confirmations": [],
            "completed_actions": [],
            "error_log": [message],
            "user_visible_result": {"error_code": "unknown_task", "message": message},
            "result": {"error_code": "unknown_task", "message": message},
        }

    def _confirmation_failure_response(self, state: TaskState, confirmation_id: str) -> dict:
        message = f"Unknown confirmation_id: {confirmation_id}"
        payload = state.to_response()
        payload["error_log"] = [*payload.get("error_log", []), message]
        result = dict(payload.get("result", {}))
        result.update({"error_code": "unknown_confirmation", "message": message})
        payload["result"] = result
        payload["user_visible_result"] = result
        return payload


_default_service: TaskService | None = None


def get_service() -> TaskService:
    global _default_service
    if _default_service is None:
        _default_service = TaskService()
    return _default_service


def start_task(query: str, session_id: str = "default") -> dict:
    return get_service().start_task(query, session_id=session_id)


def continue_task(task_id: str, user_input: str) -> dict:
    return get_service().continue_task(task_id, user_input=user_input)


def confirm_task_action(task_id: str, confirmation_id: str, approved: bool) -> dict:
    return get_service().confirm_task_action(task_id=task_id, confirmation_id=confirmation_id, approved=approved)


def get_task_status(task_id: str) -> dict:
    return get_service().get_task_status(task_id=task_id)
