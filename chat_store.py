from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORE_PATH = Path(__file__).resolve().parent / "chat_history.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_store() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return {"active_chat_id": "", "chats": {}}

    try:
        with STORE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"active_chat_id": "", "chats": {}}

    if not isinstance(data, dict):
        return {"active_chat_id": "", "chats": {}}

    data.setdefault("active_chat_id", "")
    data.setdefault("chats", {})
    if not isinstance(data["chats"], dict):
        data["chats"] = {}
    return data


def save_store(store: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STORE_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=True, indent=2)
    temp_path.replace(STORE_PATH)


def create_chat(store: dict[str, Any], title: str = "New Chat") -> str:
    chat_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    store["chats"][chat_id] = {
        "id": chat_id,
        "title": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
    }
    store["active_chat_id"] = chat_id
    save_store(store)
    return chat_id


def touch_chat(store: dict[str, Any], chat_id: str) -> None:
    chat = store["chats"].get(chat_id)
    if chat:
        chat["updated_at"] = now_iso()
        save_store(store)


def append_exchange(store: dict[str, Any], chat_id: str, user_text: str, assistant_text: str, sources: list[str] | None = None) -> None:
    chat = store["chats"].setdefault(
        chat_id,
        {
            "id": chat_id,
            "title": "New Chat",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "messages": [],
        },
    )

    messages = chat.setdefault("messages", [])
    messages.append({"role": "user", "content": user_text})
    messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "sources": sources or [],
        }
    )

    if chat.get("title", "New Chat") == "New Chat":
        first_line = user_text.strip().splitlines()[0][:48]
        chat["title"] = first_line or "New Chat"

    chat["updated_at"] = now_iso()
    store["active_chat_id"] = chat_id
    save_store(store)


def rename_chat(store: dict[str, Any], chat_id: str, title: str) -> None:
    chat = store["chats"].get(chat_id)
    if not chat:
        return

    cleaned = title.strip() or "New Chat"
    chat["title"] = cleaned
    chat["updated_at"] = now_iso()
    store["active_chat_id"] = chat_id
    save_store(store)


def delete_chat(store: dict[str, Any], chat_id: str) -> str:
    store["chats"].pop(chat_id, None)

    if not store["chats"]:
        return create_chat(store, "New Chat")

    remaining = sorted(
        store["chats"].values(),
        key=lambda chat: chat.get("updated_at", ""),
        reverse=True,
    )
    active_chat_id = remaining[0]["id"]
    store["active_chat_id"] = active_chat_id
    save_store(store)
    return active_chat_id
