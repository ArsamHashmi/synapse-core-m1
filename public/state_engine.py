# state_engine.py
from __future__ import annotations
from typing import Dict, Any, Optional, List

import memory_engine

# ========= Conversation State =========

conversation_state: Dict[str, Any] = {
    "relationship_stage": "stranger",   # "stranger" | "getting_to_know" | "friend" | "close_friend"
    "user_mood": "neutral",             # "neutral" | "sad" | "stressed" | "happy" | "angry" | "tired"
    "ai_mood": "chill",                 # "chill" | "playful" | "serious" | "supportive"
    "privacy_refusal": False,
    "message_count": 0,
    "open_loops": [],                   # raw text about future stuff ("next week", "tomorrow")
    "ai_energy": 70,                    # 0–100, how high-energy Emily feels
    "ai_trust_level": 10,               # 0–100, how comfortable she feels with the user
    # concern tracker list: each = {"type", "text", "created_at", "last_asked_at", "resolved"}
    "concerns": [],
}


def get_state_copy() -> Dict[str, Any]:
    return dict(conversation_state)


# ---------- Mood detection ----------

def detect_user_mood(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["sad", "depressed", "down", "upset", "lonely", "cry"]):
        return "sad"
    if any(w in t for w in ["stressed", "overwhelmed", "anxious", "anxiety"]):
        return "stressed"
    if any(w in t for w in ["angry", "pissed", "mad", "furious"]):
        return "angry"
    if any(w in t for w in ["excited", "hyped", "happy", "good", "great", "awesome"]):
        return "happy"
    if any(w in t for w in ["tired", "exhausted", "sleepy", "drained"]):
        return "tired"
    return "neutral"


def update_ai_mood(user_mood: str) -> str:
    if user_mood in ["sad", "stressed", "angry", "tired"]:
        return "supportive"
    if user_mood == "happy":
        return "playful"
    return "chill"


# ---------- Privacy / boundaries ----------

def wants_privacy(text: str) -> bool:
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "don't want to share",
            "dont want to share",
            "don't wanna share",
            "dont wanna share",
            "private information",
            "too private",
            "not comfortable sharing",
            "none of your business",
        ]
    )


# ---------- Relationship state ----------

def update_relationship_stage():
    count = memory_engine.get_memory_count()
    if count == 0:
        stage = "stranger"
    elif count < 3:
        stage = "getting_to_know"
    elif count < 10:
        stage = "friend"
    else:
        stage = "close_friend"

    conversation_state["relationship_stage"] = stage


# ---------- Curiosity / deeper questions ----------

def should_ask_deeper_question() -> bool:
    n = conversation_state["message_count"]
    if conversation_state["relationship_stage"] in ["friend", "close_friend"]:
        return n > 0 and n % 6 == 0
    return False


# ---------- Open loops / follow-ups ----------

def maybe_register_open_loop(text: str):
    t = text.lower()
    if "next week" in t or "tomorrow" in t or "soon" in t:
        conversation_state["open_loops"].append(text)


# ---------- Energy / trust dynamics ----------

def update_ai_energy_and_trust(user_text: str):
    conversation_state["ai_energy"] = max(
        30, conversation_state["ai_energy"] - 1
    )

    t = user_text.lower()

    if any(w in t for w in ["haha", "lol", "lmao", "love this", "this is nice"]):
        conversation_state["ai_energy"] = min(
            100, conversation_state["ai_energy"] + 3
        )
        conversation_state["ai_trust_level"] = min(
            100, conversation_state["ai_trust_level"] + 2
        )

    if any(w in t for w in ["stupid", "dumb", "hate you", "annoying"]):
        conversation_state["ai_trust_level"] = max(
            0, conversation_state["ai_trust_level"] - 5
        )


# ---------- Concern tracking (health, exams, etc.) ----------

def scan_for_concerns(text: str):
    """
    Scan user text for things Emily should FOLLOW UP on later.
    Examples:
      - "i'm sick", "i'm not feeling well"
      - "i have an exam tomorrow"
      - "my interview is next week"
      - "i'm worried about everything in life"
    """
    t = text.lower()
    msg_idx = conversation_state["message_count"]

    if any(p in t for p in ["i'm sick", "im sick", "not feeling well", "feeling sick", "got sick"]):
        _register_concern("health", text, msg_idx)
    if "exam" in t or "test" in t:
        _register_concern("exam", text, msg_idx)
    if "interview" in t or "job interview" in t:
        _register_concern("interview", text, msg_idx)
    if any(p in t for p in ["breakup", "broke up", "relationship ended"]):
        _register_concern("relationship", text, msg_idx)

    # generic life worry / lost feeling
    if ("worried" in t or "worry" in t or "don't know what to do" in t or "dont know what to do" in t or "lost about my life" in t):
        _register_concern("general_worry", text, msg_idx)


def _register_concern(ctype: str, raw_text: str, msg_idx: int):
    for c in conversation_state["concerns"]:
        if c["type"] == ctype and not c["resolved"]:
            return

    concern = {
        "type": ctype,
        "text": raw_text.strip(),
        "created_at": msg_idx,
        "last_asked_at": None,
        "resolved": False,
    }
    conversation_state["concerns"].append(concern)


def pick_concern_to_follow_up() -> Optional[dict]:
    msg_idx = conversation_state["message_count"]
    best = None
    best_score = -1

    for c in conversation_state["concerns"]:
        if c["resolved"]:
            continue

        age = msg_idx - c["created_at"]
        if age < 3:
            continue

        if c["last_asked_at"] is not None:
            since = msg_idx - c["last_asked_at"]
            if since < 8:
                continue

        score = age
        if score > best_score:
            best_score = score
            best = c

    return best


def mark_concern_asked(concern: Optional[dict]):
    if concern is None:
        return
    concern["last_asked_at"] = conversation_state["message_count"]


def maybe_mark_concerns_resolved(user_text: str):
    t = user_text.lower()
    if any(p in t for p in ["i'm better", "im better", "feeling better", "much better now"]):
        for c in conversation_state["concerns"]:
            if c["type"] == "health" and not c["resolved"]:
                c["resolved"] = True


# ---------- State update entrypoint ----------

def update_conversation_state_from_user(text: str):
    conversation_state["message_count"] += 1

    user_mood = detect_user_mood(text)
    conversation_state["user_mood"] = user_mood
    conversation_state["ai_mood"] = update_ai_mood(user_mood)

    conversation_state["privacy_refusal"] = wants_privacy(text)

    maybe_register_open_loop(text)
    update_relationship_stage()
    update_ai_energy_and_trust(text)
    scan_for_concerns(text)
    maybe_mark_concerns_resolved(text)