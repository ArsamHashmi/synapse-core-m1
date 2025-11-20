# state_engine.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
import random

import memory_engine

# ========= Conversation State =========

conversation_state: Dict[str, Any] = {
    # Social bonding
    "relationship_stage": "stranger",   # "stranger" | "getting_to_know" | "friend" | "close_friend"
    "ai_trust_level": 10,               # 0–100, how comfortable the AI feels with the user

    # Affective state
    "user_mood": "neutral",             # "neutral" | "sad" | "stressed" | "happy" | "angry" | "tired"
    "ai_mood": "chill",                 # "chill" | "playful" | "serious" | "supportive"
    "ai_energy": 70,                    # 0–100, how high-energy the AI feels

    # Safety / boundaries
    "privacy_refusal": False,

    # Interaction stats
    "message_count": 0,
    "user_engagement": 50,              # 0–100 estimate of how engaged the user is

    # Future references
    "open_loops": [],                   # raw text about future stuff ("next week", "tomorrow")

    # concern tracker list: each = {"type", "text", "created_at", "last_asked_at", "resolved", "severity"}
    "concerns": [],

    # multi-message story buffering
    "story_buffer_active": False,
    "story_buffer": "",
}

STATE_LOG = Path("state_log.jsonl")  # for research / analysis


def get_state_copy() -> Dict[str, Any]:
    return dict(conversation_state)


# ---------- Mood detection (lexical + emoji) ----------

def detect_user_mood(text: str) -> str:
    """
    Lightweight, deterministic mood classifier based on:
    - emojis / symbols
    - affective keywords
    """
    t = (text or "").strip()
    tl = t.lower()

    # Emoji / symbol cues first (strong signal)
    if any(e in t for e in ["😭", "😢", "💔", "🥹"]):
        return "sad"
    if any(e in t for e in ["😂", "🤣", "😆", "😹"]):
        return "happy"
    if any(e in t for e in ["😡", "🤬", "😤"]):
        return "angry"
    if any(e in t for e in ["😴", "🥱", "💤"]):
        return "tired"

    # Lexical cues
    if any(w in tl for w in ["sad", "depressed", "down", "upset", "lonely", "cry"]):
        return "sad"
    if any(w in tl for w in ["stressed", "overwhelmed", "anxious", "anxiety", "panic"]):
        return "stressed"
    if any(w in tl for w in ["angry", "pissed", "mad", "furious", "irritated"]):
        return "angry"
    if any(w in tl for w in ["excited", "hyped", "happy", "good", "great", "awesome", "loving it"]):
        return "happy"
    if any(w in tl for w in ["tired", "exhausted", "sleepy", "drained", "worn out"]):
        return "tired"

    return "neutral"


def update_ai_mood(user_mood: str) -> str:
    """
    Simple policy: mirror negative affect with supportive tone,
    mirror positive affect with playful tone, otherwise stay chill.
    """
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


# ---------- Relationship state (based on structured memory) ----------

def update_relationship_stage():
    """
    Compute a relationship "score" from structured memory and map it to a stage.
    Relationship-related memories and high-importance items count more.
    """
    try:
        items = memory_engine.get_structured_memory()
    except Exception as e:
        print("⚠️ failed to read structured memory in update_relationship_stage:", e)
        items = []

    score = 0
    for m in items:
        mtype = (m.get("type") or "").lower()
        imp = int(m.get("importance", 1))

        if mtype in ["relationship", "relationship_state"]:
            score += 3 * imp
        elif mtype in ["goal", "concern"]:
            score += 2 * imp
        else:
            score += 1

    if score == 0:
        stage = "stranger"
    elif score < 6:
        stage = "getting_to_know"
    elif score < 20:
        stage = "friend"
    else:
        stage = "close_friend"

    conversation_state["relationship_stage"] = stage


# ---------- Curiosity / deeper questions ----------

def should_ask_deeper_question() -> bool:
    """
    Decide probabilistically whether to ask a deeper question.

    Conditions:
    - Only in "friend" or "close_friend".
    - After at least a few turns.
    - Trust and energy must be above thresholds.
    - Then ~15% chance per eligible turn.
    """
    n = conversation_state["message_count"]
    stage = conversation_state["relationship_stage"]
    trust = conversation_state["ai_trust_level"]
    energy = conversation_state["ai_energy"]

    if stage not in ["friend", "close_friend"]:
        return False
    if n < 5:
        return False
    if trust < 30 or energy < 40:
        return False

    # probabilistic trigger; deterministic enough for stats, non-robotic in feel
    return random.random() < 0.15


# ---------- Open loops / follow-ups ----------

def maybe_register_open_loop(text: str):
    t = text.lower()
    if "next week" in t or "tomorrow" in t or "soon" in t or "next month" in t:
        conversation_state["open_loops"].append(text.strip())


# ---------- Energy / trust dynamics ----------

def update_ai_energy_and_trust(user_text: str):
    """
    Simple, interpretable dynamics:
    - Energy decays slowly every turn.
    - Positive reactions (lol, love this) increase energy + trust.
    - Hostile language decreases trust.
    """
    # base decay
    conversation_state["ai_energy"] = max(
        30, conversation_state["ai_energy"] - 1
    )

    t = user_text.lower()

    if any(w in t for w in ["haha", "lol", "lmao", "love this", "this is nice", "that's nice", "thats nice"]):
        conversation_state["ai_energy"] = min(
            100, conversation_state["ai_energy"] + 3
        )
        conversation_state["ai_trust_level"] = min(
            100, conversation_state["ai_trust_level"] + 2
        )

    if any(w in t for w in ["stupid", "dumb", "hate you", "annoying", "useless"]):
        conversation_state["ai_trust_level"] = max(
            0, conversation_state["ai_trust_level"] - 5
        )


# ---------- Engagement estimation ----------

def update_engagement_from_text(text: str):
    """
    Approximate user engagement from simple local cues:
    - message length
    - presence of questions
    """
    t = (text or "").strip()
    length = len(t)
    questions = t.count("?")

    score = 0
    if length > 40:
        score += 1
    if length > 120:
        score += 1
    if questions > 0:
        score += 1

    old = conversation_state.get("user_engagement", 50)
    # small decay + boost from score
    new = old - 1 + 5 * score
    conversation_state["user_engagement"] = max(0, min(100, new))


# ---------- Concern tracking (health, exams, etc.) ----------

def scan_for_concerns(text: str):
    """
    Scan user text for things the agent should FOLLOW UP on later.
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
    if (
        "worried" in t
        or "worry" in t
        or "don't know what to do" in t
        or "dont know what to do" in t
        or "lost about my life" in t
        or "lost in life" in t
    ):
        _register_concern("general_worry", text, msg_idx)


def _register_concern(ctype: str, raw_text: str, msg_idx: int):
    """
    Add a concern if there isn't already an unresolved one of this type.
    Also mirror it into long-term memory as a 'concern' note with severity.
    """
    for c in conversation_state["concerns"]:
        if c["type"] == ctype and not c["resolved"]:
            return

    t = raw_text.lower()
    severity = 1
    if any(w in t for w in ["really scared", "panic", "freaking out", "can't sleep", "cant sleep", "very worried"]):
        severity = 3
    elif any(w in t for w in ["worried", "nervous", "stressed", "anxious"]):
        severity = 2

    concern = {
        "type": ctype,
        "text": raw_text.strip(),
        "created_at": msg_idx,
        "last_asked_at": None,
        "resolved": False,
        "severity": severity,
    }
    conversation_state["concerns"].append(concern)

    # Mirror into long-term memory
    note = f"user has an ongoing concern about {ctype}: {raw_text.strip()}"
    try:
        memory_engine.store_memory(note, source_message=raw_text, created_at=msg_idx)
    except Exception as e:
        print("⚠️ failed to store concern in memory:", e)


def pick_concern_to_follow_up() -> Optional[dict]:
    """
    Select an unresolved concern to ask about:
    - must be at least a few messages old (age >= 3)
    - must not have been asked about too recently
    - score = age + 3*severity
    """
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

        score = age + 3 * c.get("severity", 1)
        if score > best_score:
            best_score = score
            best = c

    return best


def mark_concern_asked(concern: Optional[dict]):
    if concern is None:
        return
    concern["last_asked_at"] = conversation_state["message_count"]


def maybe_mark_concerns_resolved(user_text: str):
    """
    If user says they are better, mark health concerns resolved.
    You can extend this later for exams/interviews etc.
    """
    t = user_text.lower()
    if any(p in t for p in ["i'm better", "im better", "feeling better", "much better now"]):
        for c in conversation_state["concerns"]:
            if c["type"] == "health" and not c["resolved"]:
                c["resolved"] = True


# ---------- Logging for research ----------

def _log_state(user_text: str):
    """
    Append a JSON snapshot of the current state + raw user_text to state_log.jsonl.
    Useful for analysis / plotting in your research.
    """
    snap = {
        "ts": time.time(),
        "message_index": conversation_state["message_count"],
        "user_text": user_text,
        "state": get_state_copy(),
    }
    try:
        with STATE_LOG.open("a", encoding="utf8") as f:
            f.write(json.dumps(snap) + "\n")
    except Exception as e:
        print("⚠️ failed to write state log:", e)


# ---------- State update entrypoint ----------

def update_conversation_state_from_user(text: str):
    """
    Main entrypoint called once per user message.
    Updates all latent state variables and logs a snapshot.
    """
    conversation_state["message_count"] += 1

    user_mood = detect_user_mood(text)
    conversation_state["user_mood"] = user_mood
    conversation_state["ai_mood"] = update_ai_mood(user_mood)

    conversation_state["privacy_refusal"] = wants_privacy(text)

    maybe_register_open_loop(text)
    update_relationship_stage()
    update_ai_energy_and_trust(text)
    update_engagement_from_text(text)
    scan_for_concerns(text)
    maybe_mark_concerns_resolved(text)

    _log_state(text)

def should_hold_reply(text: str) -> bool:
    """
    Decide if we should *not* reply yet and just buffer this message.
    You can tune these rules.
    """
    t = (text or "").strip().lower()

    # explicit markers: user tells us they are sending in parts
    if t in ["wait", "one sec", "hold on", "more coming", "i'll explain", "ill explain"]:
        conversation_state["story_buffer_active"] = True
        return True

    # if we're already in story mode, we usually buffer
    if conversation_state.get("story_buffer_active", False):
        # if they say done, we should NOT hold, we’ll flush instead
        if t in ["done", "that's it", "thats it", "finished", "end"]:
            return False
        # otherwise, keep buffering
        return True

    # heuristic: messages ending with "..." mean "i'm not done yet"
    if t.endswith("..."):
        conversation_state["story_buffer_active"] = True
        return True

    return False


def append_to_story_buffer(text: str):
    """
    Add this chunk to the story buffer.
    """
    buf = conversation_state.get("story_buffer", "")
    if buf:
        buf = buf + " " + text.strip()
    else:
        buf = text.strip()
    conversation_state["story_buffer"] = buf


def consume_story_with(text: str) -> str:
    """
    When story is complete, merge buffer + current text and reset.
    """
    buf = conversation_state.get("story_buffer", "").strip()
    conversation_state["story_buffer"] = ""
    conversation_state["story_buffer_active"] = False

    if buf:
        # if 'done' / 'that's it' etc., don't duplicate that
        t = text.strip().lower()
        if t in ["done", "that's it", "thats it", "finished", "end"]:
            return buf
        return (buf + " " + text).strip()

    return text    