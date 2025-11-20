# app.py
from __future__ import annotations
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from typing import List, Dict, Any

import memory_engine
import state_engine
from persona import persona_profile
from brain import generate_bot_reply

app = Flask(__name__)

# In-memory chat history (for UI only)
# each item: {"sender": "user"/"bot", "text": "...", "time": "HH:MM"}
messages: List[Dict[str, Any]] = []


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/messages", methods=["GET"])
def get_messages():
    return jsonify(messages)


@app.route("/send", methods=["POST"])
def send_message():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Empty message"}), 400

    now = datetime.now().strftime("%H:%M")

    user_msg = {
        "sender": "user",
        "text": text,
        "time": now,
    }
    messages.append(user_msg)

    # maybe store user notes from this message
    memory_engine.maybe_store_user_message(text)

    # AI reply (raw -> watchman -> postprocess)
    bot_text = generate_bot_reply(text, messages)
    bot_msg = {
        "sender": "bot",
        "text": bot_text,
        "time": now,
    }
    messages.append(bot_msg)

    return jsonify({"user": user_msg, "bot": bot_msg})


@app.route("/push_bot_message", methods=["POST"])
def push_bot_message():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty bot message"}), 400

    now = datetime.now().strftime("%H:%M")

    bot_msg = {
        "sender": "bot",
        "text": text,
        "time": now,
    }
    messages.append(bot_msg)

    return jsonify(bot_msg)


@app.route("/user_state", methods=["GET"])
def get_user_state():
    """
    Debug endpoint: exposes what the AI 'knows' and current internal state.
    Use ONLY for research / UI panel, not for public users.
    """
    memory_notes = memory_engine.get_all_memory_notes()
    state_copy = state_engine.get_state_copy()

    return jsonify({
        "persona": persona_profile,
        "state": state_copy,
        "memory_notes": memory_notes,
    })


if __name__ == "__main__":
    memory_engine.init_memory()
    app.run(debug=True)