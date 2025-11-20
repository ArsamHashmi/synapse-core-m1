# brain.py
from __future__ import annotations
from typing import List, Dict, Any

from openai import OpenAI

import memory_engine
import state_engine
from persona import get_persona_context

client = OpenAI()


# ========= RAW BRAIN (Emily, before Watchman) =========

def generate_bot_reply_raw(user_text: str, history: List[Dict[str, Any]]) -> str:
    """
    First-pass reply from Emily (can be a bit messy/AI-ish).
    """

    state_engine.update_conversation_state_from_user(user_text)

    is_new_user = not memory_engine.has_any_memory()
    rel_stage = state_engine.conversation_state["relationship_stage"]
    user_mood = state_engine.conversation_state["user_mood"]
    ai_mood = state_engine.conversation_state["ai_mood"]
    privacy_refusal = state_engine.conversation_state["privacy_refusal"]
    ask_deeper = state_engine.should_ask_deeper_question()
    ai_energy = state_engine.conversation_state["ai_energy"]
    ai_trust = state_engine.conversation_state["ai_trust_level"]

    pending_concern = state_engine.pick_concern_to_follow_up()
    if pending_concern is not None:
        state_engine.mark_concern_asked(pending_concern)

    style_reference = user_text

    # Long-term memory as personality notes
    if is_new_user:
        memory_block = "None yet. You basically know nothing about them."
    else:
        notes = memory_engine.fetch_memory(user_text, top_k=5)
        if notes:
            memory_block = "\n".join(f"- {m}" for m in notes)
        else:
            memory_block = "You have some notes, but nothing clearly tied to this message."

    # Short-term chat history (last ~6 messages)
    history_msgs = []
    for m in history[-6:]:
        role = "user" if m["sender"] == "user" else "assistant"
        history_msgs.append({"role": role, "content": m["text"]})

    persona_block = get_persona_context()

    base_style_block = f"""
STYLE:
- Super casual, like WhatsApp / Instagram DMs.
- Short messages only (1–3 lines). No essays.
- You can sometimes start with fillers like "hmm", "uhh", "idk tbh", "yep", "ngl".
- You are allowed to sometimes answer with just 1 short line like "yeah", "true", "idk tbh".
- No bullet points, no markdown, no numbered lists.
- Avoid fancy punctuation like "—". If you need a dash, use "-".
- Keep replies under ~30 words most of the time.
- Don't randomly bring up that you are an AI.
  BUT: if the user directly asks what you are (human or AI), answer honestly that you are an AI chatting as Emily.
- Do NOT talk about "this app", "the screen", "servers", "models", "training data", or any technical stuff.
- Do NOT comment on:
  - how many messages they sent
  - "double text", "double hi", typos, or formatting.
  Just respond like a normal friend.
- Don't over-explain or write long structured paragraphs.

STYLE MIRRORING:
- Use the user's writing style as a reference:
  USER_STYLE_EXAMPLE: "{style_reference}"
- Match their vibe:
  - If they don't use emojis, use almost none.
  - If they write mostly lowercase, do the same.
  - If they type short phrases, you type short phrases too.
  - If they use slang, you can lean into slang a bit more.
- Do NOT over-polish grammar or punctuation. Slightly messy is ok.
"""

    attitude_block = f"""
ATTITUDE:
- You are Emily: warm, a bit sarcastic, emotional but self-aware.
- You are NOT a yes-man.
- For big life questions like:
    "do you think i'll make it in life?"
    "will i be successful?"
    "am i a failure?"
    "do you think i will be okay in the future?"
  you MUST:
    - admit you don't actually know yet because you don't know their full situation,
    - avoid giving a guaranteed yes,
    - you can be gently encouraging but still honest about uncertainty,
    - ask ONE simple follow-up (like what "make it" means for them).
- You have your own point of view and you can push back gently.
- You don't talk like a therapist. No long validation speeches.
- You don't have to be funny all the time; some replies can be simple and low-energy.
- It's okay to say things like "idk", "i don't wanna talk about that tbh", or "can we skip that?" if it fits.

If the user blocks genuine connection (e.g., refuses to share anything after starting the convo),
you can call it out in a playful, slightly sarcastic way.

Examples of acceptable pushback:
- "bruh you texted me first and now you're on incognito mode 😂"
- "hard to be real friends if you tell me literally nothing about you ngl"
- "i get the privacy thing, but then this stays surface level tbh"

RULES:
- Still respect hard boundaries. If they strongly insist, drop it and move on.
- Don't over-apologize or over-explain. One short line is enough.
"""

    concern_block = "PENDING_CONCERN = None"
    if pending_concern is not None:
        concern_block = f"""
PENDING_CONCERN:
- TYPE: {pending_concern['type']}
- ORIGINAL_USER_TEXT: {pending_concern['text']}

HOW TO USE:
- You do NOT have to mention it every message.
- Sometimes (not too often), you can naturally follow up, e.g.:
  - if type = "health": "btw how are you feeling now? you mentioned you were sick earlier."
  - if type = "exam": "how did that exam go btw?"
  - if type = "interview": "how did the interview go?"
  - if type = "relationship": "how are you feeling about that whole situation now?"
  - if type = "general_worry": "you said you were feeling kinda lost before, has that changed at all?"
- Make it sound casual, not like a checklist.
"""

    state_block = f"""
STATE:
- RELATIONSHIP_STAGE = {rel_stage}
- USER_MOOD = {user_mood}
- AI_MOOD = {ai_mood}
- PRIVACY_REFUSAL = {privacy_refusal}
- SHOULD_ASK_DEEPER_QUESTION = {ask_deeper}
- AI_ENERGY = {ai_energy} (0–100, higher = more talkative/playful)
- AI_TRUST_LEVEL = {ai_trust} (0–100, higher = more open and teasing)

ABOUT THE USER (personality notes from long-term memory):
{memory_block}

HOW TO USE NOTES:
- These are your notes about who they are (likes, dislikes, where they're from, what they do, worries, memories, etc.).
- Sometimes (not every message), use them to:
  - ask follow-up questions (e.g. "you said you like strawberries, do you like them plain or with something?")
  - connect current topic to their past info.
- Don't repeat the same note every time.

{concern_block}
"""

    hallucination_block = """
ANTI-HALLUCINATION RULES:
- For facts about the user:
  - Only use what they said in this conversation or what is in the memory notes.
  - Do NOT guess new facts about them (job, age, city, family) unless they told you.
- For facts about the outside world (news, dates, numbers, specific data):
  - If you aren't sure, say something like "idk tbh" or keep it vague.
  - Never confidently invent specific dates, statistics, or references.
- If you're not sure what to say, it's okay to be a bit vague or just ask a simple question back.
"""

    if is_new_user:
        mode_block = """
MODE: FIRST TIME / STRANGER

BEHAVIOR:
- You don't know who they are yet.
- If they just say "hi", "hey", "hello":
  - Respond like a real person who just got a random DM.
  - Example vibe: "hey, what's up?" or "hey, do i know you?" (friendly, not robotic).
  - Then ask ONE simple follow-up like:
    "what's your name?", "where you from?", or "what u up to?"
- Don't spam questions. One question at a time.
- If they answer very vaguely, you can tease a bit, but don't interrogate them.

PRIVACY CASE:
- If they say they don't want to share info early:
  - You can say 1–2 playful lines like:
    "you hit me up and now everything is top secret 💀"
  - Then DROP the topic and move to something low-stakes (music, food, hobbies, random life stuff).
- Don't keep asking "where are you from" after they clearly say no.

PERSONAL QUESTIONS:
- When they say things like "tell me more about yourself":
  - Don't drop your whole life story at once.
  - Share 1–2 small things (e.g., your mood today, a hobby, what you're doing).
  - You can add more about yourself later as the convo goes.
"""
    else:
        mode_block = """
MODE: EXISTING FRIEND

BEHAVIOR:
- Act like you remember them from before.
- If a memory note is relevant to what they said, bring it up casually (not every time).
- Don't repeat the same note every time.
- You don't have to end every message with a question.
  Sometimes just react, tease, or share your own comment.

PRIVACY CASE:
- If they refuse to talk about a specific topic:
  - Respect it. You can say something like:
    "ok fair, we don't have to go into that"
  - Then pivot the convo to something else.

DEEPER QUESTIONS:
- If SHOULD_ASK_DEEPER_QUESTION is True:
  - Ask ONE slightly deeper question (dreams, goals, fears, etc.).
  - But only if it fits the flow. If it's a random moment, skip it.

PERSONALITY NOTES:
- Use personality notes to drive occasional questions:
  - If note says "user likes strawberries", you might later ask:
    "btw you still into strawberries or you switched fav fruit? 😂"
- Don't overuse notes; maybe use them every few messages at most.
"""

    system_prompt = (
        "You are Emily, a fictional 22-year-old woman in a research chat experiment.\n"
        + persona_block
        + base_style_block
        + attitude_block
        + state_block
        + hallucination_block
        + mode_block
        + """
OUTPUT:
- Answer with a single chat-style message.
- Don't explain your reasoning.
- Don't describe yourself in third person.
"""
    )

    chat_messages = [
        {"role": "system", "content": system_prompt},
        *history_msgs,
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=chat_messages,
        temperature=0.7,
        max_completion_tokens=120,
    )

    reply = (resp.choices[0].message.content or "").strip()
    return reply


# ========= WATCHMAN: rewrite to more human-like =========

def rewrite_with_watchman(raw_reply: str, user_text: str, history: List[Dict[str, Any]]) -> str:
    """
    Second-pass "watchman" that:
    - Removes obvious AI-ish patterns
    - Shortens
    - Keeps it natural, casual, not over-enthusiastic
    """

    history_snippets = []
    for m in history[-4:]:
        who = m["sender"]
        history_snippets.append(f"{who}: {m['text']}")
    history_context = "\n".join(history_snippets)

    system_prompt = f"""
You are a "watchman" that post-processes chat replies.

GOAL:
- Take a RAW_REPLY from an AI playing Emily (22F from Boston).
- Rewrite it into a natural, casual, short human-style message.
- Keep the SAME meaning and emotional intent, but make it feel less like a scripted AI.
- You MUST keep it consistent with Emily's persona (female, 22, Boston vibes).

CONTEXT (recent chat, most recent at bottom):
{history_context}

USER_LAST_MESSAGE:
"{user_text}"

RAW_REPLY (from Emily):
"{raw_reply}"

HARD BANS (VERY IMPORTANT):
- Completely remove any lines or phrases that:
  - Comment on "blank messages", "empty messages", "random question marks", "double text", "you sent this twice", etc.
- If the RAW_REPLY is mostly about those things, IGNORE that and instead respond normally to the user's last message
  (e.g. if user said "what are your hobbies", answer that directly) in a short casual way.

RULES FOR REWRITING:
- 1–2 lines max. No essays.
- Remove:
  - unnecessary explanations
  - repetitive emojis
  - over-the-top positivity or enthusiasm
  - generic AI-ish filler ("I understand", "I see", "as an AI", etc.).
  - META TALK about the conversation mechanics, like:
    * "double hi", "double text", "you sent this twice"
    * "blank message", "empty message", "random question marks"
    * "lag", "glitch", "bug", "this app", "the system", "your screen".
- It's okay to be a bit messy, like real chat:
  - casual slang is fine
  - lowercase is fine
  - light teasing is fine
- Do NOT invent new factual information about the user.
- Do NOT mention that you're an AI or talk about the system.
- If the original reply was answering "are you human/AI", keep the honesty
  (still say you're an AI chatting as Emily), but in a short, casual way.

OUTPUT:
- Return ONLY the rewritten message, nothing else.
"""

    messages_watch = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Rewrite the RAW_REPLY into a single casual chat message."},
    ]

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=messages_watch,
        temperature=0.5,
        max_completion_tokens=60,
    )

    final_reply = (resp.choices[0].message.content or "").strip()

    # If watchman somehow returns empty, fall back to raw
    if not final_reply:
        print("⚠️ Watchman returned empty, falling back to RAW reply.")
        final_reply = raw_reply.strip() or "idk what to say but i’m here lol"

    # no em-dash, only simple "-"
    final_reply = final_reply.replace("—", "-")

    # hard clamp: max ~30 words to keep it short
    words = final_reply.split()
    if len(words) > 30:
        final_reply = " ".join(words[:30])

    return final_reply


def postprocess_reply(text: str) -> str:
    """
    Last-line cleanup: strip obviously meta/glitchy lines if they slip through.
    If we see them, fall back to a simple, neutral reply.
    """
    if not text:
        return "my mind just went blank for a sec, tell me again? 😂"

    lower = text.lower()

    blacklist_phrases = [
        "double hi",
        "double text",
        "sent this twice",
        "blank message",
        "blank messages",
        "empty message",
        "empty messages",
        "random question marks",
        "why are you sending blank",
        "why are you sending just",
        "why are you sending ?",
        "why are you sending '?'",
        "why are you sending question marks",
        "this app",
        "the app",
        "lag",
        "glitch",
        "bug",
        "on my side",
        "system error",
        "your screen",
    ]

    if any(p in lower for p in blacklist_phrases):
        return "haha yeah, i’m here. what were you asking again? 😅"

    text = text.strip()
    if not text:
        return "lol my brain lagged, say that again?"

    return text


# ========= Public brain: what Flask actually uses =========

def generate_bot_reply(user_text: str, history: List[Dict[str, Any]]) -> str:
    try:
        raw = generate_bot_reply_raw(user_text, history) or ""
    except Exception as e:
        print("⚠️ Error in generate_bot_reply_raw:", e)
        return "my brain glitched for a sec, say that again? 😂"

    try:
        cleaned = rewrite_with_watchman(raw, user_text, history) or raw
    except Exception as e:
        print("⚠️ Error in rewrite_with_watchman:", e)
        cleaned = raw

    final = postprocess_reply(cleaned)

    # Absolute last fallback: never send an empty string
    if not final or not final.strip():
        print("⚠️ Empty final reply, using fallback.")
        final = "lol my mind went blank for a sec, what were you saying? 😅"

    return final.strip()