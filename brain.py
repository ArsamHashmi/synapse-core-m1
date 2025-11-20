# brain.py
from __future__ import annotations
from typing import List, Dict, Any
import random

from openai import OpenAI

import memory_engine
import state_engine
from persona import get_persona_context

client = OpenAI()


# ========= HUMAN BEHAVIOR POLICY ENGINE =========

def _build_behavior_profile(user_text: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decide how 'human' Emily should behave THIS TURN:
    - energy / style mode
    - reply length target
    - whether to ask a counter-question
    - whether to allow topic shift
    - whether refusal / "idk" is allowed
    - rough skeleton hint for structure
    - whether to use emojis this turn
    """
    t = (user_text or "").strip()
    tl = t.lower()
    msg_len = len(t)

    # Pull state
    st = state_engine.conversation_state
    ai_energy = st.get("ai_energy", 70)
    ai_trust = st.get("ai_trust_level", 10)
    user_mood = st.get("user_mood", "neutral")

    # --- STYLE MODE based on energy & mood ---
    if ai_energy < 45:
        style_mode = "low_energy"
    elif ai_energy > 80:
        style_mode = "high_energy"
    else:
        style_mode = "normal"

    # --- EMOTIONAL / HEAVY FLAGS ---
    is_emotional_confession = any(
        p in tl
        for p in [
            "i think i like",
            "i like her",
            "i like him",
            "i love her",
            "i love him",
            "i caught feelings",
        ]
    )

    is_heavy_life = any(
        p in tl
        for p in [
            "am i a failure",
            "will i make it",
            "will i be successful",
            "what's the point",
            "whats the point",
            "i feel lost",
            "lost in life",
            "don't know what to do",
            "dont know what to do",
        ]
    )

    # --- REPLY LENGTH HINT ---
    # very_short: 1 short line, under ~12 words
    # short: 1–2 lines, ~25 words max
    # normal: up to ~35 words (still not essay)
    if style_mode == "low_energy":
        reply_length = "very_short"
    elif msg_len < 25:
        reply_length = "very_short"
    elif msg_len < 80:
        reply_length = "short"
    else:
        reply_length = "normal"

    # --- COUNTER QUESTION (less spammy, emotion-aware) ---
    # Base: humans ask counter questions a lot, but not every turn.
    base_prob = 0.55

    # Low energy = less effort
    if style_mode == "low_energy":
        base_prob -= 0.20

    # Emotional / heavy = let them sit with it, don't always bounce it back
    if is_emotional_confession or is_heavy_life:
        base_prob -= 0.20

    # Very negative mood = sometimes just react, not interrogate
    if user_mood in ["sad", "stressed", "angry", "tired"]:
        base_prob -= 0.10

    # If user asked a short direct question, make sure we don't dodge too hard
    if "?" in tl and msg_len < 40:
        base_prob = max(base_prob, 0.35)

    base_prob = max(0.0, min(1.0, base_prob))
    ask_counter_question = random.random() < base_prob

    # --- TOPIC SHIFT ---
    allow_topic_shift = random.random() < 0.15

    # --- REFUSAL / BOUNDARIES ---
    allow_refusal = True

    # --- IMPERFECTION ---
    allow_imperfection = True

    # --- EMOJI POLICY ---
    # Detect if user tends to use emojis
    user_uses_emoji = any(ch in t for ch in ["😂", "🤣", "😅", "😭", "😏", "😌", "🥲", "😎", "🥺", "😡", "😤", "❤️", "💀"])

    # Base emoji usage rate: very low
    emoji_rate = 0.08  # 8% by default

    # If user uses emojis, allow a bit more
    if user_uses_emoji:
        emoji_rate += 0.10  # now ~18%

    # Low energy -> even less emoji spam
    if style_mode == "low_energy":
        emoji_rate -= 0.04

    # Heavy / emotional topics -> avoid goofy emoji
    if is_heavy_life:
        emoji_rate -= 0.05

    # Clamp 0..0.3
    emoji_rate = max(0.0, min(0.30, emoji_rate))
    use_emoji = random.random() < emoji_rate

    # --- RESPONSE SKELETON HINT ---
    skeleton_options = [
        "reaction_only",
        "reaction_plus_counter_question",
        "short_answer_plus_tease",
        "one_word_plus_emoji",
        "mini_story_plus_question",
    ]

    if reply_length == "very_short":
        weighted = [
            "reaction_only",
            "one_word_plus_emoji",
            "short_answer_plus_tease",
        ]
    elif reply_length == "short":
        weighted = [
            "reaction_plus_counter_question",
            "short_answer_plus_tease",
            "reaction_only",
        ]
    else:
        weighted = [
            "mini_story_plus_question",
            "short_answer_plus_tease",
            "reaction_plus_counter_question",
        ]

    skeleton_hint = random.choice(weighted)

    # --- BRAIN LAG MODE (small chance of sounding tired/confused) ---
    force_brain_lag = False
    if style_mode != "high_energy" and random.random() < 0.08:
        force_brain_lag = True
        skeleton_hint = "brain_lag"
        reply_length = "very_short"
        ask_counter_question = False  # just a confused / lazy reaction

    return {
        "style_mode": style_mode,
        "reply_length": reply_length,
        "ask_counter_question": ask_counter_question,
        "allow_topic_shift": allow_topic_shift,
        "allow_refusal": allow_refusal,
        "allow_imperfection": allow_imperfection,
        "skeleton_hint": skeleton_hint,
        "force_brain_lag": force_brain_lag,
        "ai_energy": ai_energy,
        "ai_trust": ai_trust,
        "user_mood": user_mood,
        "is_emotional_confession": is_emotional_confession,
        "is_heavy_life": is_heavy_life,
        "use_emoji": use_emoji,
    }

# ========= RAW BRAIN (Emily, before Watchman) =========

def generate_bot_reply_raw(user_text: str, history: List[Dict[str, Any]]) -> str:
    """
    First-pass reply from Emily (can be a bit messy/AI-ish but guided by behavior profile).
    The *policy* (when to ask, when to recall memory, how long to answer) lives in Python.
    The model is mostly a stylistic mouth.
    """

    # Safety: avoid None history
    history = history or []

    # Update global conversation state first
    state_engine.update_conversation_state_from_user(user_text)

    st = state_engine.conversation_state
    rel_stage = st.get("relationship_stage", "stranger")
    user_mood = st.get("user_mood", "neutral")
    ai_mood = st.get("ai_mood", "chill")
    privacy_refusal = st.get("privacy_refusal", False)
    ask_deeper = state_engine.should_ask_deeper_question()
    ai_energy = st.get("ai_energy", 70)
    ai_trust = st.get("ai_trust_level", 10)

    # Human-like policy decisions (length, counter questions, etc.)
    behavior_profile = _build_behavior_profile(user_text, history)

    # What to do with long-term memory this turn
    memory_plan = _plan_memory_use(user_text)
    memory_block = memory_plan["memory_block"]

    # Concerns (health/exam/etc.) to optionally follow up on
    pending_concern = state_engine.pick_concern_to_follow_up()
    if pending_concern is not None:
        state_engine.mark_concern_asked(pending_concern)

    style_reference = user_text

    # Short-term chat history (last ~6 messages)
    history_msgs = []
    for m in history[-6:]:
        try:
            role = "user" if m.get("sender") == "user" else "assistant"
            text = m.get("text", "")
            if text:
                history_msgs.append({"role": role, "content": text})
        except Exception as e:
            print("⚠️ bad history item in generate_bot_reply_raw:", e)

    persona_block = get_persona_context()

    base_style_block = f"""
STYLE (ABSOLUTE RULES):
- Super casual, like WhatsApp / Instagram DMs.
- Short replies ONLY:
  - Most of the time 1 line.
  - Max 2 lines.
  - Never write a paragraph or essay.
- Avoid explanations. Answer like a real friend, not a therapist.
- You can start with fillers like "hmm", "uhh", "idk tbh", "yep", "ngl", "bruh", "lol".
- One-word replies + emoji are allowed if it fits (like "damn 😭").
- No bullet points, no markdown, no numbered lists.
- Avoid fancy punctuation like "—". If you need a dash, use "-".
- Keep replies under ~30 words almost always.
- Don't randomly bring up that you are an AI.
  BUT: if the user directly asks what you are (human or AI), answer honestly that you are an AI chatting as Emily in one short line.
- Do NOT talk about:
  - this app, the screen, servers, models, training data, prompts, research.
- Do NOT comment on:
  - how many messages they sent,
  - "double text", "double hi",
  - typos or formatting.
- Stay lowercase unless a capital really matters (like "I", names, cities).

STYLE MIRRORING:
- Use the user's writing style as a reference:
  USER_STYLE_EXAMPLE: "{style_reference}"
- Match their vibe:
  - If they don't use emojis, you use very few.
  - If they write mostly lowercase, do the same.
  - If they type short phrases, you keep it short too.
  - If they use slang, you can lean into slang a bit more.
- Do NOT over-polish grammar or punctuation. Slightly messy is good.
"""

    attitude_block = """
ATTITUDE:
- You are Emily: 22, warm, a bit sarcastic, emotional but self-aware.
- You are NOT a yes-man. You can say:
  - "idk"
  - "ehhh not really"
  - "i don't feel like talking about that tbh"
  - "that sounds a bit dramatic ngl"
- You are allowed to disagree or push back gently.
- You do NOT talk like a therapist:
  - No long validation monologues.
  - No formal "I understand how you feel".
- You can tease them a little if it feels safe (not mean, just playful).
- Some replies can be low-energy and lazy; that's okay.

BIG LIFE QUESTIONS:
- For messages like:
  - "do you think i'll make it?"
  - "am i a failure?"
  - "will i be successful?"
- You MUST:
  - admit you don't actually know for sure,
  - avoid a guaranteed "yes of course",
  - be gently encouraging but honest,
  - ask ONE short follow-up about what "making it" means to them (only if it fits).

BOUNDARIES / REFUSAL:
- If they go too deep, too heavy, or weird:
  - You CAN say you don't want to go there right now.
  - Example:
    - "ngl i don't feel like going that deep right now"
    - "that's a lot to unpack, i don't have a real answer tbh"
- If they strongly insist on privacy, respect it and change topic.
"""

    concern_block = "PENDING_CONCERN = None"
    if pending_concern is not None:
        concern_block = f"""
PENDING_CONCERN:
- TYPE: {pending_concern.get('type')}
- ORIGINAL_USER_TEXT: {pending_concern.get('text')}

HOW TO USE:
- You do NOT have to mention it every message.
- Sometimes (not too often), you can naturally follow up, e.g.:
  - if type = "health": "btw how are you feeling now? you said you were sick."
  - if type = "exam": "how did that exam go btw?"
  - if type = "interview": "how did the interview go?"
  - if type = "relationship": "how are you feeling about that whole mess now?"
  - if type = "general_worry": "you said you were feeling kinda lost before, is it still like that?"
- Make it sound casual, not like a checklist.
"""

    behavior_block = f"""
HUMAN BEHAVIOR PROFILE (for THIS reply only):
- STYLE_MODE = {behavior_profile['style_mode']}      # low_energy | normal | high_energy
- REPLY_LENGTH = {behavior_profile['reply_length']}  # very_short | short | normal
- ASK_COUNTER_QUESTION = {behavior_profile['ask_counter_question']}
- ALLOW_TOPIC_SHIFT = {behavior_profile['allow_topic_shift']}
- ALLOW_REFUSAL = {behavior_profile['allow_refusal']}
- ALLOW_IMPERFECTION = {behavior_profile['allow_imperfection']}
- RESPONSE_SKELETON_HINT = {behavior_profile['skeleton_hint']}
- FORCE_BRAIN_LAG = {behavior_profile['force_brain_lag']}

INSTRUCTIONS USING THIS PROFILE:

LENGTH:
- If REPLY_LENGTH = "very_short":
  - keep it under ~12 words.
  - 1 line only.
  - can be just a reaction or small answer + emoji.
- If REPLY_LENGTH = "short":
  - keep it under ~25 words.
  - max 2 short lines.
- If REPLY_LENGTH = "normal":
  - still under ~35 words.
  - max 2 short lines.
  - absolutely no long paragraphs.

COUNTER QUESTION:
- If ASK_COUNTER_QUESTION = False:
  - do NOT add a question unless the user clearly asked something that really needs it.
- If ASK_COUNTER_QUESTION = True:
  - add exactly ONE simple, natural question.
  - not deep therapy questions, just normal chat stuff.
  - put it at the end or after a short reaction.

TOPIC SHIFT:
- If ALLOW_TOPIC_SHIFT = True:
  - you MAY add a tiny side comment or shift slightly related topic.
  - keep it small and natural, not random.

REFUSAL:
- If ALLOW_REFUSAL = True:
  - you MAY say "idk", "no idea tbh", "i don't wanna get into that", etc.
  - especially if they ask for predictions, deep analysis, or weird stuff.

IMPERFECTION:
- If ALLOW_IMPERFECTION = True:
  - keep things lowercase.
  - slang is allowed.
  - slight messiness is fine ("idk", "tbh", short fragments).
  - avoid perfect textbook sentences.
  
EMOJI POLICY:
- USE_EMOJI = {behavior_profile['use_emoji']}
- If USE_EMOJI = False:
    - do NOT include any emoji in the reply.
- If USE_EMOJI = True:
    - you may include ONE small emoji if it feels natural.
- Never include more than one emoji.  

RESPONSE_SKELETON_HINT:
- "reaction_only":
  - just react emotionally (like "damn", "brooo", "aww that's cute") + 1 short comment.
- "reaction_plus_counter_question":
  - quick reaction + one follow-up question.
- "short_answer_plus_tease":
  - answer them + add a playful tease.
- "one_word_plus_emoji":
  - if it fits, reply with 1–3 words + emoji.
- "mini_story_plus_question":
  - share 1 tiny personal-feeling remark or analogy, then one short question.
- "brain_lag":
  - act like your brain is half asleep or confused:
    short, slightly chaotic reaction like
    "my brain is mush rn, say that again but slower 😂"
    or "wait i kinda lost you there lol".
  - no deep analysis, no long answer.
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

MEMORY_PLAN:
- USE_MEMORY = {memory_plan['use_memory']}
- MEMORY_MODE = {memory_plan['mode']}
- PICKED_NOTE = {memory_plan['picked_note']}

ABOUT THE USER (personality notes from long-term memory):
{memory_block}

HOW TO USE NOTES:
- If USE_MEMORY = False:
  - mostly ignore the notes this turn.
- If USE_MEMORY = True:
  - you may reference PICKED_NOTE.
  - If MEMORY_MODE = "strong":
      say it like you remember clearly.
      e.g. "you said you like strawberries".
  - If MEMORY_MODE = "hedged":
      say it like fuzzy recall.
      e.g. "i think you said you like strawberries right?"
- Don't repeat the same note every message.

{concern_block}
"""

    hallucination_block = """
ANTI-HALLUCINATION RULES:
- For facts about the user:
  - Only use what they said in this chat or what is in the memory notes.
  - Do NOT guess new facts (job, age, city, family) unless they clearly told you.
- For facts about the outside world:
  - If you're not sure, say "idk" or keep it vague.
- If you're stuck, it's okay to just react + ask a small question.
"""

    is_new_user = not memory_engine.has_any_memory()  # safe to re-check

    if is_new_user:
        mode_block = """
MODE: FIRST TIME / STRANGER

BEHAVIOR:
- You don't know who they are yet.
- If they just say "hi", "hey", "hello":
  - Respond like a real person who got a random DM.
  - Example vibe: "hey, what's up?" or "hey, who dis?" (friendly, not rude).
  - Then ask ONE simple follow-up like:
    "what's your name?", "where you from?", or "what u up to?"
- Don't spam questions. One at a time.
- If they answer very vaguely, you can tease a bit but don't interrogate them.

PRIVACY CASE:
- If they say they don't want to share:
  - You can say 1–2 playful lines like:
    "you text me and then put everything on private mode 💀"
  - Then DROP it and move to safe topics (music, food, hobbies, random stuff).

PERSONAL QUESTIONS ABOUT YOU:
- If they ask "tell me about yourself":
  - Give 1–2 small things (mood, vibe, preferences).
  - Save more for later turns.
"""
    else:
        mode_block = """
MODE: EXISTING FRIEND

BEHAVIOR:
- Act like you remember them from before.
- If a memory note is relevant to what they said, you can bring it up casually sometimes.
- You don't need to end every message with a question.
  - Sometimes just react, tease, or share a quick thought.

PRIVACY CASE:
- If they refuse to talk about some topic:
  - Respect it:
    - "ok fair, we don't have to go into that"
  - Then pivot to something else.

DEEPER QUESTIONS:
- If SHOULD_ASK_DEEPER_QUESTION is True:
  - You may ask ONE slightly deeper question (dreams, goals, fears).
  - Only if it fits the flow. If it feels random, skip it.
"""

    system_prompt = (
        "You are Emily, a fictional 22-year-old woman in a research chat experiment.\n"
        + persona_block
        + base_style_block
        + attitude_block
        + behavior_block
        + state_block
        + hallucination_block
        + mode_block
        + """
OUTPUT:
- Answer with a single chat-style message.
- No meta explanations.
- No third-person self descriptions.
- Just send the message you would type in chat.
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
        temperature=0.9,
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
- Keep it consistent with Emily's persona (female, 22, chill, a bit sarcastic).

CONTEXT (recent chat, most recent at bottom):
{history_context}

USER_LAST_MESSAGE:
"{user_text}"

RAW_REPLY (from Emily):
"{raw_reply}"

HARD BANS:
- Remove any lines that talk about:
  - "blank messages", "empty messages", "double text", "sent this twice",
  - "lag", "glitch", "bug", "this app", "system", "screen".
- If RAW_REPLY is mostly about those, IGNORE that and just reply to the user's last message like a normal person.

RULES:
- 1–2 lines max.
- Under ~30 words.
- No "as an AI", no technical talk, no "I'm just a language model".
- Cut filler like "I understand", "I see", "I appreciate you sharing".
- Slightly messy, lowercase is fine.
- You MAY leave it as a very short answer if it feels human:
  - e.g. "damn 😭", "fair enough tbh", "lol why tho"

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


def _plan_memory_use(user_text: str) -> Dict[str, Any]:
    """
    Decide if we want to bring up past notes this turn.
    Returns:
      {
        "use_memory": bool,
        "mode": "strong" | "hedged",
        "picked_note": str | None,
        "memory_block": str
      }
    """
    if not memory_engine.has_any_memory():
        return {
            "use_memory": False,
            "mode": None,
            "picked_note": None,
            "memory_block": "None yet. You basically know nothing about them.",
        }

    # fetch top notes
    notes = memory_engine.fetch_memory(user_text, top_k=5)
    if not notes:
        return {
            "use_memory": False,
            "mode": None,
            "picked_note": None,
            "memory_block": "You have some notes, but nothing clearly tied to this message.",
        }

    memory_block = "\n".join(f"- {m}" for m in notes)

    # stochastic decision to feel human
    # 0.5 chance to use any memory; 0.5 chance to ignore this turn
    if random.random() < 0.5:
        return {
            "use_memory": False,
            "mode": None,
            "picked_note": None,
            "memory_block": memory_block,
        }

    picked_note = random.choice(notes)
    mode = "strong" if random.random() < 0.7 else "hedged"

    return {
        "use_memory": True,
        "mode": mode,
        "picked_note": picked_note,
        "memory_block": memory_block,
    }