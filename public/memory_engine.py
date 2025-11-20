# memory_engine.py
from __future__ import annotations
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import faiss
import numpy as np
from openai import OpenAI

# ========= OpenAI + Embedding config =========

MEMORY_FILE = Path("memory.json")
FAISS_FILE = Path("memory.index")
EMBED_DIM = 1536
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()


# ========= Structured Memory Item =========

@dataclass
class MemoryItem:
    """
    A single memory fact about the user.

    text:        human-readable note (e.g. "user likes horror movies")
    type:        high-level category:
                  - identity, preference, goal, worry, relationship,
                    story, personality_trait, mood_pattern, other, legacy
    tags:        list of short keywords (e.g. ["movies", "horror"])
    importance:  1–3 (3 = very central / repeated / emotionally loaded)
    source_msg:  original user message snippet
    created_at:  optional turn index or timestamp (you can wire this later)
    """
    text: str
    type: str = "generic"
    tags: List[str] | None = None
    importance: int = 1
    source_msg: str | None = None
    created_at: int | None = None


# FAISS index + in-memory memory items
faiss_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(EMBED_DIM)
memory_items: List[MemoryItem] = []


# ========= Core I/O & Embeddings =========

def init_memory():
    """
    Load FAISS index + memory.json if present, otherwise start fresh.
    Call this once at app startup.
    """
    global faiss_index, memory_items

    if MEMORY_FILE.exists() and FAISS_FILE.exists():
        print("🔄 Loading existing memory...")

        # ---- Load structured or legacy memory.json ----
        with MEMORY_FILE.open("r", encoding="utf8") as f:
            raw = json.load(f)

        loaded_items: List[MemoryItem] = []
        if isinstance(raw, list):
            for entry in raw:
                # legacy: list of strings
                if isinstance(entry, str):
                    loaded_items.append(
                        MemoryItem(
                            text=entry,
                            type="legacy",
                            tags=[],
                            importance=1,
                            source_msg=None,
                            created_at=None,
                        )
                    )
                # new: list of dicts with "text" at minimum
                elif isinstance(entry, dict) and "text" in entry:
                    loaded_items.append(
                        MemoryItem(
                            text=entry.get("text", "").strip(),
                            type=entry.get("type", "generic"),
                            tags=entry.get("tags") or [],
                            importance=int(entry.get("importance", 1)),
                            source_msg=entry.get("source_msg"),
                            created_at=entry.get("created_at"),
                        )
                    )

        memory_items = [m for m in loaded_items if m.text]

        # ---- Load FAISS index ----
        faiss_index = faiss.read_index(str(FAISS_FILE))

        if faiss_index.ntotal != len(memory_items):
            print(
                f"⚠️ Index/text size mismatch (index={faiss_index.ntotal}, "
                f"items={len(memory_items)}). Using the smaller count."
            )
            n = min(faiss_index.ntotal, len(memory_items))
            memory_items = memory_items[:n]
        print(f"✅ Loaded {len(memory_items)} memory items.")
    else:
        print("🆕 Starting fresh memory.")
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        memory_items = []


def _get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")
    if emb.shape[0] != EMBED_DIM:
        raise ValueError(f"Embedding dim mismatch: expected {EMBED_DIM}, got {emb.shape[0]}")
    return emb


def _save_memory():
    """
    Persist current memory_items + FAISS index to disk.
    """
    with MEMORY_FILE.open("w", encoding="utf8") as f:
        json.dump([asdict(m) for m in memory_items], f, ensure_ascii=False, indent=2)
    faiss.write_index(faiss_index, str(FAISS_FILE))
    print("💾 Memory saved to disk.")


# ========= Note Classification =========

def _classify_note(note_text: str) -> tuple[str, List[str], int]:
    """
    Classify a memory note into a type + tags + importance.

    Types (use one of):
      - identity           (where they live, study, work, background)
      - preference         (likes/dislikes, hobbies, tastes)
      - goal               (dreams, ambitions, long-term plans)
      - worry              (ongoing anxieties, fears, problems)
      - relationship       (family/partner/friend roles that matter)
      - story              (memories, repeated narratives, origin stories)
      - personality_trait  (stable traits: overthinking, ambitious, shy, etc.)
      - mood_pattern       (recurring moods: often anxious, often tired)
      - other              (anything that doesn’t fit but is still useful)
      - legacy             (for old unclassified notes)

    Importance:
      - 1 = minor detail
      - 2 = moderately important
      - 3 = central / emotionally loaded / repeated theme
    """
    note_text = (note_text or "").strip()
    if not note_text:
        return "other", [], 1

    system_prompt = """
You are classifying ONE short memory about a user.

Return a compact JSON object with this exact structure:
{
  "type": "...",
  "tags": ["...", "..."],
  "importance": 1
}

Allowed "type" values:
- "identity"
- "preference"
- "goal"
- "worry"
- "relationship"
- "story"
- "personality_trait"
- "mood_pattern"
- "other"

Guidelines:
- "identity": where they live, study, work, or background.
- "preference": likes/dislikes, hobbies, taste in music/food/etc.
- "goal": anything they want to achieve or become.
- "worry": repeated concerns about future, career, money, health, etc.
- "relationship": info about family/partner/friends that matters to them.
- "story": memories or personal stories (especially childhood, key events).
- "personality_trait": stable traits, e.g. overthinks, ambitious, shy, chaotic.
- "mood_pattern": recurring emotional tendencies (e.g. often anxious, often numb).
- "other": if none of the above clearly fits.

"tags":
- Short keywords only, 1–3 words each.
- Derived ONLY from the note text, do not invent new facts.

"importance":
- 1 if it's a small detail.
- 2 if it matters a bit for who they are.
- 3 if it seems central, emotionally heavy, or repeated.

Do NOT explain.
Do NOT add extra fields.
ONLY return the JSON object.
"""

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": note_text},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=msgs,
            temperature=0.0,
            max_completion_tokens=64,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        note_type = str(data.get("type", "other")).strip() or "other"
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if str(t).strip()]
        importance = int(data.get("importance", 1))
        if importance < 1 or importance > 3:
            importance = 1
        return note_type, tags, importance
    except Exception as e:
        print("⚠️ Error in _classify_note:", e)
        return "other", [], 1


# ========= Public Memory Ops =========

def store_memory(
    text: str,
    source_message: Optional[str] = None,
    created_at: Optional[int] = None,
):
    """
    Store a single user note string, with classification.

    Examples:
    - "user likes strawberries"
    - "user is worried about their career"
    - "user has a childhood memory about playing with a ball"
    """
    global memory_items, faiss_index

    text = (text or "").strip()
    if not text:
        return

    # Simple dedup: if exact same text exists, just bump importance.
    for item in memory_items:
        if item.text.lower() == text.lower():
            item.importance = min(3, item.importance + 1)
            print(f"⚠️ Duplicate note, bumping importance: {text}")
            _save_memory()
            return

    note_type, tags, importance = _classify_note(text)

    emb = _get_embedding(text)
    faiss_index.add(np.array([emb]))

    item = MemoryItem(
        text=text,
        type=note_type,
        tags=tags,
        importance=importance,
        source_msg=source_message,
        created_at=created_at,
    )
    memory_items.append(item)
    print(f"✅ Stored memory note: {item.text}  (type={item.type}, importance={item.importance})")
    _save_memory()


def retrieve_related(query: str, top_k: int = 3) -> List[str]:
    """
    Semantic search over stored user notes.
    Returns the "text" field of the most related notes.
    """
    if len(memory_items) == 0:
        return []

    query = (query or "").strip()
    if not query:
        return []

    qvec = _get_embedding(query)
    D, I = faiss_index.search(np.array([qvec]), min(top_k, len(memory_items)))

    result: List[str] = []
    for idx in I[0]:
        if 0 <= idx < len(memory_items):
            result.append(memory_items[idx].text)
    return result


def fetch_memory(query: str, top_k: int = 3) -> List[str]:
    """
    Returns ONLY the user notes strings, e.g.:
    - "user likes strawberries"
    - "user lives in chicago"
    """
    return retrieve_related(query, top_k)


def get_all_memory_notes() -> List[str]:
    """
    For debug / UI panel – only note texts.
    """
    return [m.text for m in memory_items]


def get_structured_memory() -> List[Dict[str, Any]]:
    """
    For debug / UI panel – full structured items.
    """
    return [asdict(m) for m in memory_items]


def get_memory_count() -> int:
    return len(memory_items)


def has_any_memory() -> bool:
    return len(memory_items) > 0


# ========= General user info extractor (multi-dimension) =========

def extract_user_notes(user_text: str) -> List[str]:
    """
    Use GPT to pull out ANY useful, longer-term info about the user:
    - identity (where they live, what they study/do)
    - stable preferences (likes/dislikes, hobbies, taste)
    - goals & dreams (career, life, travel, money, etc.)
    - ongoing worries/problems (career, money, health, relationships, studies, feeling lost)
    - relationships (family, partner, friends) IF they mention them
    - meaningful memories (e.g. "user has a childhood memory about playing with a ball")
    - personality traits or patterns (e.g. overthinks, ambitious, perfectionist)
    - recurring mood patterns (e.g. often anxious about future)

    Examples:
    - "i like strawberries" ->
        user likes strawberries
    - "i'm from chicago" ->
        user is from chicago
    - "im worried about my career... idk what to do" ->
        user is worried about their career path
    - "i met my college friend and we remembered how i always played with my red ball as a kid" ->
        user has a childhood memory about playing with a ball
    - "i always overthink everything" ->
        user tends to overthink things
    - "i want to move to europe someday" ->
        user wants to move to europe someday

    The model MUST NOT guess. If nothing clear, return [].
    """

    system_prompt = """
You log facts about the user from a single message.

Your job:
- Extract up to FIVE short, concrete notes that would be useful later, such as:
  - identity (where they live, what they study or do)
  - stable likes/dislikes (food, hobbies, topics, music, style)
  - long-term goals or dreams (career, life, travel, money, family)
  - ongoing worries / problems (career, money, health, relationships, studies, feeling lost)
  - relationships they mention (e.g. "younger sister", "mom", "boyfriend") ONLY if relevant to them
  - meaningful memories or stories they mention (e.g. childhood, important events)
    For those, write them as: "user has a memory about ..."
  - personality traits that clearly show up (e.g. "user tends to overthink", "user is very ambitious")
  - recurring mood patterns IF stated (e.g. "user often feels anxious about the future")

Format:
- Return each note on its own line.
- Each note MUST be a short sentence starting with "user".
  Examples:
    user likes strawberries
    user lives in chicago
    user is worried about their career path
    user wants to start their own business
    user has a childhood memory about playing with a ball
    user tends to overthink things
    user often feels anxious about their future

Important rules:
- Do NOT invent or guess hidden facts.
- Only use what is explicitly or very clearly implied by the message.
- Do NOT write temporary moods like "user is tired today".
- Do NOT include generic lines like "user sent a message" or "user is talking to emily".
- If there is no clear long-term info, return exactly: NONE
"""

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=msgs,
        temperature=0.0,
        max_completion_tokens=128,
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content or content.upper().startswith("NONE"):
        return []

    notes: List[str] = []
    for line in content.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        if len(line) > 200:
            line = line[:200]
        notes.append(line)

    return notes


def maybe_store_user_message(text: str, msg_index: Optional[int] = None):
    """
    Decide whether to store notes from this message.

    Strategy:
    - Run extractor on MOST messages (except ultra-short ones).
    - Let the model decide what is worth logging.
    - This way things like:
        "im worried about my career"
        "i used to play with a ball as a kid"
        "i always overthink stuff"
      get stored as structured, reusable notes.
    """
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) < 5:
        return

    try:
        notes = extract_user_notes(cleaned)
    except Exception as e:
        print("⚠️ Error in extract_user_notes:", e)
        return

    if not notes:
        return

    for note in notes:
        try:
            store_memory(note, source_message=cleaned, created_at=msg_index)
        except Exception as e:
            print("⚠️ Error storing note:", note, e)


# ========= High-level Profile Summary (for research/UI) =========

def summarize_user_profile(max_notes: int = 80) -> str:
    """
    Build a natural-language summary of "who the user is"
    based on stored memory items.

    This is for:
    - debugging
    - your research paper (showing emergent persona)
    - or a side panel in the UI.
    """
    if not memory_items:
        return "no stable info about the user yet."

    # Sort by importance (3 -> 1) and then truncate
    sorted_items = sorted(
        memory_items,
        key=lambda m: m.importance,
        reverse=True,
    )
    notes = [m.text for m in sorted_items[:max_notes]]

    joined = "\n".join(f"- {n}" for n in notes)

    system_prompt = """
You are summarizing a user's personality and life based on memory notes.

Input:
- A list of short "user ..." facts and stories.

Output:
- A short, human-readable summary (1–3 short paragraphs) describing:
  - who they are (identity/background),
  - what they like/dislike,
  - their goals and worries,
  - their relationships (high-level only),
  - their personality traits and patterns,
  - any recurring themes or stories.

Rules:
- DO NOT add new facts that are not in the notes.
- You may gently compress similar notes, but stay faithful.
- Keep it informal but clear, like you're describing a friend.
"""

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": joined},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=msgs,
            temperature=0.4,
            max_completion_tokens=220,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return summary or "profile summary is empty."
    except Exception as e:
        print("⚠️ Error in summarize_user_profile:", e)
        return "profile summary unavailable due to an error."