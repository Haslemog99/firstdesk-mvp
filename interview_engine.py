"""
Interview question generation, answer evaluation, and scoring.

Two modes:
  - standalone: questions come from the JD only
  - post_sim:   questions also reference what the candidate did in the simulation

Designed to be additive to the FirstDesk MVP — no existing code is mutated.
The engine takes the same `llm` (ChatFireworks) and Mongo collections that
app.py already wires up.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger("firstdesk.interview")


# ──────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────

QUESTION_GEN_SYSTEM = """You are an expert interview question generator.
Given a job description, produce realistic interview rounds with questions.

Rules:
- Phone screen: 4-5 questions about motivation, culture fit, salary, availability.
- Technical / hiring-manager round: 4-5 questions about domain knowledge,
  problem-solving, tools, and behaviour under pressure.
- If post_sim mode: at least 2 questions per round MUST reference specific
  actions from the simulation summary, e.g. "I see you escalated the
  client query to your manager — walk me through that decision."
- Include a `follow_up` for every question — used when the answer is vague.
- Make questions feel natural, not robotic. Vary difficulty within each round.
- Use UK spelling. No emojis. JSON only — no commentary, no markdown fences.
"""

QUESTION_GEN_USER_TEMPLATE = """Mode: {mode}
Number of rounds: {num_rounds}
Role title: {role_title}
Job description:
\"\"\"
{job_description}
\"\"\"
{sim_block}
Return ONLY a single JSON object with this exact shape:

{{
  "role_title": "string",
  "company": "string",
  "rounds": [
    {{
      "round": 1,
      "type": "phone_screen",
      "persona": {{"name": "Realistic Name", "role": "HR Recruiter", "tone": "warm, conversational"}},
      "questions": [
        {{
          "id": "q1",
          "text": "string",
          "category": "motivation|technical|behavioural|leadership|situational",
          "follow_up": "string",
          "evaluation_criteria": "string",
          "references_simulation": false,
          "simulation_reference": null
        }}
      ]
    }}
  ]
}}
"""

ANSWER_EVAL_SYSTEM = """You are an expert interview evaluator.
You return a single JSON object scoring one answer. JSON only — no commentary.

Scoring guide (content_score):
  9-10: Exceptional — specific, structured, impactful.
  7-8:  Strong — good content, clear structure, minor improvements possible.
  5-6:  Adequate — answered but lacked depth or specificity.
  3-4:  Weak — vague, unfocused, missing key elements.
  1-2:  Poor — didn't answer the question, unprofessional, or factually wrong.
"""

ANSWER_EVAL_USER_TEMPLATE = """Question: {question_text}
Category: {category}
Evaluation criteria: {criteria}
Answer transcript:
\"\"\"
{transcript}
\"\"\"
Speaking duration (seconds): {duration_sec}
Filler words detected: {filler_count}
Voice signals — confidence: {confidence}, anxiety: {anxiety}
{sim_block}
Return ONLY:

{{
  "content_score": 0.0,
  "strengths": ["..."],
  "improvements": ["..."],
  "consistency_flag": null,
  "coaching_tip": "..."
}}
"""

ROUND_FEEDBACK_SYSTEM = """You produce a one-paragraph feedback summary for an
interview round. JSON only."""

ROUND_FEEDBACK_USER_TEMPLATE = """Round {round_num} ({round_type}) just finished.
Persona: {persona_name} ({persona_role}).
Per-question evaluations (JSON):
{evals_json}

Return ONLY:

{{
  "round_score": 0,
  "round_feedback": "two or three sentences"
}}
"""


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_FILLER_RE = re.compile(
    r"\b(um+|uh+|er+|ah+|like|you know|sort of|kind of|basically|literally|i mean)\b",
    re.IGNORECASE,
)

# Hume-ish dimensions to surface in our compact emotion summary. We don't
# require Hume — these are also the keys our text-mode UI synthesises.
_VOICE_KEYS = ("confidence", "anxiety", "enthusiasm", "dominant_emotion")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


def _parse_json_envelope(text: str) -> Optional[dict]:
    """Forgiving JSON extraction — strips ```fences``` and trailing commas."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", s[start:end + 1])
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def _count_fillers(transcript: str) -> int:
    if not transcript:
        return 0
    return len(_FILLER_RE.findall(transcript))


def _speaking_pace_wpm(transcript: str, duration_sec: float) -> int:
    """Rough WPM. duration_sec ≤ 0 → estimate from word count alone."""
    words = len((transcript or "").split())
    if duration_sec and duration_sec > 0:
        return int(round(words / max(duration_sec, 1) * 60))
    return int(round(words * 2.4))  # ~25s assumption


def _safe_float(v: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


# ──────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────

@dataclass
class _SimulationSummary:
    """Compact, prompt-friendly summary of what the candidate did in a sim."""
    career: Optional[str]
    company: Optional[str]
    cumulative_score: int
    completed_tasks: list[str]
    flagged_actions: list[str]
    standout_actions: list[str]

    def to_prompt(self) -> str:
        if not (self.completed_tasks or self.flagged_actions or self.standout_actions):
            return ""
        lines = ["Simulation summary:"]
        if self.career:
            lines.append(f"- Career: {self.career}; Company: {self.company or 'n/a'}")
        lines.append(f"- Cumulative score: {self.cumulative_score}/100")
        if self.completed_tasks:
            lines.append("- Completed tasks: " + "; ".join(self.completed_tasks[:8]))
        if self.standout_actions:
            lines.append("- Standout actions: " + "; ".join(self.standout_actions[:6]))
        if self.flagged_actions:
            lines.append("- Manager-flagged mistakes: " + "; ".join(self.flagged_actions[:6]))
        return "\n".join(lines)


class InterviewEngine:
    """Stateless coordinator. Each method takes the data it needs."""

    def __init__(self, llm: Any, interviews_col: Any, sessions_col: Any):
        self.llm = llm
        self.interviews_col = interviews_col
        self.sessions_col = sessions_col

    # ── Public API ────────────────────────────────────────

    def generate_interview(
        self,
        *,
        job_description: str,
        mode: str = "standalone",
        role_title: str = "",
        simulation_summary: Optional[_SimulationSummary] = None,
        num_rounds: int = 2,
    ) -> dict:
        """Generate an interview. Returns the JSON envelope (rounds)."""
        sim_block = ""
        if mode == "post_sim" and simulation_summary is not None:
            sim_block = "\n" + simulation_summary.to_prompt() + "\n"

        user = QUESTION_GEN_USER_TEMPLATE.format(
            mode=mode,
            num_rounds=max(1, min(int(num_rounds or 2), 4)),
            role_title=(role_title or "")[:120],
            job_description=(job_description or "")[:6000],
            sim_block=sim_block,
        )
        try:
            resp = self.llm.invoke([("system", QUESTION_GEN_SYSTEM), ("human", user)])
            raw = getattr(resp, "content", None) or str(resp)
        except Exception as ex:
            log.exception("interview.generate.llm_failed kind=%s", type(ex).__name__)
            return self._fallback_envelope(role_title or "the role", num_rounds)

        env = _parse_json_envelope(raw)
        if not isinstance(env, dict) or not env.get("rounds"):
            log.warning("interview.generate.parse_failed preview=%r", (raw or "")[:240])
            return self._fallback_envelope(role_title or "the role", num_rounds)

        return self._normalise_envelope(env, mode=mode)

    def evaluate_answer(
        self,
        *,
        question: dict,
        transcript: str,
        emotion_data: Optional[dict] = None,
        body_language: Optional[dict] = None,
        simulation_summary: Optional[_SimulationSummary] = None,
        duration_sec: Optional[float] = None,
    ) -> dict:
        """Score a single answer. Returns the per-question record (mergable)."""
        emo = emotion_data or {}
        confidence = _safe_float(emo.get("confidence"), default=0.5)
        anxiety = _safe_float(emo.get("anxiety"), default=0.3)
        enthusiasm = _safe_float(emo.get("enthusiasm"), default=0.5)
        dominant = str(emo.get("dominant_emotion") or "Neutral")[:32]

        fillers = int(emo.get("filler_words") if emo.get("filler_words") is not None
                      else _count_fillers(transcript))
        wpm = int(emo.get("speaking_pace_wpm") or _speaking_pace_wpm(transcript, duration_sec or 0))

        sim_block = ""
        if simulation_summary is not None and question.get("references_simulation"):
            sim_block = "\n" + simulation_summary.to_prompt() + "\n"

        user_prompt = ANSWER_EVAL_USER_TEMPLATE.format(
            question_text=question.get("text", "")[:400],
            category=question.get("category", "general"),
            criteria=(question.get("evaluation_criteria") or "")[:240],
            transcript=(transcript or "").strip()[:4000] or "(no answer given)",
            duration_sec=int(duration_sec or 0),
            filler_count=fillers,
            confidence=f"{confidence:.2f}",
            anxiety=f"{anxiety:.2f}",
            sim_block=sim_block,
        )

        score, strengths, improvements, tip, flag = 5.0, [], [], "", None
        try:
            resp = self.llm.invoke([("system", ANSWER_EVAL_SYSTEM), ("human", user_prompt)])
            raw = getattr(resp, "content", None) or str(resp)
            env = _parse_json_envelope(raw) or {}
            score = max(0.0, min(10.0, float(env.get("content_score", 5.0))))
            strengths = [str(s)[:200] for s in (env.get("strengths") or [])][:5]
            improvements = [str(s)[:200] for s in (env.get("improvements") or [])][:5]
            tip = str(env.get("coaching_tip") or "")[:300]
            flag = env.get("consistency_flag")
        except Exception as ex:
            log.warning("interview.eval.llm_failed kind=%s — using heuristic", type(ex).__name__)
            score = self._heuristic_score(transcript, fillers)
            improvements = ["Couldn't reach the evaluator service — heuristic score shown."]

        # Weight content with delivery signals to derive a single per-question rating.
        delivery_penalty = 0.0
        if fillers >= 5:
            delivery_penalty += 0.5
        if wpm and (wpm < 90 or wpm > 200):
            delivery_penalty += 0.5
        if anxiety > 0.7:
            delivery_penalty += 0.4
        adjusted = max(0.0, min(10.0, score - delivery_penalty))

        return {
            "answer_transcript": (transcript or "")[:6000],
            "answer_duration_sec": int(duration_sec or 0),
            "emotion": {
                "confidence": round(confidence, 3),
                "anxiety": round(anxiety, 3),
                "enthusiasm": round(enthusiasm, 3),
                "filler_words": fillers,
                "speaking_pace_wpm": wpm,
                "dominant_emotion": dominant,
            },
            "body_language": body_language or None,
            "content_score": round(adjusted, 2),
            "raw_content_score": round(score, 2),
            "content_feedback": tip or (improvements[0] if improvements else ""),
            "strengths": strengths,
            "improvements": improvements,
            "coaching_tip": tip,
            "consistency_flag": flag,
            "answered_at": datetime.utcnow().isoformat(),
        }

    def generate_round_feedback(self, round_data: dict) -> dict:
        """Average + LLM-summarise a finished round."""
        evals = []
        for q in (round_data.get("questions") or []):
            if q.get("content_score") is None:
                continue
            evals.append({
                "q": q.get("text", "")[:160],
                "score": q.get("content_score"),
                "strengths": q.get("strengths", []),
                "improvements": q.get("improvements", []),
            })

        if not evals:
            return {"round_score": 0, "round_feedback": "No answers recorded for this round."}

        avg = sum(e["score"] for e in evals) / max(1, len(evals))
        round_score = int(round(avg * 10))  # 0-10 → 0-100

        # Best-effort narrative summary
        feedback = f"Average score {avg:.1f}/10 across {len(evals)} answers."
        try:
            resp = self.llm.invoke([
                ("system", ROUND_FEEDBACK_SYSTEM),
                ("human", ROUND_FEEDBACK_USER_TEMPLATE.format(
                    round_num=round_data.get("round", 1),
                    round_type=round_data.get("type", "round"),
                    persona_name=(round_data.get("persona") or {}).get("name", ""),
                    persona_role=(round_data.get("persona") or {}).get("role", ""),
                    evals_json=json.dumps(evals, ensure_ascii=False)[:3000],
                )),
            ])
            raw = getattr(resp, "content", None) or str(resp)
            env = _parse_json_envelope(raw) or {}
            if isinstance(env.get("round_score"), (int, float)):
                round_score = int(round(env["round_score"]))
            if env.get("round_feedback"):
                feedback = str(env["round_feedback"])[:600]
        except Exception:
            pass

        return {"round_score": max(0, min(100, round_score)), "round_feedback": feedback}

    def calculate_scores(self, interview_data: dict) -> dict:
        """Final score breakdown.

        Standalone: content (50%) + delivery (30%) + body language (20%)
        Post-sim:   simulation (50%) + content (30%) + delivery (20%)
        """
        all_q = [q for r in (interview_data.get("rounds") or [])
                 for q in (r.get("questions") or [])
                 if q.get("content_score") is not None]
        if not all_q:
            return {"content": 0, "delivery": 0, "body_language": 0,
                    "interview_score": 0, "simulation_score": None,
                    "employability_score": 0}

        content_avg = sum(q["content_score"] for q in all_q) / len(all_q)  # 0-10
        content_pct = int(round(content_avg * 10))                          # 0-100

        delivery_pct = self._delivery_score(all_q)
        body_pct = self._body_score(all_q)

        sim_pct = interview_data.get("scores", {}).get("simulation_score") if interview_data.get("scores") else None
        if sim_pct is None and interview_data.get("simulation_score") is not None:
            sim_pct = int(interview_data["simulation_score"])

        if interview_data.get("mode") == "post_sim" and sim_pct is not None:
            employability = int(round(sim_pct * 0.5 + content_pct * 0.3 + delivery_pct * 0.2))
        else:
            employability = int(round(content_pct * 0.5 + delivery_pct * 0.3 + body_pct * 0.2))

        return {
            "content": content_pct,
            "delivery": delivery_pct,
            "body_language": body_pct,
            "interview_score": int(round(content_pct * 0.6 + delivery_pct * 0.4)),
            "simulation_score": sim_pct,
            "employability_score": max(0, min(100, employability)),
        }

    # ── Persistence helpers ───────────────────────────────

    def create_record(
        self,
        *,
        user_id: str,
        envelope: dict,
        job_description: str,
        mode: str,
        simulation_id: Optional[str],
        role_title: str,
    ) -> dict:
        doc = {
            "id": _new_id("int"),
            "user_id": user_id,
            "mode": mode,
            "simulation_id": simulation_id,
            "job_description": (job_description or "")[:8000],
            "role_title": (envelope.get("role_title") or role_title or "")[:160],
            "company": (envelope.get("company") or "")[:160],
            "status": "in_progress",
            "current_round": 1,
            "rounds": envelope.get("rounds") or [],
            "scores": {"content": 0, "delivery": 0, "body_language": 0,
                       "interview_score": 0, "simulation_score": None,
                       "employability_score": 0},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        self.interviews_col.insert_one(doc)
        return doc

    def load(self, interview_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        q: dict = {"id": interview_id}
        if user_id:
            q["user_id"] = user_id
        return self.interviews_col.find_one(q, {"_id": 0})

    def save_answer(
        self,
        *,
        interview_id: str,
        user_id: str,
        round_num: int,
        question_id: str,
        evaluation: dict,
    ) -> Optional[dict]:
        """Merge evaluation into the matching question, persist, return the
        updated interview doc."""
        doc = self.load(interview_id, user_id)
        if not doc:
            return None
        rounds = doc.get("rounds") or []
        target_round = next((r for r in rounds if r.get("round") == round_num), None)
        if not target_round:
            return None
        q = next((qq for qq in (target_round.get("questions") or [])
                  if qq.get("id") == question_id), None)
        if not q:
            return None
        q.update(evaluation)

        # If every question in the round has a score, finalise the round.
        if all(qq.get("content_score") is not None for qq in target_round["questions"]):
            rd = self.generate_round_feedback(target_round)
            target_round["round_score"] = rd["round_score"]
            target_round["round_feedback"] = rd["round_feedback"]

        # Recompute scores so the FE always sees the latest aggregate.
        doc["rounds"] = rounds
        doc["scores"] = self.calculate_scores(doc)

        self.interviews_col.update_one(
            {"id": interview_id, "user_id": user_id},
            {"$set": {"rounds": rounds, "scores": doc["scores"]}},
        )
        return doc

    def complete(self, interview_id: str, user_id: str) -> Optional[dict]:
        doc = self.load(interview_id, user_id)
        if not doc:
            return None
        scores = self.calculate_scores(doc)
        self.interviews_col.update_one(
            {"id": interview_id, "user_id": user_id},
            {"$set": {"scores": scores, "status": "completed",
                      "completed_at": datetime.utcnow().isoformat()}},
        )
        doc["scores"] = scores
        doc["status"] = "completed"
        return doc

    # ── Internal ──────────────────────────────────────────

    def _delivery_score(self, all_q: list[dict]) -> int:
        """0-100 from voice signals + filler/pace penalties."""
        if not all_q:
            return 0
        score = 0.0
        for q in all_q:
            emo = q.get("emotion") or {}
            confidence = float(emo.get("confidence") or 0.5)
            anxiety = float(emo.get("anxiety") or 0.3)
            enthusiasm = float(emo.get("enthusiasm") or 0.5)
            fillers = int(emo.get("filler_words") or 0)
            wpm = int(emo.get("speaking_pace_wpm") or 0)

            base = (confidence * 0.55 + enthusiasm * 0.35 + (1 - anxiety) * 0.10) * 100
            if fillers >= 5:
                base -= min(15, fillers * 2)
            if wpm and (wpm < 90 or wpm > 200):
                base -= 8
            score += max(0, min(100, base))
        return int(round(score / len(all_q)))

    def _body_score(self, all_q: list[dict]) -> int:
        readings = [q.get("body_language") for q in all_q if q.get("body_language")]
        if not readings:
            return 0
        n = 0
        total = 0
        for b in readings:
            ec = float(b.get("eye_contact_pct") or 0)
            posture = float(b.get("posture_score") or 0)
            gestures = float(b.get("gesture_count") or 0)
            # Cap gesture contribution — moderate gestures are best.
            g_score = max(0, min(100, 100 - abs(gestures - 8) * 6))
            total += ec * 0.45 + posture * 0.40 + g_score * 0.15
            n += 1
        return int(round(total / n))

    def _heuristic_score(self, transcript: str, fillers: int) -> float:
        """Used only when the LLM is unreachable."""
        words = len((transcript or "").split())
        if words < 8:
            return 2.5
        base = 6.5 if words >= 40 else 5.0
        base -= min(2.0, fillers * 0.4)
        return max(1.0, min(9.0, base))

    def _normalise_envelope(self, env: dict, *, mode: str) -> dict:
        """Trim to safe limits + ensure each question has a stable id."""
        rounds_out = []
        for i, r in enumerate((env.get("rounds") or [])[:4]):
            rtype = str(r.get("type") or "round")[:32]
            persona = r.get("persona") or {}
            qs_out = []
            for j, q in enumerate((r.get("questions") or [])[:8]):
                qs_out.append({
                    "id": str(q.get("id") or f"q{j+1}")[:8],
                    "text": str(q.get("text") or "")[:600],
                    "category": str(q.get("category") or "general")[:24],
                    "follow_up": str(q.get("follow_up") or "")[:300],
                    "evaluation_criteria": str(q.get("evaluation_criteria") or "")[:300],
                    "references_simulation": bool(q.get("references_simulation")) if mode == "post_sim" else False,
                    "simulation_reference": (q.get("simulation_reference") or None) if mode == "post_sim" else None,
                    "answer_transcript": None,
                    "answer_duration_sec": 0,
                    "emotion": None,
                    "body_language": None,
                    "content_score": None,
                    "content_feedback": "",
                })
            if not qs_out:
                continue
            rounds_out.append({
                "round": int(r.get("round") or (i + 1)),
                "type": rtype,
                "persona": {
                    "name": str(persona.get("name") or "")[:80],
                    "role": str(persona.get("role") or "")[:80],
                    "tone": str(persona.get("tone") or "")[:80],
                },
                "questions": qs_out,
                "round_score": None,
                "round_feedback": "",
            })

        return {
            "role_title": str(env.get("role_title") or "")[:160],
            "company": str(env.get("company") or "")[:160],
            "rounds": rounds_out,
        }

    def _fallback_envelope(self, role_title: str, num_rounds: int) -> dict:
        """Generic skeleton when the LLM is unavailable. Ensures the system
        still works for demos / smoke tests."""
        n = max(1, min(int(num_rounds or 2), 4))
        rounds = [
            {
                "round": 1,
                "type": "phone_screen",
                "persona": {"name": "Emma Richards", "role": "HR Recruiter", "tone": "warm, conversational"},
                "questions": [
                    {"id": "q1", "text": f"Tell me about yourself and why you're interested in the {role_title} role.",
                     "category": "motivation", "follow_up": "What specifically drew you to our team?",
                     "evaluation_criteria": "Specific motivation tied to the role and company.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                    {"id": "q2", "text": "Tell me about a time you had to manage competing priorities.",
                     "category": "behavioural", "follow_up": "What was the impact?",
                     "evaluation_criteria": "STAR structure; concrete actions; measurable outcome.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                    {"id": "q3", "text": "What are your salary expectations and notice period?",
                     "category": "situational", "follow_up": "Is that flexible?",
                     "evaluation_criteria": "Clear range; aware of market; professional framing.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                ],
                "round_score": None, "round_feedback": "",
            },
        ]
        if n >= 2:
            rounds.append({
                "round": 2,
                "type": "technical",
                "persona": {"name": "James Morrison", "role": "Hiring Manager", "tone": "serious, probing"},
                "questions": [
                    {"id": "q1", "text": f"Walk me through how you'd approach your first 30 days as a {role_title}.",
                     "category": "situational", "follow_up": "What would you measure success by?",
                     "evaluation_criteria": "Structured plan; learning + delivery balance.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                    {"id": "q2", "text": "Describe a hard problem you solved recently. What made it hard?",
                     "category": "technical", "follow_up": "How did you validate the fix?",
                     "evaluation_criteria": "Concrete problem; reasoning; verification step.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                    {"id": "q3", "text": "Tell me about a time you disagreed with a manager. How did you handle it?",
                     "category": "behavioural", "follow_up": "What did you learn?",
                     "evaluation_criteria": "Maturity; respectful disagreement; outcome.",
                     "references_simulation": False, "simulation_reference": None,
                     "answer_transcript": None, "answer_duration_sec": 0, "emotion": None,
                     "body_language": None, "content_score": None, "content_feedback": ""},
                ],
                "round_score": None, "round_feedback": "",
            })
        return {
            "role_title": role_title[:160],
            "company": "Generic Co",
            "rounds": rounds[:n],
        }


# ──────────────────────────────────────────────────────────────────
# Bridge from a sim_id → SimulationSummary (used by post_sim mode)
# ──────────────────────────────────────────────────────────────────

def build_simulation_summary(
    *,
    sessions_col: Any,
    tasks_col: Any,
    actions_col: Any,
    reputation_col: Any,
    progress_col: Any,
    user_id: str,
    simulation_id: str,
) -> Optional[_SimulationSummary]:
    """Best-effort summary for post_sim mode. Returns None if the sim is gone."""
    try:
        prog = progress_col.find_one({"user_id": user_id}, {"_id": 0}) or {}
        career = prog.get("current_career")
        score = int(prog.get("cumulative_score") or 0)

        completed = []
        for t in tasks_col.find(
            {"simulation_id": simulation_id, "status": "done"},
            {"_id": 0, "title": 1, "awarded_points": 1, "points": 1},
        ).limit(20):
            completed.append(f"{t.get('title','')} (+{t.get('awarded_points',t.get('points',0))}pts)")

        flagged = []
        for a in actions_col.find(
            {"simulation_id": simulation_id, "flagged": True},
            {"_id": 0, "summary": 1, "action": 1},
        ).limit(20):
            flagged.append(a.get("summary") or a.get("action", ""))

        standout = []
        for a in reputation_col.find(
            {"simulation_id": simulation_id},
            {"_id": 0, "summary": 1, "delta": 1},
        ).sort("delta", -1).limit(8):
            if (a.get("delta") or 0) > 0:
                standout.append(a.get("summary", ""))

        company = (sessions_col.find_one({"simulation_id": simulation_id}, {"_id": 0, "company": 1}) or {}).get("company")

        return _SimulationSummary(
            career=career,
            company=company,
            cumulative_score=score,
            completed_tasks=[c for c in completed if c],
            flagged_actions=[f for f in flagged if f],
            standout_actions=[s for s in standout if s],
        )
    except Exception as ex:
        log.warning("interview.build_summary_failed kind=%s", type(ex).__name__)
        return None
