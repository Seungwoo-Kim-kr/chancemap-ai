"""
Rule-based fit scoring for ChanceMap AI.

Scoring is intentionally kept deterministic (rule-based) for stability and
explainability. Graph traversal provides the skill/document details;
this module converts those details into a structured FitResult.

Score weights:
  major_fit       20 pts  — major/field alignment
  skill_fit       30 pts  — required (70%) + preferred (30%) skill match
  experience_fit  20 pts  — project keyword overlap with opportunity criteria
  eligibility_fit 20 pts  — graduation status, international student policy
  document_fit    10 pts  — ratio of required documents already held
"""

from dataclasses import dataclass, field


@dataclass
class FitResult:
    total_score: int
    verdict: str
    breakdown: dict[str, int]
    matched: list[str] = field(default_factory=list)
    needs_confirmation: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


WEIGHTS = {
    "major_fit":       20,
    "skill_fit":       30,
    "experience_fit":  20,
    "eligibility_fit": 20,
    "document_fit":    10,
}

_VERDICTS = [
    (80, "High fit — recommended"),
    (60, "Conditional — verify requirements"),
    (40, "Needs improvement before applying"),
    (0,  "Low fit at current stage"),
]


def _verdict(score: int) -> str:
    for threshold, label in _VERDICTS:
        if score >= threshold:
            return label
    return _VERDICTS[-1][1]


def calculate_fit(
    user: dict,
    opportunity: dict,
    skill_match: dict | None = None,
) -> FitResult:
    """
    Compute a structured fit score between a user profile and an opportunity.

    Args:
        user: Structured user profile dict.
        opportunity: Opportunity dict from the DB.
        skill_match: Optional pre-computed graph skill match details
                     (from graph.get_skill_match_details). If None, falls back
                     to list-based comparison.
    """
    breakdown: dict[str, int] = {}
    matched: list[str] = []
    needs_confirmation: list[str] = []
    missing: list[str] = []

    # ── 1. Major fit (20 pts) ──────────────────────────────────────────────
    opp_majors = [m.lower() for m in opportunity["eligibility"].get("major", [])]
    user_major = user.get("major", "").lower()

    if not opp_majors or "any" in opp_majors:
        breakdown["major_fit"] = WEIGHTS["major_fit"]
        matched.append("No major restriction")
    elif any(m in user_major or user_major in m for m in opp_majors):
        breakdown["major_fit"] = WEIGHTS["major_fit"]
        matched.append(f"Major match: {user.get('major')}")
    else:
        breakdown["major_fit"] = 0
        missing.append(f"Major mismatch (required: {opportunity['eligibility']['major']})")

    # ── 2. Skill fit (30 pts) — uses graph details when available ──────────
    if skill_match:
        req_matched  = skill_match["required_matched"]
        req_missing  = skill_match["required_missing"]
        pref_matched = skill_match["preferred_matched"]
    else:
        user_skills = {s.lower() for s in user.get("skills", [])}
        req_skills  = opportunity.get("required_skills", [])
        pref_skills = opportunity.get("preferred_skills", [])
        req_matched  = [s for s in req_skills if s.lower() in user_skills]
        req_missing  = [s for s in req_skills if s.lower() not in user_skills]
        pref_matched = [s for s in pref_skills if s.lower() in user_skills]

    req_skills_total = len(req_matched) + len(req_missing)
    req_ratio  = len(req_matched) / req_skills_total if req_skills_total else 1.0
    pref_ratio = len(pref_matched) / (len(pref_matched) + len(skill_match["preferred_missing"])) \
        if skill_match and (len(pref_matched) + len(skill_match["preferred_missing"])) > 0 \
        else (1.0 if not opportunity.get("preferred_skills") else 0.0)

    skill_score  = int(WEIGHTS["skill_fit"] * 0.7 * req_ratio)
    skill_score += int(WEIGHTS["skill_fit"] * 0.3 * pref_ratio)
    breakdown["skill_fit"] = min(skill_score, WEIGHTS["skill_fit"])

    if req_matched:
        matched.append(f"Required skills matched: {req_matched}")
    if pref_matched:
        matched.append(f"Preferred skills matched: {pref_matched}")
    if req_missing:
        missing.append(f"Required skills missing: {req_missing}")

    # ── 3. Experience fit (20 pts) ────────────────────────────────────────
    project_keywords = {
        kw.lower()
        for p in user.get("projects", [])
        for kw in p.get("keywords", [])
    }
    criteria    = {c.lower() for c in opportunity.get("selection_criteria", [])}
    pref_lower  = {s.lower() for s in opportunity.get("preferred_skills", [])}
    relevant    = project_keywords & (pref_lower | criteria)

    if relevant:
        breakdown["experience_fit"] = WEIGHTS["experience_fit"]
        matched.append(f"Relevant project keywords: {sorted(relevant)}")
    elif user.get("projects"):
        breakdown["experience_fit"] = int(WEIGHTS["experience_fit"] * 0.5)
        matched.append("Has projects (indirect relevance)")
    else:
        breakdown["experience_fit"] = 0
        missing.append("No project or research experience listed")

    # ── 4. Eligibility fit (20 pts) ───────────────────────────────────────
    eligibility  = opportunity.get("eligibility", {})
    elig_score   = WEIGHTS["eligibility_fit"]
    intl_allowed = eligibility.get("international_student_allowed", "unknown")
    is_intl      = user.get("student_status") == "international_student"

    if is_intl:
        if intl_allowed == "yes":
            matched.append("International students explicitly allowed")
        elif intl_allowed == "no":
            elig_score -= 10
            missing.append("International students not eligible — consider other opportunities")
        else:
            elig_score -= 3
            needs_confirmation.append("Confirm international student eligibility with the organizer")
    else:
        if intl_allowed in ("yes", "unknown"):
            matched.append("Domestic student eligible")

    grad_status = eligibility.get("graduation_status", [])
    user_grad   = "expected_graduate" if user.get("expected_graduation_year") else "student"
    if grad_status and "any" not in grad_status and user_grad not in grad_status:
        elig_score -= 5
        needs_confirmation.append(f"Confirm whether your graduation status qualifies (listed: {grad_status})")

    for note in opportunity.get("risk_notes", []):
        if note not in needs_confirmation:
            needs_confirmation.append(note)

    breakdown["eligibility_fit"] = max(elig_score, 0)

    # ── 5. Document fit (10 pts) ──────────────────────────────────────────
    req_docs  = set(opportunity.get("required_documents", []))
    have_docs = set(user.get("available_documents", []))
    ready     = req_docs & have_docs
    not_ready = req_docs - have_docs

    doc_ratio = len(ready) / len(req_docs) if req_docs else 1.0
    breakdown["document_fit"] = int(WEIGHTS["document_fit"] * doc_ratio)

    if ready:
        matched.append(f"Documents ready: {sorted(ready)}")
    if not_ready:
        missing.append(f"Documents to obtain: {sorted(not_ready)}")

    total = sum(breakdown.values())
    return FitResult(
        total_score=total,
        verdict=_verdict(total),
        breakdown=breakdown,
        matched=matched,
        needs_confirmation=needs_confirmation,
        missing=missing,
    )
