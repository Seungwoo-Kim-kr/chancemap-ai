"""
ChanceMap AI — Opportunity Graph Agent
An Agentic MCP service that connects users' profiles to relevant
opportunities (internships, competitions, scholarships, government
support, startup programs, and career education) using NetworkX Graph
traversal and BM25 retrieval.

Target users: students, job seekers, career changers, and entrepreneurs.
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from tools.graph import (
    build_graph,
    find_opportunities_by_skills,
    get_skill_match_details,
    graph_summary,
)
from tools.retriever import build_bm25_index, retrieve, build_user_query
from tools.scoring import calculate_fit

# ── Data & Index Initialization ───────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "opportunities.json", encoding="utf-8") as f:
    OPPORTUNITIES: list[dict] = json.load(f)

OPP_INDEX: dict[str, dict] = {opp["id"]: opp for opp in OPPORTUNITIES}

# Build NetworkX Graph (user → skill → opportunity → document)
GRAPH = build_graph(OPPORTUNITIES)

# Build BM25 index for semantic keyword retrieval
BM25 = build_bm25_index(OPPORTUNITIES)

PORT = int(os.environ.get("PORT", 8000))

# ── FastMCP App ───────────────────────────────────────────────────────────
mcp = FastMCP(
    name="chancemap-ai",
    instructions=(
        "ChanceMap AI is an Agentic MCP service that analyzes a user's major, "
        "skills, projects, career stage, and document status using a Graph structure, "
        "then connects them to internships, competitions, scholarships, government "
        "support programs, startup funding, and career education opportunities. "
        "The service provides fit scores, preparation checklists, and application "
        "strategy points. It does NOT predict acceptance probability — it provides "
        "application fit and preparation priority. "
        "Suitable for students, job seekers, career changers, and entrepreneurs."
    ),
    host="0.0.0.0",
    port=PORT,
)

# ── Document label map ────────────────────────────────────────────────────
_DOC_LABELS = {
    "resume": "Resume",
    "transcript": "Transcript",
    "certificate_of_enrollment": "Certificate of Enrollment",
    "portfolio": "Portfolio",
    "cover_letter": "Cover Letter",
    "recommendation_letter": "Recommendation Letter",
    "english_certificate": "English Proficiency Certificate",
    "graduation_certificate": "Graduation Certificate",
    "project_proposal": "Project Proposal",
    "team_introduction": "Team Introduction",
}


# ── Tool 1: analyze_user_profile ─────────────────────────────────────────
@mcp.tool()
def analyze_user_profile(
    major: str,
    career_stage: str,
    skills: str,
    target_field: str,
    target_opportunity_type: str,
    degree_level: str = "undergraduate",
    expected_graduation_year: int = 0,
    available_documents: str = "",
    projects: str = "",
    constraints: str = "",
) -> str:
    """
    Structure the user's background into a profile for opportunity matching.

    Args:
        major: Field of study or professional background (e.g. Artificial Intelligence, Mechanical Engineering)
        career_stage: Current stage — student | job_seeker | early_career | career_changer | entrepreneur | anyone
        skills: Comma-separated skills (e.g. Python, RAG, Data Analysis)
        target_field: Comma-separated fields of interest (e.g. AI, Data, Startup)
        target_opportunity_type: Comma-separated types — internship | competition | scholarship | research | government_support | startup_support | job | education
        degree_level: undergraduate | graduate | phd (default: undergraduate)
        expected_graduation_year: Graduation year, 0 if already graduated (e.g. 2027)
        available_documents: Comma-separated documents already prepared (e.g. resume, portfolio)
        projects: Semicolon-separated project titles (e.g. RAG QA System;Graph Retrieval Research)
        constraints: Comma-separated constraints or notes (e.g. international student, currently employed)
    """
    user_skills = [s.strip() for s in skills.split(",") if s.strip()]
    user_fields  = [f.strip() for f in target_field.split(",") if f.strip()]
    opp_types    = [t.strip() for t in target_opportunity_type.split(",") if t.strip()]
    docs_ready   = [d.strip() for d in available_documents.split(",") if d.strip()]
    user_projs   = [
        {"title": p.strip(), "keywords": p.strip().split()}
        for p in projects.split(";") if p.strip()
    ]
    user_constraints = [c.strip() for c in constraints.split(",") if c.strip()]

    # Graph: find candidate opportunities via skill traversal
    graph_scores = find_opportunities_by_skills(GRAPH, user_skills)
    top_graph = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    lines = [
        "## User Profile Structured",
        "",
        f"**Major / Background** : {major}",
        f"**Career Stage**        : {career_stage}",
        f"**Degree Level**        : {degree_level}",
        f"**Graduation Year**     : {expected_graduation_year or 'Already graduated'}",
        f"**Skills**              : {', '.join(user_skills) or 'None listed'}",
        f"**Target Fields**       : {', '.join(user_fields) or 'Any'}",
        f"**Opportunity Types**   : {', '.join(opp_types) or 'Any'}",
        f"**Documents Ready**     : {', '.join(docs_ready) or 'None'}",
        f"**Constraints / Notes** : {', '.join(user_constraints) or 'None'}",
    ]

    if user_projs:
        lines += ["", "**Projects / Experience**:"]
        for p in user_projs:
            lines.append(f"  - {p['title']}")

    if top_graph:
        lines += [
            "",
            "**Graph Traversal — Top skill-matched opportunities**:",
        ]
        for opp_id, score in top_graph:
            opp = OPP_INDEX.get(opp_id, {})
            lines.append(f"  - [{opp_id}] {opp.get('title', opp_id)} (graph score: {score})")

    lines += [
        "",
        "---",
        f"Graph index: {graph_summary(GRAPH)}",
        "",
        "Next steps:",
        "  • Use `search_opportunities` to find matching opportunities",
        "  • Use `analyze_fit` with a specific opportunity ID for detailed scoring",
        f"  • Available opportunity IDs: {', '.join(list(OPP_INDEX.keys())[:10])} ...",
    ]

    return "\n".join(lines)


# ── Tool 2: search_opportunities ─────────────────────────────────────────
@mcp.tool()
def search_opportunities(
    skills: str = "",
    target_field: str = "",
    target_type: str = "",
    career_stage: str = "",
    query: str = "",
) -> str:
    """
    Search for relevant opportunities using BM25 retrieval and Graph-based skill matching.

    Args:
        skills: Comma-separated user skills (e.g. Python, RAG, LLM)
        target_field: Comma-separated fields of interest (e.g. AI, Data, Startup)
        target_type: Comma-separated opportunity types to filter (internship | competition | scholarship | research | government_support | startup_support | job | education)
        career_stage: Filter by career stage (student | job_seeker | early_career | career_changer | entrepreneur | anyone). Leave empty for no filter.
        query: Optional free-text search query (e.g. 'LLM service startup funding')
    """
    user_skills  = [s.strip() for s in skills.split(",") if s.strip()]
    user_fields  = [f.strip() for f in target_field.split(",") if f.strip()]
    filter_types = {t.strip().lower() for t in target_type.split(",") if t.strip()}

    # BM25: semantic keyword search
    bm25_query = query or build_user_query(user_skills, user_fields, list(filter_types))
    bm25_results = retrieve(BM25, OPPORTUNITIES, bm25_query, top_k=15)

    # Graph: skill-based score boost
    graph_scores = find_opportunities_by_skills(GRAPH, user_skills)

    # Merge BM25 score + Graph score
    combined: list[tuple[dict, float]] = []
    for opp, bm25_score in bm25_results:
        # Type filter
        if filter_types and opp["type"] not in filter_types:
            continue
        # Career stage filter
        if career_stage:
            opp_stages = opp.get("career_stage", ["anyone"])
            if career_stage not in opp_stages and "anyone" not in opp_stages:
                continue
        g_score = graph_scores.get(opp["id"], 0)
        combined.append((opp, bm25_score + g_score * 0.5))

    # If BM25 returns nothing (no query), fall back to all opportunities filtered
    if not combined:
        for opp in OPPORTUNITIES:
            if filter_types and opp["type"] not in filter_types:
                continue
            if career_stage:
                opp_stages = opp.get("career_stage", ["anyone"])
                if career_stage not in opp_stages and "anyone" not in opp_stages:
                    continue
            g_score = graph_scores.get(opp["id"], 0)
            combined.append((opp, float(g_score)))

    combined.sort(key=lambda x: x[1], reverse=True)

    if not combined:
        return (
            "No opportunities found matching your criteria.\n"
            "Try broadening `target_type` or removing `career_stage` filter."
        )

    intl_map = {"yes": "✅", "no": "❌", "unknown": "❓"}

    lines = [f"## Search Results: {len(combined)} opportunities found", ""]
    for rank, (opp, score) in enumerate(combined[:12], 1):
        intl = intl_map.get(opp["eligibility"].get("international_student_allowed", "unknown"), "❓")
        lines += [
            f"### {rank}. [{opp['id']}] {opp['title']}",
            f"- **Type**: {opp['type']} | **Field**: {', '.join(opp['field'])} | **Deadline**: {opp['deadline']}",
            f"- **Career Stage**: {', '.join(opp.get('career_stage', ['anyone']))} | **International**: {intl}",
            f"- **Required Skills**: {', '.join(opp['required_skills']) or 'None'}",
            f"- **Preferred Skills**: {', '.join(opp.get('preferred_skills', [])) or 'None'}",
            "",
        ]

    lines += [
        "---",
        "Use `analyze_fit <opportunity_id>` to get a detailed fit score and preparation plan.",
    ]
    return "\n".join(lines)


# ── Tool 3: analyze_fit ───────────────────────────────────────────────────
@mcp.tool()
def analyze_fit(
    opportunity_id: str,
    major: str,
    skills: str,
    career_stage: str = "student",
    expected_graduation_year: int = 0,
    available_documents: str = "",
    projects: str = "",
) -> str:
    """
    Compare the user's profile against a specific opportunity using Graph traversal
    and rule-based scoring. Returns fit score, matched/missing conditions, and next steps.

    Args:
        opportunity_id: Opportunity ID (e.g. opp_001). Use search_opportunities to find IDs.
        major: Field of study or professional background
        skills: Comma-separated skills (e.g. Python, RAG, LLM)
        career_stage: student | job_seeker | early_career | career_changer | entrepreneur | anyone
        expected_graduation_year: Expected graduation year, 0 if already graduated
        available_documents: Comma-separated documents already prepared
        projects: Semicolon-separated project titles with keywords
    """
    opp = OPP_INDEX.get(opportunity_id)
    if not opp:
        ids = ", ".join(list(OPP_INDEX.keys())[:10])
        return f"Opportunity ID '{opportunity_id}' not found.\nAvailable IDs (first 10): {ids}"

    user_skills = [s.strip() for s in skills.split(",") if s.strip()]

    user = {
        "major": major,
        "student_status": (
            "international_student" if "international" in career_stage
            else "domestic_student"
        ),
        "career_stage": career_stage,
        "expected_graduation_year": expected_graduation_year,
        "skills": user_skills,
        "projects": [
            {"title": p.strip(), "keywords": p.strip().split()}
            for p in projects.split(";") if p.strip()
        ],
        "available_documents": [d.strip() for d in available_documents.split(",") if d.strip()],
    }

    # Graph traversal: get skill match details from graph edges
    skill_match = get_skill_match_details(GRAPH, opportunity_id, user_skills)

    result = calculate_fit(user, opp, skill_match=skill_match)

    bar = "█" * (result.total_score // 10) + "░" * (10 - result.total_score // 10)

    lines = [
        f"## [{opp['id']}] {opp['title']}",
        f"*{opp['description']}*",
        "",
        f"**Fit Score** : {result.total_score}/100  [{bar}]",
        f"**Verdict**   : {result.verdict}",
        f"**Deadline**  : {opp['deadline']}",
        "",
        "### Score Breakdown",
        "| Criterion | Score | Max |",
        "|-----------|-------|-----|",
        f"| Major alignment      | {result.breakdown['major_fit']} | 20 |",
        f"| Skill match          | {result.breakdown['skill_fit']} | 30 |",
        f"| Project experience   | {result.breakdown['experience_fit']} | 20 |",
        f"| Eligibility          | {result.breakdown['eligibility_fit']} | 20 |",
        f"| Document readiness   | {result.breakdown['document_fit']} | 10 |",
        "",
    ]

    if skill_match["required_matched"]:
        lines.append(f"**Graph — Required skills matched**: {skill_match['required_matched']}")
    if skill_match["preferred_matched"]:
        lines.append(f"**Graph — Preferred skills matched**: {skill_match['preferred_matched']}")
    if skill_match["required_missing"]:
        lines.append(f"**Graph — Required skills missing**: {skill_match['required_missing']}")
    lines.append("")

    if result.matched:
        lines += ["### ✅ Conditions Met"]
        for m in result.matched:
            lines.append(f"- {m}")
        lines.append("")

    if result.needs_confirmation:
        lines += ["### ❓ Needs Confirmation"]
        for n in result.needs_confirmation:
            lines.append(f"- {n}")
        lines.append("")

    if result.missing:
        lines += ["### ❌ Missing or Insufficient"]
        for miss in result.missing:
            lines.append(f"- {miss}")
        lines.append("")

    lines += [
        "---",
        "Next: Use `generate_checklist` for preparation steps, "
        "or `generate_strategy` for application talking points.",
    ]

    return "\n".join(lines)


# ── Tool 4: generate_checklist ────────────────────────────────────────────
@mcp.tool()
def generate_checklist(
    opportunity_id: str,
    available_documents: str = "",
    needs_confirmation: str = "",
) -> str:
    """
    Generate a prioritized preparation checklist for a specific opportunity,
    including required documents and action items for today.

    Args:
        opportunity_id: Opportunity ID (e.g. opp_001)
        available_documents: Comma-separated documents already prepared
        needs_confirmation: Comma-separated items that need external verification
    """
    opp = OPP_INDEX.get(opportunity_id)
    if not opp:
        return f"Opportunity ID '{opportunity_id}' not found."

    have     = {d.strip() for d in available_documents.split(",") if d.strip()}
    req_docs = opp.get("required_documents", [])
    ready    = [d for d in req_docs if d in have]
    missing  = [d for d in req_docs if d not in have]

    confirm_items = list({
        c.strip()
        for c in (needs_confirmation + "," + ",".join(opp.get("risk_notes", []))).split(",")
        if c.strip()
    })

    lines = [
        f"## [{opp['id']}] {opp['title']} — Preparation Checklist",
        "",
        f"**Deadline** : {opp['deadline']}",
        f"**Type**     : {opp['type']} | **Location** : {opp.get('location', 'N/A')}",
        "",
        "### Required Documents",
    ]

    for d in req_docs:
        icon  = "✅" if d in have else "⬜"
        label = _DOC_LABELS.get(d, d)
        lines.append(f"  {icon} {label}")

    lines += ["", "### Action Items (Priority Order)"]

    step = 1
    if missing:
        lines.append(f"  {step}. Obtain missing documents:")
        for d in missing:
            lines.append(f"       - {_DOC_LABELS.get(d, d)}")
        step += 1

    for item in confirm_items:
        lines.append(f"  {step}. {item}")
        step += 1

    lines += [
        f"  {step}. Update resume with latest skills and projects",
        f"  {step+1}. Prepare or update portfolio / project links",
        f"  {step+2}. Add deadline to calendar: {opp['deadline']}",
    ]

    if opp.get("application_url"):
        lines += ["", f"  **Application link**: {opp['application_url']}"]

    lines += [
        "",
        "---",
        "Use `generate_strategy` to get application talking points and appeal tips.",
    ]

    return "\n".join(lines)


# ── Tool 5: generate_strategy ─────────────────────────────────────────────
@mcp.tool()
def generate_strategy(
    opportunity_id: str,
    major: str,
    skills: str,
    career_stage: str = "student",
    projects: str = "",
) -> str:
    """
    Generate application strategy: key selling points, how to frame project experience,
    and which skills to highlight based on Graph-matched opportunity requirements.

    Args:
        opportunity_id: Opportunity ID (e.g. opp_001)
        major: Field of study or professional background
        skills: Comma-separated skills
        career_stage: student | job_seeker | early_career | career_changer | entrepreneur | anyone
        projects: Semicolon-separated project titles
    """
    opp = OPP_INDEX.get(opportunity_id)
    if not opp:
        return f"Opportunity ID '{opportunity_id}' not found."

    user_skills = [s.strip() for s in skills.split(",") if s.strip()]
    project_list = [p.strip() for p in projects.split(";") if p.strip()]

    # Graph traversal: get skill details from graph edges
    skill_match = get_skill_match_details(GRAPH, opportunity_id, user_skills)

    lines = [
        f"## [{opp['id']}] {opp['title']} — Application Strategy",
        "",
        "### Key Selling Points",
        "",
    ]

    point = 1

    if skill_match["required_matched"]:
        lines += [
            f"**{point}. Lead with required skill experience**",
            f"   - You match: {skill_match['required_matched']}",
            "   - Frame as 'what problem did you solve' not just 'I used this tool'",
            "",
        ]
        point += 1

    if skill_match["preferred_matched"]:
        lines += [
            f"**{point}. Use preferred skills as differentiators**",
            f"   - You also have: {skill_match['preferred_matched']} (preferred, not required)",
            "   - Highlight these early — they set you apart from other applicants",
            "",
        ]
        point += 1

    if project_list:
        lines += [f"**{point}. Connect projects to this opportunity**"]
        for proj in project_list:
            lines.append(f"   - '{proj}' → link to {opp['type']} context: describe impact, not just process")
        lines += [
            "   - Include: what you built, what problem it solved, measurable outcome",
            "",
        ]
        point += 1

    # Career-stage specific framing
    stage_tips = {
        "international_student": (
            "international student",
            "Frame overseas study as global perspective + independent problem solving under different systems"
        ),
        "career_changer": (
            "career changer",
            "Frame previous industry experience as unique context — you see AI problems that pure CS grads miss"
        ),
        "entrepreneur": (
            "entrepreneur / founder",
            "Frame startup experience as end-to-end product ownership + customer validation mindset"
        ),
        "early_career": (
            "early-career professional",
            "Frame work experience as practical execution ability that students lack"
        ),
    }
    for stage_key, (stage_label, tip) in stage_tips.items():
        if stage_key in career_stage:
            lines += [
                f"**{point}. Leverage your {stage_label} background**",
                f"   - {tip}",
                "",
            ]
            point += 1
            break

    criteria = opp.get("selection_criteria", [])
    if criteria:
        lines += [
            f"**{point}. Address selection criteria directly**",
            f"   - This opportunity evaluates: {', '.join(criteria)}",
            "   - Make sure each criterion is addressed at least once in your application",
            "",
        ]

    lines += [
        "---",
        "### Important Note",
        "- This strategy is generated from your profile and the opportunity's graph structure.",
        "- Always verify eligibility requirements directly with the organizer.",
        "- Use `generate_checklist` to confirm your document readiness.",
    ]

    return "\n".join(lines)


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    mcp.run(transport="sse", host="0.0.0.0", port=PORT)
