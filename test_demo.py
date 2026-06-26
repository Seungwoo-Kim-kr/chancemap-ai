"""
ChanceMap AI — End-to-End Demo Test
Runs three scenarios to validate full Graph + BM25 + scoring pipeline.

Scenario 1: International AI student (internship + competition)
Scenario 2: Career changer (AI education + job)
Scenario 3: Entrepreneur (government support + startup)
"""

import json
from pathlib import Path
from tools.graph import build_graph, get_skill_match_details, graph_summary
from tools.retriever import build_bm25_index, retrieve, build_user_query
from tools.scoring import calculate_fit

DATA_DIR = Path(__file__).parent / "data"
opps     = json.loads((DATA_DIR / "opportunities.json").read_text(encoding="utf-8"))
users    = json.loads((DATA_DIR / "demo_user.json").read_text(encoding="utf-8"))

GRAPH = build_graph(opps)
BM25  = build_bm25_index(opps)
OPP_INDEX = {o["id"]: o for o in opps}

SEP  = "=" * 65
SEP2 = "-" * 65

DOC_LABELS = {
    "resume": "Resume", "transcript": "Transcript",
    "certificate_of_enrollment": "Certificate of Enrollment",
    "portfolio": "Portfolio", "cover_letter": "Cover Letter",
    "recommendation_letter": "Recommendation Letter",
    "project_proposal": "Project Proposal",
    "team_introduction": "Team Introduction",
}


def bar(score: int) -> str:
    f = score // 10
    return "█" * f + "░" * (10 - f)


def run_scenario(user: dict, label: str, opp_type_filter: list[str]):
    print(f"\n{SEP}")
    print(f"  SCENARIO: {label}")
    print(SEP)

    # Profile summary
    print(f"\n  Major       : {user['major']}")
    print(f"  Career Stage: {user['career_stage']}")
    print(f"  Skills      : {', '.join(user['skills'])}")
    print(f"  Goal        : {', '.join(user['target_opportunity_type'])}")
    print(f"  Documents   : {', '.join(user['available_documents'])}")

    # BM25 + Graph retrieval
    bm25_query = build_user_query(
        user["skills"], user["target_field"], user["target_opportunity_type"]
    )
    bm25_results = retrieve(BM25, opps, bm25_query, top_k=10)

    from tools.graph import find_opportunities_by_skills
    graph_scores = find_opportunities_by_skills(GRAPH, user["skills"])

    # Merge + filter by type
    filter_set = set(opp_type_filter)
    combined = []
    for opp, bm25_score in bm25_results:
        if filter_set and opp["type"] not in filter_set:
            continue
        g = graph_scores.get(opp["id"], 0)
        combined.append((opp, bm25_score + g * 0.5))
    combined.sort(key=lambda x: x[1], reverse=True)

    # Fit scoring for top results
    scored = []
    for opp, _ in combined[:8]:
        skill_match = get_skill_match_details(GRAPH, opp["id"], user["skills"])
        fit = calculate_fit(user, opp, skill_match=skill_match)
        scored.append((fit, opp))
    scored.sort(key=lambda x: x[0].total_score, reverse=True)

    # Ranking table
    print(f"\n  {'Rank':<5} {'Score':>6}  {'Verdict':<26}  Title")
    print(f"  {SEP2}")
    for rank, (fit, opp) in enumerate(scored[:6], 1):
        print(f"  {rank:<5} {fit.total_score:>3}/100  {fit.verdict:<26}  {opp['title']}")

    # Top pick detail
    if not scored:
        print("\n  No results for this filter. Try broadening opportunity type.")
        return

    best_fit, best_opp = scored[0]
    skill_match = get_skill_match_details(GRAPH, best_opp["id"], user["skills"])

    print(f"\n  {SEP2}")
    print(f"  TOP PICK: {best_opp['title']}")
    print(f"  Score  : {best_fit.total_score}/100 [{bar(best_fit.total_score)}]  →  {best_fit.verdict}")
    print(f"  Type   : {best_opp['type']}  |  Deadline: {best_opp['deadline']}")

    if skill_match["required_matched"]:
        print(f"\n  Graph ✅ Required skills matched : {skill_match['required_matched']}")
    if skill_match["preferred_matched"]:
        print(f"  Graph ⭐ Preferred skills matched: {skill_match['preferred_matched']}")
    if skill_match["required_missing"]:
        print(f"  Graph ❌ Required skills missing : {skill_match['required_missing']}")

    if best_fit.needs_confirmation:
        print(f"\n  ❓ Confirm:")
        for n in best_fit.needs_confirmation[:3]:
            print(f"     - {n}")

    # Checklist
    req_docs  = best_opp.get("required_documents", [])
    have      = set(user["available_documents"])
    not_ready = [d for d in req_docs if d not in have]

    print(f"\n  Action Items:")
    step = 1
    if not_ready:
        print(f"  {step}. Prepare: {', '.join(DOC_LABELS.get(d,d) for d in not_ready)}")
        step += 1
    for note in best_fit.needs_confirmation[:2]:
        print(f"  {step}. {note}")
        step += 1
    print(f"  {step}. Update resume / portfolio with latest projects")
    print(f"  {step+1}. Register deadline: {best_opp['deadline']}")


def main():
    print(f"\n{SEP}")
    print("  ChanceMap AI — Demo Test")
    print(f"  {graph_summary(GRAPH)}")
    print(f"  Opportunities loaded: {len(opps)}")
    print(SEP)

    u1 = users["scenario_1_international_student"]
    u2 = users["scenario_2_career_changer"]
    u3 = users["scenario_3_entrepreneur"]

    run_scenario(u1, "International AI Student — Internship & Research",
                 ["internship", "competition", "research"])

    run_scenario(u2, "Career Changer (Mech Eng → AI) — Education & Job",
                 ["education", "job", "competition"])

    run_scenario(u3, "Entrepreneur — Government Support & Startup Funding",
                 ["government_support", "startup_support", "competition"])

    print(f"\n{SEP}")
    print("  All scenarios completed.")
    print(SEP)


if __name__ == "__main__":
    main()
