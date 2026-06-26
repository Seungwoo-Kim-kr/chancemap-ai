"""
Opportunity Graph — NetworkX-based graph for ChanceMap AI.

Node types  : opportunity, skill, document, field
Edge types  : requires_skill, prefers_skill, requires_document, belongs_to_field
"""

import networkx as nx


def build_graph(opportunities: list[dict]) -> nx.DiGraph:
    """Build a directed graph connecting opportunities to skills, documents, and fields."""
    G = nx.DiGraph()

    for opp in opportunities:
        opp_id = opp["id"]
        G.add_node(opp_id, type="opportunity", title=opp["title"],
                   opp_type=opp["type"], deadline=opp["deadline"])

        for skill in opp.get("required_skills", []):
            node = f"skill:{skill.lower()}"
            G.add_node(node, type="skill", name=skill)
            G.add_edge(opp_id, node, relation="requires_skill")

        for skill in opp.get("preferred_skills", []):
            node = f"skill:{skill.lower()}"
            G.add_node(node, type="skill", name=skill)
            G.add_edge(opp_id, node, relation="prefers_skill")

        for doc in opp.get("required_documents", []):
            node = f"doc:{doc}"
            G.add_node(node, type="document", name=doc)
            G.add_edge(opp_id, node, relation="requires_document")

        for field in opp.get("field", []):
            node = f"field:{field.lower()}"
            G.add_node(node, type="field", name=field)
            G.add_edge(opp_id, node, relation="belongs_to_field")

    return G


def find_opportunities_by_skills(G: nx.DiGraph, user_skills: list[str]) -> dict[str, int]:
    """
    Multi-hop traversal: user skills → shared skill nodes → connected opportunities.
    Returns {opportunity_id: graph_match_score}.
    """
    scores: dict[str, int] = {}
    for skill in user_skills:
        skill_node = f"skill:{skill.lower()}"
        if skill_node not in G:
            continue
        for pred in G.predecessors(skill_node):
            if G.nodes[pred].get("type") != "opportunity":
                continue
            relation = G[pred][skill_node].get("relation", "")
            scores[pred] = scores.get(pred, 0) + (2 if relation == "requires_skill" else 1)
    return scores


def find_opportunities_by_fields(G: nx.DiGraph, user_fields: list[str]) -> dict[str, int]:
    """Traversal: user interest fields → field nodes → connected opportunities."""
    scores: dict[str, int] = {}
    for field in user_fields:
        field_node = f"field:{field.lower()}"
        if field_node not in G:
            continue
        for pred in G.predecessors(field_node):
            if G.nodes[pred].get("type") != "opportunity":
                continue
            scores[pred] = scores.get(pred, 0) + 1
    return scores


def get_required_documents(G: nx.DiGraph, opp_id: str) -> list[str]:
    """Traverse opportunity → document edges to get required documents."""
    docs = []
    for _, neighbor, data in G.out_edges(opp_id, data=True):
        if data.get("relation") == "requires_document":
            docs.append(G.nodes[neighbor]["name"])
    return docs


def get_skill_match_details(G: nx.DiGraph, opp_id: str, user_skills: list[str]) -> dict:
    """
    Return matched/missing required and preferred skills for an opportunity.
    Uses graph edges rather than list comparison.
    """
    user_skill_set = {s.lower() for s in user_skills}
    required_matched, required_missing = [], []
    preferred_matched, preferred_missing = [], []

    for _, neighbor, data in G.out_edges(opp_id, data=True):
        skill_name = G.nodes[neighbor].get("name", neighbor)
        relation = data.get("relation", "")
        if relation == "requires_skill":
            if skill_name.lower() in user_skill_set:
                required_matched.append(skill_name)
            else:
                required_missing.append(skill_name)
        elif relation == "prefers_skill":
            if skill_name.lower() in user_skill_set:
                preferred_matched.append(skill_name)
            else:
                preferred_missing.append(skill_name)

    return {
        "required_matched": required_matched,
        "required_missing": required_missing,
        "preferred_matched": preferred_matched,
        "preferred_missing": preferred_missing,
    }


def graph_summary(G: nx.DiGraph) -> str:
    node_types: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    return (
        f"Graph: {G.number_of_nodes()} nodes "
        f"({', '.join(f'{v} {k}' for k, v in node_types.items())}), "
        f"{G.number_of_edges()} edges"
    )
