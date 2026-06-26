"""
BM25-based opportunity retriever for ChanceMap AI.

BM25 enables semantic keyword retrieval across opportunity titles,
descriptions, fields, and skills — improving on simple keyword matching.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9가-힣]+", text.lower())


def _opportunity_to_doc(opp: dict) -> list[str]:
    """Combine all searchable fields of an opportunity into a token list."""
    parts = [
        opp.get("title", ""),
        opp.get("description", ""),
        " ".join(opp.get("field", [])),
        " ".join(opp.get("required_skills", [])),
        " ".join(opp.get("preferred_skills", [])),
        " ".join(opp.get("selection_criteria", [])),
        opp.get("type", ""),
    ]
    return _tokenize(" ".join(parts))


def build_bm25_index(opportunities: list[dict]) -> BM25Okapi:
    """Build a BM25 index over all opportunities."""
    corpus = [_opportunity_to_doc(opp) for opp in opportunities]
    return BM25Okapi(corpus)


def retrieve(
    bm25: BM25Okapi,
    opportunities: list[dict],
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[tuple[dict, float]]:
    """
    Retrieve top-k opportunities matching the query using BM25.

    Returns a list of (opportunity, bm25_score) tuples sorted by score descending.
    """
    tokens = _tokenize(query)
    if not tokens:
        return [(opp, 0.0) for opp in opportunities[:top_k]]

    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = [
        (opportunities[i], float(score))
        for i, score in ranked
        if score > min_score
    ]
    return results[:top_k]


def build_user_query(user_skills: list[str], user_fields: list[str],
                     opp_types: list[str], extra: str = "") -> str:
    """Construct a BM25 query string from user profile fields."""
    parts = user_skills + user_fields + opp_types + ([extra] if extra else [])
    return " ".join(parts)
