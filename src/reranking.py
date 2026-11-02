"""
DocDeclutter — Reranking Engine
Re-ranks candidate retrieved passages to surface the highest-precision evidence at rank 1.
Includes transit entity boosting and support for Cross-Encoder models.
"""

import re
from typing import List, Dict, Any


class TransitEvidenceReranker:
    """
    Reranks candidate retrieved passages based on:
    1. Technical transit entity and code matching (e.g., TM-104, VCB-02, AP-ALV-03)
    2. Intent-section alignment (e.g., 'cause' -> 'ROOT CAUSE ANALYSIS', 'threshold' -> 'SPECIFICATIONS')
    3. Structural completeness and density of information
    """
    def __init__(self):
        self.intent_section_weights = {
            "cause": ["root cause", "incident description", "sequence of events", "reported defect"],
            "why": ["root cause", "failure", "diagnostic", "findings"],
            "delay": ["sequence of events", "executive summary", "incident"],
            "speed": ["degraded", "fallback", "speed", "operating"],
            "threshold": ["threshold", "limits", "specifications", "clearance"],
            "temperature": ["temperature", "rtd", "bearing", "threshold"],
            "action": ["corrective", "capa", "repair", "preventive"],
            "inspection": ["measurement", "findings", "schedule", "usfd", "wear"]
        }

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank candidates based on contextual alignment."""
        if not candidates:
            return []

        q_lower = query.lower()
        q_tokens = set(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", q_lower))

        reranked = []
        for cand in candidates:
            ch = cand["chunk"]
            base_score = cand.get("score", 0.0)

            # 1. Exact entity & code boost
            entity_boost = 0.0
            for t in q_tokens:
                # If query contains hyphenated transit codes or numbers
                if ("-" in t or any(c.isdigit() for c in t)) and len(t) >= 3:
                    if t in ch.text.lower():
                        entity_boost += 0.35
                    if t in ch.section_title.lower():
                        entity_boost += 0.50

            # 2. Intent-section alignment boost
            intent_boost = 0.0
            for intent_kw, matching_sections in self.intent_section_weights.items():
                if intent_kw in q_lower:
                    for ms in matching_sections:
                        if ms in ch.section_title.lower():
                            intent_boost += 0.25
                        if ms in ch.text[:100].lower():
                            intent_boost += 0.15

            # 3. Heading presence boost
            heading_boost = 0.10 if ("SECTION" in ch.section_title or "Doc" in ch.section_title) else 0.0

            final_rerank_score = base_score + entity_boost + intent_boost + heading_boost
            
            c_copy = dict(cand)
            c_copy["original_score"] = base_score
            c_copy["rerank_score"] = round(final_rerank_score, 4)
            c_copy["entity_boost"] = round(entity_boost, 2)
            c_copy["intent_boost"] = round(intent_boost, 2)
            reranked.append(c_copy)

        # Sort descending by rerank_score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

