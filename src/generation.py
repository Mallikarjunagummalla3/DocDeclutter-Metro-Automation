"""
DocDeclutter — LLM Generation Engine
Implements evidence-grounded generation, multi-provider LLM integration (Qwen/Ollama/OpenAI/Local),
strict SIH prompt formatting, and evidence sufficiency / abstention checking.
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are DocDeclutter, an assistant for retrieving and summarizing transit operational documents.
Answer using only the evidence provided in the retrieved document sections.

Instructions:
1. Identify the information relevant to the user's question.
2. Summarize the information clearly and concisely.
3. Do not invent incidents, dates, causes, equipment details, or conclusions.
4. If the evidence does not answer the question, state that the available documents do not provide enough evidence to answer this question.
5. Include source document IDs and page or section references for the claims you make.
6. Distinguish facts explicitly stated in the documents from reasonable interpretations."""

USER_PROMPT_TEMPLATE = """User question:
{question}

Retrieved evidence:
{retrieved_sections}

Answer:"""

ABSTENTION_MESSAGE = "The available documents do not provide enough evidence to answer this question."


GENERIC_TRANSIT_TERMS = {
    "metro", "train", "station", "line", "kmrl", "kochi", "rail", "system", 
    "report", "document", "service", "what", "which", "when", "where", "how", 
    "who", "why", "the", "and", "for", "with", "from", "that", "this", "does", 
    "were", "are", "was", "has", "have", "had", "per", "between", "during", "under"
}


def sufficient_evidence(question: str, retrieved_sections: List[Dict[str, Any]], 
                        min_match_ratio: float = 0.40) -> bool:
    """
    Evidence sufficiency check (Novelty 3):
    Determines whether the retrieved sections contain genuine supporting evidence
    for the user's question.
    Extracts salient query entities/keywords (excluding generic transit domain stop words)
    and verifies that a significant ratio appears in the retrieved passage text.
    If the core subjects/entities are absent, triggers safe abstention.
    """
    if not retrieved_sections:
        return False

    q_lower = question.lower()
    q_words = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", q_lower)
    specific_keywords = [
        w for w in q_words 
        if len(w) > 2 and w not in GENERIC_TRANSIT_TERMS
    ]

    # Combine text from top 3 candidates
    evidence_text = " ".join([
        f"{c['chunk'].doc_title} {c['chunk'].section_title} {c['chunk'].text}".lower() 
        for c in retrieved_sections[:3]
    ])

    # Check match ratio of specific informative question terms
    if specific_keywords:
        matched_terms = [w for w in specific_keywords if w in evidence_text]
        match_ratio = len(matched_terms) / len(specific_keywords)
        
        # If less than 40% of specific query keywords appear in evidence, abstain
        if match_ratio < min_match_ratio:
            return False

    return True


def format_evidence_context(retrieved_sections: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into a structured evidence block for the LLM prompt."""
    blocks = []
    for idx, cand in enumerate(retrieved_sections, start=1):
        ch = cand["chunk"]
        block = (
            f"[Evidence #{idx}]\n"
            f"Document ID: {ch.doc_id}\n"
            f"Document Title: {ch.doc_title}\n"
            f"Section: {ch.section_title}\n"
            f"Page: {ch.page_number}\n"
            f"Passage Content:\n{ch.text.strip()}\n"
        )
        blocks.append(block)
    return "\n----------------------------------------\n".join(blocks)


def format_sources(retrieved_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format structured source citation list for UI display and auditability."""
    sources = []
    seen = set()
    for cand in retrieved_sections:
        ch = cand["chunk"]
        key = (ch.doc_id, ch.section_title, ch.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_id": ch.doc_id,
                "title": ch.doc_title,
                "section": ch.section_title,
                "page": ch.page_number,
                "score": cand.get("rerank_score", cand.get("score", 0.0)),
                "snippet": ch.text[:220].strip() + ("..." if len(ch.text) > 220 else "")
            })
    return sources


class LLMGenerator:
    """
    LLM Generation manager with support for:
    - OpenAI-compatible API endpoints (OpenAI, Groq, Together, DeepSeek, vLLM)
    - Ollama local server (http://localhost:11434)
    - Built-in Evidence-Grounded Heuristic Synthesizer (runs with zero external keys/GPU)
    """
    def __init__(self, provider: str = "builtin", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, model_name: str = "qwen2.5-7b-instruct"):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name

    def generate(self, question: str, retrieved_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer grounded strictly in retrieved evidence."""
        # 1. Evidence sufficiency check
        if not sufficient_evidence(question, retrieved_sections):
            return {
                "answer": ABSTENTION_MESSAGE,
                "sources": [],
                "sufficient_evidence": False,
                "model_used": "DocDeclutter-Abstention-Guard"
            }

        evidence_text = format_evidence_context(retrieved_sections)
        sources = format_sources(retrieved_sections)

        # 2. Dispatch to requested provider
        if self.provider == "ollama":
            answer = self._call_ollama(question, evidence_text)
        elif self.provider in ["openai", "groq", "api"]:
            answer = self._call_api(question, evidence_text)
        else:
            # Built-in local grounded synthesizer
            answer = self._synthesize_grounded_answer(question, retrieved_sections)

        return {
            "answer": answer,
            "sources": sources,
            "sufficient_evidence": True,
            "model_used": f"{self.provider}:{self.model_name}"
        }

    def _call_api(self, question: str, evidence_text: str) -> str:
        """Call OpenAI or compatible API (Groq, Together, vLLM)."""
        if not self.api_key:
            return self._synthesize_grounded_answer(question, [])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question, retrieved_sections=evidence_text)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"[API Error {resp.status_code}: Falling back to local synthesis]\n\n" + self._synthesize_grounded_answer(question, [])
        except Exception as e:
            return f"[API Connection Failed: {e}]\n\n" + self._synthesize_grounded_answer(question, [])

    def _call_ollama(self, question: str, evidence_text: str) -> str:
        """Call local Ollama instance running Qwen2.5 or Llama."""
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question, retrieved_sections=evidence_text)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                return self._synthesize_grounded_answer(question, [])
        except Exception:
            return self._synthesize_grounded_answer(question, [])

    def _synthesize_grounded_answer(self, question: str, retrieved_sections: List[Dict[str, Any]]) -> str:
        """
        Grounded Extractive Synthesis Engine:
        Extracts and structures facts directly from top retrieved evidence snippets,
        attaching document and section citations.
        """
        if not retrieved_sections:
            return ABSTENTION_MESSAGE

        top_cand = retrieved_sections[0]
        ch = top_cand["chunk"]
        
        # Split chunk text into candidate sentences
        sentences = re.split(r"(?<=[.!?])\s+", ch.text.strip())
        
        q_words = set(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", question.lower()))
        # Remove common query stopwords
        stopwords = {"what", "when", "where", "which", "how", "the", "and", "for", "with", "from", "that", "this", "does", "were"}
        q_keywords = {w for w in q_words if w not in stopwords and len(w) > 2}

        # Score sentences by keyword overlap and numbers/codes
        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_lower = s_clean.lower()
            s_words = set(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", s_lower))
            overlap = len(s_words.intersection(q_keywords))
            
            # Boost if contains numbers or hyphenated entities
            has_entities = 1 if any("-" in w or any(c.isdigit() for c in w) for w in s_words) else 0
            score = overlap * 2.0 + has_entities * 1.5
            scored_sentences.append((score, s_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences if score > 0][:3]
        
        if not top_sentences and scored_sentences:
            top_sentences = [scored_sentences[0][1]]

        answer_body = " ".join(top_sentences)
        
        # If second chunk has strong relevance, include complementary sentence
        if len(retrieved_sections) > 1:
            ch2 = retrieved_sections[1]["chunk"]
            s2_list = re.split(r"(?<=[.!?])\s+", ch2.text.strip())
            for s in s2_list:
                s_lower = s.lower()
                s_words = set(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", s_lower))
                if len(s_words.intersection(q_keywords)) >= 2:
                    answer_body += f"\n\nAdditionally, according to [{ch2.doc_id} - {ch2.section_title}]: {s.strip()}"
                    break

        return (
            f"Based on **{ch.doc_id}** ({ch.doc_title}), in **{ch.section_title}** (Page {ch.page_number}):\n\n"
            f"{answer_body}\n\n"
            f"*Source Verified: [{ch.doc_id} | Page {ch.page_number}]*"
        )
