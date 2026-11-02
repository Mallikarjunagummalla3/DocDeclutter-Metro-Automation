"""
DocDeclutter — Experimental Evaluation Framework
Executes controlled benchmark experiments comparing Baseline vs Proposed configurations
across Recall@k, MRR, Groundedness, Abstention Rate, and Response Time.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from src.ingestion import ingest_batch, Document
from src.chunking import chunk_documents_batch, structure_aware_chunking, baseline_fixed_chunking
from src.retrieval import HybridRetriever
from src.reranking import TransitEvidenceReranker
from src.generation import LLMGenerator, sufficient_evidence, ABSTENTION_MESSAGE


def evaluate_query_retrieval(query_data: Dict[str, Any], retrieved_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates retrieval for a single query:
    Checks if expected_doc_id AND expected_section are present in top-1, top-3, top-5.
    """
    expected_doc_id = query_data.get("expected_doc_id")
    expected_section = query_data.get("expected_section")
    is_answerable = query_data.get("is_answerable", True)

    if not is_answerable or not expected_doc_id:
        return {
            "doc_hit_1": None,
            "doc_hit_5": None,
            "section_hit_1": None,
            "section_hit_5": None,
            "reciprocal_rank": None
        }

    retrieved_doc_ids = [c["chunk"].doc_id for c in retrieved_candidates]
    retrieved_sections = [c["chunk"].section_title for c in retrieved_candidates]
    
    doc_hit_1 = 1.0 if (len(retrieved_doc_ids) > 0 and retrieved_doc_ids[0] == expected_doc_id) else 0.0
    doc_hit_5 = 1.0 if expected_doc_id in retrieved_doc_ids[:5] else 0.0

    # Section-level matching: check if expected section appears in retrieved chunk section title or text
    section_hit_1 = 0.0
    section_hit_5 = 0.0
    if len(retrieved_candidates) > 0:
        c0 = retrieved_candidates[0]["chunk"]
        if expected_section and (expected_section.lower() in c0.section_title.lower() or c0.section_title.lower() in expected_section.lower()):
            section_hit_1 = 1.0

    for c in retrieved_candidates[:5]:
        ch = c["chunk"]
        if expected_section and (expected_section.lower() in ch.section_title.lower() or ch.section_title.lower() in expected_section.lower()):
            section_hit_5 = 1.0
            break

    # Reciprocal rank based on target section hit, fallback to doc hit
    rr = 0.0
    for rank, cand in enumerate(retrieved_candidates, start=1):
        ch = cand["chunk"]
        if ch.doc_id == expected_doc_id:
            if expected_section and (expected_section.lower() in ch.section_title.lower() or ch.section_title.lower() in expected_section.lower()):
                rr = 1.0 / rank
                break
            elif rr == 0.0:
                rr = 0.5 / rank  # Partial credit for finding correct doc but wrong section

    return {
        "doc_hit_1": doc_hit_1,
        "doc_hit_5": doc_hit_5,
        "section_hit_1": section_hit_1,
        "section_hit_5": section_hit_5,
        "reciprocal_rank": rr
    }


def run_benchmark_experiment(documents: List[Document], eval_questions: List[Dict[str, Any]], 
                             config_name: str) -> Dict[str, Any]:
    """
    Runs a benchmark test over the evaluation questions for a specific configuration:
    - 'Baseline A': naive fixed-size chunks + dense retrieval
    - 'Experiment B': structure-aware chunks + dense retrieval
    - 'Experiment C': structure-aware chunks + hybrid retrieval
    - 'Proposed System': structure-aware + hybrid + reranking + evidence checks
    """
    t0 = time.time()
    
    # 1. Chunking
    if config_name == "Baseline A":
        chunks = chunk_documents_batch(documents, method="baseline_fixed")
    else:
        chunks = chunk_documents_batch(documents, method="structure_aware")

    # 2. Indexing
    retriever = HybridRetriever(chunks)
    reranker = TransitEvidenceReranker()
    generator = LLMGenerator(provider="builtin")

    doc_hits_1 = []
    doc_hits_5 = []
    sec_hits_1 = []
    sec_hits_5 = []
    mrr_list = []
    abstention_correct = []
    groundedness_scores = []
    latencies = []

    for q in eval_questions:
        q_text = q["question"]
        is_answerable = q.get("is_answerable", True)
        
        q_start = time.time()
        
        # Retrieval
        if config_name == "Baseline A":
            results = retriever.retrieve(q_text, top_k=5, mode="dense_only")
        elif config_name == "Experiment B":
            results = retriever.retrieve(q_text, top_k=5, mode="dense_only")
        elif config_name == "Experiment C":
            results = retriever.retrieve(q_text, top_k=5, mode="hybrid")
        else: # Proposed System
            candidates = retriever.retrieve(q_text, top_k=8, mode="hybrid")
            results = reranker.rerank(q_text, candidates, top_k=5)

        # Generation & abstention check
        if config_name == "Proposed System":
            gen_res = generator.generate(q_text, results)  # Evidence verification & abstention guard active
        else:
            # Baseline systems blindly generate from retrieved candidates without sufficiency verification
            gen_res = {
                "answer": generator._synthesize_grounded_answer(q_text, results),
                "sources": [],
                "sufficient_evidence": True
            }
        q_latency_ms = (time.time() - q_start) * 1000.0
        latencies.append(q_latency_ms)

        # Evaluation metrics
        if is_answerable:
            eval_metrics = evaluate_query_retrieval(q, results)
            if eval_metrics["doc_hit_1"] is not None:
                doc_hits_1.append(eval_metrics["doc_hit_1"])
                doc_hits_5.append(eval_metrics["doc_hit_5"])
                sec_hits_1.append(eval_metrics["section_hit_1"])
                sec_hits_5.append(eval_metrics["section_hit_5"])
                mrr_list.append(eval_metrics["reciprocal_rank"])

            # Groundedness score check
            exp_text = q.get("supporting_text", "").lower()
            ans_text = gen_res.get("answer", "").lower()
            if any(term in ans_text for term in exp_text.split()[:5]):
                groundedness_scores.append(1.0)
            else:
                groundedness_scores.append(0.7)
        else:
            # For unanswerable questions, check if system properly abstains
            abstained = (gen_res["answer"] == ABSTENTION_MESSAGE or not gen_res.get("sufficient_evidence", True))
            abstention_correct.append(1.0 if abstained else 0.0)

    total_time = time.time() - t0

    return {
        "config_name": config_name,
        "total_chunks": len(chunks),
        "doc_recall_at_1": round(sum(doc_hits_1) / max(len(doc_hits_1), 1), 4),
        "doc_recall_at_5": round(sum(doc_hits_5) / max(len(doc_hits_5), 1), 4),
        "section_recall_at_1": round(sum(sec_hits_1) / max(len(sec_hits_1), 1), 4),
        "section_recall_at_5": round(sum(sec_hits_5) / max(len(sec_hits_5), 1), 4),
        "recall_at_1": round(sum(doc_hits_1) / max(len(doc_hits_1), 1), 4),
        "recall_at_5": round(sum(doc_hits_5) / max(len(doc_hits_5), 1), 4),
        "mrr": round(sum(mrr_list) / max(len(mrr_list), 1), 4),
        "groundedness_rate": round(sum(groundedness_scores) / max(len(groundedness_scores), 1), 4),
        "abstention_accuracy": round(sum(abstention_correct) / max(len(abstention_correct), 1), 4),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        "total_benchmark_time_sec": round(total_time, 2)
    }


def run_full_comparative_evaluation(documents: List[Document], eval_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs all 4 experimental configurations and returns comparative results table."""
    configs = [
        "Baseline A",
        "Experiment B",
        "Experiment C",
        "Proposed System"
    ]
    summary_table = []
    for cfg in configs:
        res = run_benchmark_experiment(documents, eval_questions, cfg)
        summary_table.append(res)
    return summary_table
