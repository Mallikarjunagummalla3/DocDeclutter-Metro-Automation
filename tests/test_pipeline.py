"""
Unit and Integration Tests for DocDeclutter Pipeline
Verifies Ingestion, Structure-Aware Chunking, Hybrid Retrieval,
Reranking, Abstention Guard, and Grounded Generation.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ingestion import ingest_batch, Document
from src.chunking import structure_aware_chunking, baseline_fixed_chunking, chunk_documents_batch
from src.retrieval import HybridRetriever
from src.reranking import TransitEvidenceReranker
from src.generation import LLMGenerator, ABSTENTION_MESSAGE, sufficient_evidence
from src.dataset_generator import DOCS_DIR, EVAL_FILE


class TestDocDeclutterPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.docs = ingest_batch(DOCS_DIR)
        cls.chunks = chunk_documents_batch(cls.docs, method="structure_aware")
        cls.retriever = HybridRetriever(cls.chunks)
        cls.reranker = TransitEvidenceReranker()
        cls.generator = LLMGenerator(provider="builtin")

    def test_01_document_ingestion(self):
        """Verify that documents are ingested with metadata and non-empty content."""
        self.assertGreater(len(self.docs), 0, "No documents were ingested.")
        for doc in self.docs:
            self.assertTrue(doc.doc_id, f"Document {doc.filepath} has empty doc_id")
            self.assertGreater(len(doc.content), 50, f"Document {doc.doc_id} content is too short")
            self.assertGreaterEqual(len(doc.pages), 1, f"Document {doc.doc_id} has no pages")

    def test_02_structure_aware_chunking(self):
        """Verify that structure-aware chunking preserves section titles and metadata."""
        self.assertGreater(len(self.chunks), 0, "No chunks were produced.")
        has_section_title = any("SECTION" in c.section_title.upper() for c in self.chunks)
        self.assertTrue(has_section_title, "No chunks preserved a SECTION title.")
        
        # Verify contextual_text format
        sample_chunk = self.chunks[0]
        self.assertIn("[Document:", sample_chunk.contextual_text)
        self.assertIn("Section:", sample_chunk.contextual_text)
        self.assertIn("Page:", sample_chunk.contextual_text)

    def test_03_hybrid_retrieval_aluva_incident(self):
        """Verify hybrid retrieval surfaces the Aluva incident document for Aluva delay queries."""
        query = "What caused the metro service delay on 14 October 2024 at Aluva Station?"
        results = self.retriever.retrieve(query, top_k=5, mode="hybrid")
        self.assertGreater(len(results), 0, "Hybrid retrieval returned empty results.")
        
        top_doc_ids = [c["chunk"].doc_id for c in results]
        self.assertIn("KMRL-INC-2024-089", top_doc_ids, "Expected KMRL-INC-2024-089 in top results.")

    def test_04_hybrid_retrieval_technical_code(self):
        """Verify exact transit equipment code TM-104 is matched accurately."""
        query = "What are the temperature thresholds for TM-104 traction motor bearings?"
        results = self.retriever.retrieve(query, top_k=5, mode="hybrid")
        self.assertGreater(len(results), 0)
        
        top_doc_ids = [c["chunk"].doc_id for c in results]
        self.assertIn("KMRL-RS-TM-003", top_doc_ids, "Expected KMRL-RS-TM-003 for TM-104 query.")

    def test_05_transit_reranker(self):
        """Verify transit reranker promotes section-aligned and entity-matching candidates."""
        query = "What is the root cause of the Aluva service delay?"
        candidates = self.retriever.retrieve(query, top_k=6, mode="hybrid")
        reranked = self.reranker.rerank(query, candidates, top_k=3)
        
        self.assertGreater(len(reranked), 0)
        top_cand = reranked[0]["chunk"]
        self.assertEqual(top_cand.doc_id, "KMRL-INC-2024-089")

    def test_06_abstention_on_irrelevant_query(self):
        """Verify system abstains when queried on unanswerable out-of-scope topics."""
        unanswerable_query = "What is the price of a ticket from Delhi Metro Rajiv Chowk to Noida?"
        candidates = self.retriever.retrieve(unanswerable_query, top_k=3, mode="hybrid")
        res = self.generator.generate(unanswerable_query, candidates)
        
        self.assertEqual(res["answer"], ABSTENTION_MESSAGE, "System failed to abstain on out-of-scope query.")
        self.assertFalse(res["sufficient_evidence"], "Sufficiency check did not flag insufficient evidence.")

    def test_07_grounded_answer_generation(self):
        """Verify answer generation includes citations and facts for a valid query."""
        query = "What is the maximum speed permitted in Restricted Manual (RM) mode?"
        candidates = self.retriever.retrieve(query, top_k=3, mode="hybrid")
        res = self.generator.generate(query, candidates)
        
        self.assertTrue(res["sufficient_evidence"])
        self.assertIn("25 km/h", res["answer"], "Answer did not contain expected speed limit.")
        self.assertGreater(len(res["sources"]), 0, "No sources were returned.")


if __name__ == "__main__":
    unittest.main()

