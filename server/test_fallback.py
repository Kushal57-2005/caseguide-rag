import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))
os.chdir(str(SERVER_DIR))

from dotenv import load_dotenv
load_dotenv(".env")

from fastapi.testclient import TestClient
from app.main import app
from app.schemas.chat_schema import FALLBACK_MESSAGE
from app.services.llm_tracker import llm_tracker
from app.services.retrieval_service import SIMILARITY_THRESHOLD

client = TestClient(app)


def test_zero_retrieved_documents():
    """
    Test 1: Zero retrieved documents.
    Filtering ChromaDB by a non-existent document_id returns 0 chunks.
    Proves:
      - status == 'fallback'
      - response == predefined FALLBACK_MESSAGE
      - sources == []
      - LLM calls == 0 (no query refinement LLM, no answer generation LLM).
    """
    print("\n" + "=" * 65)
    print("TEST 1: Zero Retrieved Documents (Filtering by nonexistent doc_id)")
    print("=" * 65)
    llm_tracker.reset()

    payload = {
        "question": "What are the rules regarding driving licences?",
        "top_k": 3,
        "document_id": "nonexistent-doc-id-99999"
    }

    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"HTTP Error: {res.text}"
    data = res.json()

    print(f"Response Status: {data.get('status')}")
    print(f"Response Sources: {data.get('sources')}")
    print(f"LLM Calls Recorded: {llm_tracker.get_count()}")

    assert data["status"] == "fallback", f"Expected 'fallback', got {data.get('status')}"
    assert data["response"] == FALLBACK_MESSAGE, "Fallback message does not match exact specification"
    assert data["sources"] == [], f"Expected empty sources list, got {data['sources']}"
    assert "query" not in data, "Fallback must NOT return a structured response"
    assert "citations" not in data, "Fallback must NOT return citations"
    assert llm_tracker.get_count() == 0, f"Critical violation: Expected 0 LLM calls, but got {llm_tracker.get_count()}"

    print(">>> TEST 1 PASSED: Zero retrieved documents -> Exactly 0 LLM calls.")


def test_all_documents_below_threshold():
    """
    Test 2: All documents below similarity threshold (0.40).
    An out-of-scope query ('What is the weather in Paris today?') retrieves chunks
    with similarity < 0.40, which are removed.
    Proves:
      - status == 'fallback'
      - response == predefined FALLBACK_MESSAGE
      - sources == []
      - LLM calls == 0 (no query refinement LLM, no answer generation LLM).
    """
    print("\n" + "=" * 65)
    print("TEST 2: All Documents Below Threshold (Out-of-Scope Query)")
    print("=" * 65)
    llm_tracker.reset()

    payload = {
        "question": "What is the current weather forecast and temperature in Paris today?",
        "top_k": 5
    }

    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"HTTP Error: {res.text}"
    data = res.json()

    print(f"Response Status: {data.get('status')}")
    print(f"Response Sources: {data.get('sources')}")
    print(f"LLM Calls Recorded: {llm_tracker.get_count()}")

    assert data["status"] == "fallback", f"Expected 'fallback', got {data.get('status')}"
    assert data["response"] == FALLBACK_MESSAGE, "Fallback message does not match exact specification"
    assert data["sources"] == [], f"Expected empty sources list, got {data['sources']}"
    assert "query" not in data, "Fallback must NOT return a structured response"
    assert "citations" not in data, "Fallback must NOT return citations"
    assert llm_tracker.get_count() == 0, f"Critical violation: Expected 0 LLM calls, but got {llm_tracker.get_count()}"

    print(">>> TEST 2 PASSED: All chunks < 0.40 -> Exactly 0 LLM calls.")


def test_relevant_documents_found():
    """
    Test 3: Relevant documents found (similarity >= 0.40).
    A core Motor Vehicle Law query ('What are driver duties in case of accident under Section 134?')
    retrieves high-similarity chunks (>= 0.40).
    Proves:
      - Status is ANSWERED or PARTIAL_ANSWER
      - Normal RAG execution proceeds
      - Citations and explanation generated
      - LLM calls > 0 (normal execution).
    """
    print("\n" + "=" * 65)
    print("TEST 3: Relevant Documents Found (In-Scope Query)")
    print("=" * 65)
    llm_tracker.reset()

    payload = {
        "question": "What are the legal duties of a driver in case of an accident where a person is injured?",
        "top_k": 3
    }

    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"HTTP Error: {res.text}"
    data = res.json()

    print(f"Response Status: {data.get('metadata', {}).get('status')}")
    print(f"Citations Found: {len(data.get('citations', []))}")
    print(f"LLM Calls Recorded: {llm_tracker.get_count()}")

    assert "status" not in data or data.get("status") != "fallback", "Relevant query should NOT trigger fallback"
    assert data["metadata"]["status"] in ["ANSWERED", "PARTIAL_ANSWER"]
    assert len(data.get("citations", [])) > 0, "Expected at least 1 verified citation"
    assert llm_tracker.get_count() > 0, f"Expected LLM calls for in-scope query, got {llm_tracker.get_count()}"

    print(f">>> TEST 3 PASSED: Relevant documents found -> Normal execution ({llm_tracker.get_count()} LLM calls).")


if __name__ == "__main__":
    print(f"Configured Similarity Threshold: {SIMILARITY_THRESHOLD}")
    test_zero_retrieved_documents()
    test_all_documents_below_threshold()
    test_relevant_documents_found()
    print("\n" + "=" * 65)
    print("ALL 3 FALLBACK VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
