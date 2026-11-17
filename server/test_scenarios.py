import os, sys
sys.path.insert(0, r"C:\Users\kusha\OneDrive\Desktop\caseGuide_backend\server")
os.chdir(r"C:\Users\kusha\OneDrive\Desktop\caseGuide_backend\server")
from dotenv import load_dotenv
load_dotenv(".env")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_scenario_1():
    print("\n========================================================")
    print("SCENARIO 1: Standard Legal Question (Section 134 Driver Duties)")
    print("========================================================")
    payload = {
        "question": "What are the legal duties of a driver in case of an accident where a person is injured?",
        "top_k": 3
    }
    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    print("Metadata Status:", data["metadata"]["status"])
    print("Problem Title:", data["problem_summary"]["title"])
    print("Category:", data["problem_summary"]["category"])
    print("Refined Query:", data["query"]["refined_query"])
    print("Plain Language Query:", data["query"]["plain_language_query"])
    print("Answer Summary:\n", data["answer"]["summary"])
    print("\nAnswer Explanation:")
    for pt in data["answer"]["explanation"]:
        print("  *", pt)
    print(f"\nCitations ({len(data['citations'])}):")
    for c in data["citations"]:
        print(f"  - [{c['citation_id']}] {c['act']} {c['section']}: {c['title']} (Doc: {c['file_name']}, Pg {c['page_number']})")
        print(f"    Quote: {c['supporting_text'][:90]}...")
    print(f"\nRetrieved Chunks ({len(data['retrieval']['chunks'])}):")
    for ch in data["retrieval"]["chunks"]:
        print(f"  - #{ch['retrieval_rank']} | Raw Dist: {ch['raw_distance']} | Sim: {ch['similarity_score']} | Pg {ch['page_number']} | Doc: {ch['file_name']}")
    print(f"\nNext Steps ({len(data['next_steps']['actions'])}):")
    for s in data["next_steps"]["actions"]:
        print(f"  - Step {s['step']}: {s['title']} ({s['source_type']}) - {s['description'][:80]}...")
    print(f"\nWarnings ({len(data['warnings'])}):")
    for w in data["warnings"]:
        print(f"  - {w['type']}: {w['message']}")
    print("\nLimitations:", data["limitations"])
    assert data["metadata"]["status"] in ["ANSWERED", "PARTIAL_ANSWER"]
    print("\n>>> Scenario 1 PASSED!")


def test_scenario_2():
    print("\n========================================================")
    print("SCENARIO 2: Good Samaritan Protection & Document Production")
    print("========================================================")
    payload = {
        "question": "What legal protections exist for Good Samaritans who help accident victims under Section 134A?",
        "top_k": 3
    }
    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    print("Metadata Status:", data["metadata"]["status"])
    print("Answer Summary:\n", data["answer"]["summary"])
    print(f"Citations count: {len(data['citations'])}")
    for c in data["citations"]:
        print(f"  - {c['act']} {c['section']} (Pg {c['page_number']})")
    assert data["metadata"]["status"] in ["ANSWERED", "PARTIAL_ANSWER"]
    print(">>> Scenario 2 PASSED!")


def test_scenario_3():
    print("\n========================================================")
    print("SCENARIO 3: Question with Insufficient Evidence")
    print("========================================================")
    payload = {
        "question": "What are the rules and licensing requirements for deep-sea seabed polymetallic nodule mining in Antarctica under the International Seabed Authority?",
        "top_k": 3
    }
    res = client.post("/api/chat/", json=payload)
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    print("Metadata Status:", data["metadata"]["status"])
    print("Answer Summary:\n", data["answer"]["summary"])
    print(f"Warnings ({len(data['warnings'])}):", [w["type"] for w in data["warnings"]])
    print("Limitations:", data["limitations"])
    assert data["metadata"]["status"] in ["INSUFFICIENT_EVIDENCE", "PARTIAL_ANSWER"]
    print(">>> Scenario 3 PASSED!")


if __name__ == "__main__":
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    print("\nALL SCENARIOS COMPLETED SUCCESSFULLY!")
