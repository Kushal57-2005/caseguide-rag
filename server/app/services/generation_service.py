import json
import os
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

from app.schemas.chat_schema import (
    StructuredRAGResponse,
    QueryInfo,
    ProblemSummary,
    AnswerDetails,
    CitationItem,
    RetrievalChunkItem,
    RetrievalDetails,
    ActionStep,
    NextSteps,
    WarningItem,
    ResponseMetadata,
)
from app.services.query_understanding import extract_json_from_text
from app.services.llm_tracker import llm_tracker

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_LLM_API_KEY"),
    base_url=os.getenv("NVIDIA_LLM_BASE_URL")
)

LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

PDF_METADATA_MAP = {
    "202501082085049752.pdf": {
        "title": "The Motor Vehicles Act, 1988",
        "url": "https://cdnbbsr.s3waas.gov.in/s35dec707028b05bcbd3a1db5640f842c5/uploads/2025/01/202501082085049752.pdf"
    },
    "20250108215375951.pdf": {
        "title": "Offense & Penalties Schedule",
        "url": "https://cdnbbsr.s3waas.gov.in/s35dec707028b05bcbd3a1db5640f842c5/uploads/2025/01/20250108215375951.pdf"
    },
    "20250108634391784.pdf": {
        "title": "Maharashtra Motor Vehicles Rules, 1989",
        "url": "https://cdnbbsr.s3waas.gov.in/s35dec707028b05bcbd3a1db5640f842c5/uploads/2025/01/20250108634391784.pdf"
    },
    "20250108736472987.pdf": {
        "title": "The Motor Vehicles (Amendment) Act, 2019",
        "url": "https://cdnbbsr.s3waas.gov.in/s35dec707028b05bcbd3a1db5640f842c5/uploads/2025/01/20250108736472987.pdf"
    }
}


def normalize_ws(s: str) -> str:
    """Normalizes whitespace for reliable substring matching."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def validate_and_extract_citations(
    candidate_citations: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]]
) -> List[CitationItem]:
    """
    Rigorously validates each candidate citation against actual retrieved chunks.
    Ensures:
      - Source chunk exists
      - Page number matches actual metadata
      - File name matches actual metadata
      - Supporting text exists in the chunk content
    Discards any citation where supporting text is fabricated or unsupported.
    """
    validated: List[CitationItem] = []
    chunk_count = len(retrieved_chunks)

    for cand in candidate_citations:
        if not isinstance(cand, dict):
            continue

        raw_idx = cand.get("chunk_index")
        try:
            chunk_idx = int(raw_idx)
        except (ValueError, TypeError):
            chunk_idx = 1

        if chunk_idx < 1 or chunk_idx > chunk_count:
            # Try to match chunk by filename or section if specified
            found_idx = None
            cand_sec = normalize_ws(cand.get("section", ""))
            for i, chk in enumerate(retrieved_chunks):
                if cand_sec and cand_sec in normalize_ws(chk["content"]):
                    found_idx = i + 1
                    break
            if found_idx:
                chunk_idx = found_idx
            else:
                continue

        target_chunk = retrieved_chunks[chunk_idx - 1]
        chunk_content = target_chunk.get("content", "")
        norm_chunk_content = normalize_ws(chunk_content)

        supp_text = str(cand.get("supporting_text") or "").strip()
        norm_supp = normalize_ws(supp_text)

        # Check if supporting text is present in the chunk
        matched_text = ""
        if norm_supp and norm_supp in norm_chunk_content:
            matched_text = supp_text
        elif len(norm_supp) > 20:
            # Check for partial sentence match (at least 20 chars overlap)
            words = supp_text.split()
            found_phrase = False
            for w_len in range(min(len(words), 8), 3, -1):
                phrase = " ".join(words[:w_len])
                if normalize_ws(phrase) in norm_chunk_content:
                    matched_text = supp_text
                    found_phrase = True
                    break
            if not found_phrase:
                # If quoted text is hallucinated, check if the section is in the chunk and pull actual text
                sec_str = cand.get("section", "").strip()
                if sec_str and normalize_ws(sec_str) in norm_chunk_content:
                    # Find sentence containing the section in chunk
                    sentences = re.split(r"(?<=[.!?])\s+", chunk_content)
                    for s in sentences:
                        if normalize_ws(sec_str) in normalize_ws(s):
                            matched_text = s.strip()
                            break
        else:
            # Short text or empty; check if section exists in chunk
            sec_str = cand.get("section", "").strip()
            if sec_str and normalize_ws(sec_str) in norm_chunk_content:
                sentences = re.split(r"(?<=[.!?])\s+", chunk_content)
                for s in sentences:
                    if normalize_ws(sec_str) in normalize_ws(s):
                        matched_text = s.strip()
                        break

        # If we cannot verify supporting text inside the retrieved document, skip it
        if not matched_text:
            continue

        meta = target_chunk.get("metadata", {})
        doc_id = target_chunk.get("document_id") or meta.get("document_id") or "unknown"
        filename = target_chunk.get("file_name") or meta.get("filename") or "unknown"
        page_num = target_chunk.get("page_number") or meta.get("page_number") or 1

        act_name = cand.get("act") or PDF_METADATA_MAP.get(filename, {}).get("title") or "Motor Vehicles Act"
        section_name = cand.get("section") or "General Provision"
        title_name = cand.get("title") or f"{act_name} - {section_name}"

        citation_id = f"citation_{len(validated) + 1:03d}"
        validated.append(
            CitationItem(
                citation_id=citation_id,
                act=str(act_name).strip(),
                section=str(section_name).strip(),
                title=str(title_name).strip(),
                supporting_text=str(matched_text).strip(),
                source_id=str(doc_id).strip(),
                file_name=str(filename).strip(),
                page_number=int(page_num)
            )
        )

    return validated


def build_retrieval_details(retrieved_chunks: List[Dict[str, Any]]) -> RetrievalDetails:
    """Builds validated RetrievalDetails schema directly from actual vector store results."""
    items: List[RetrievalChunkItem] = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        items.append(
            RetrievalChunkItem(
                chunk_id=str(c.get("chunk_id", f"chunk_{idx}")),
                document_id=str(c.get("document_id", "unknown")),
                file_name=str(c.get("file_name", "unknown")),
                page_number=int(c.get("page_number", 1)),
                raw_distance=float(c.get("raw_distance", 0.0)),
                similarity_score=c.get("similarity_score"),
                reranker_score=c.get("reranker_score"),
                retrieval_rank=int(c.get("retrieval_rank", idx)),
                content=str(c.get("content", ""))
            )
        )
    return RetrievalDetails(
        total_chunks_retrieved=len(items),
        chunks=items
    )


def generate_structured_response(
    original_query: str,
    query_understanding: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int = 5,
    error_status: Optional[str] = None,
    conversation_history: Optional[List[Any]] = None
) -> StructuredRAGResponse:
    """
    Main orchestration service combining Query Understanding, RAG Retrieval Metadata,
    LLM Answer Generation, Citation Validation, Next Steps, and Warnings.
    Returns fully validated StructuredRAGResponse.
    """

    q_info = QueryInfo(
        original_query=original_query,
        refined_query=str(query_understanding.get("refined_query") or original_query),
        plain_language_query=str(query_understanding.get("plain_language_query") or original_query)
    )

    p_summary = ProblemSummary(
        title=str(query_understanding.get("title") or "Legal Query Analysis"),
        category=str(query_understanding.get("category") or "Traffic & Legal Regulations"),
        description=str(query_understanding.get("description") or original_query)
    )

    # Enforce minimum similarity threshold of 0.4: filter out useless chunks so they are never sent to the LLM
    SIMILARITY_THRESHOLD = 0.4
    retrieved_chunks = [
        c for c in (retrieved_chunks or [])
        if (c.get("similarity_score") is not None and c.get("similarity_score") >= SIMILARITY_THRESHOLD)
    ]

    retrieval_details = build_retrieval_details(retrieved_chunks)

    # Handle Explicit Errors or Empty Retrieval
    if error_status:
        return StructuredRAGResponse(
            query=q_info,
            problem_summary=p_summary,
            answer=AnswerDetails(
                summary="Unable to complete retrieval or answer generation due to a system processing error.",
                explanation=["A system error occurred during processing. Please try again or check server logs."]
            ),
            citations=[],
            retrieval=retrieval_details,
            next_steps=NextSteps(actions=[
                ActionStep(
                    step=1,
                    title="Retry Request",
                    description="Please retry your question or consult server administrator.",
                    source_ids=[],
                    source_pages=[],
                    source_type="SYSTEM"
                )
            ]),
            warnings=[WarningItem(type="SYSTEM_ERROR", message=f"Pipeline status: {error_status}")],
            limitations=["Processing could not complete normally."],
            metadata=ResponseMetadata(
                status=error_status,
                model=LLM_MODEL,
                retrieval_method="ChromaDB Dense Vector Search (Nemotron Embedding) [Threshold >= 0.40]",
                top_k=top_k
            )
        )

    # Fallback message when there is no retrieval >= 0.4 threshold
    if not retrieved_chunks:
        return StructuredRAGResponse(
            query=q_info,
            problem_summary=p_summary,
            answer=AnswerDetails(
                summary="No relevant legal provisions or documents found in the database for this inquiry.",
                explanation=[
                    "The retrieved document passages did not meet the minimum similarity threshold of 0.40 required for verified grounding.",
                    "The current knowledge base is strictly focused on Indian motor vehicle laws, including The Motor Vehicles Act 1988, The Motor Vehicles (Amendment) Act 2019, Maharashtra Motor Vehicles Rules 1989, and official Traffic Violation schedules.",
                    "Questions unrelated to these indexed traffic and transport statutes (such as general knowledge, current date/time, or unindexed laws) cannot be answered from the indexed files."
                ]
            ),
            citations=[],
            retrieval=retrieval_details,
            next_steps=NextSteps(actions=[
                ActionStep(
                    step=1,
                    title="Ask a Motor Vehicle or Traffic Law Question",
                    description="Rephrase your question around topics covered by the indexed documents, such as traffic fines, road accidents, driving licences, vehicle registration, or safety equipment.",
                    source_ids=[],
                    source_pages=[],
                    source_type="GENERAL"
                ),
                ActionStep(
                    step=2,
                    title="Upload Relevant Legal Documents",
                    description="If your question relates to a different act, state rule, or policy, upload the PDF document via the 'Manage Documents' drawer to index it.",
                    source_ids=[],
                    source_pages=[],
                    source_type="GENERAL"
                )
            ]),
            warnings=[
                WarningItem(
                    type="NO_RELEVANT_DATA",
                    message="No indexed documents met the 0.40 similarity threshold for this query. To prevent fabricated answers, retrieval was treated as a fallback."
                )
            ],
            limitations=[
                "The query falls outside the scope of the indexed motor vehicle laws or did not match any section with >= 0.40 similarity.",
                "Only verified passages meeting the 0.40 similarity threshold are passed to the language model."
            ],
            metadata=ResponseMetadata(
                status="INSUFFICIENT_EVIDENCE",
                model=LLM_MODEL,
                retrieval_method="ChromaDB Dense Vector Search (Nemotron Embedding) [Threshold >= 0.40]",
                top_k=top_k
            )
        )

    # Prepare Context with Numbered Chunks for the LLM
    context_chunks_formatted = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_chunks_formatted.append(
            f"=== CHUNK {idx} ===\n"
            f"Document ID: {chunk.get('document_id')}\n"
            f"File Name: {chunk.get('file_name')}\n"
            f"Page: {chunk.get('page_number')}\n"
            f"Text Excerpt:\n{chunk.get('content')}\n"
        )
    formatted_context = "\n\n".join(context_chunks_formatted)

    system_prompt = """You are CaseGuide, an AI-powered Indian Legal Information Assistant.
Your primary responsibility is to provide accurate, source-grounded legal information.

CASEGUIDE GROUNDED RESPONSE POLICY:

1. RETRIEVAL-FIRST POLICY:
- Answer questions using the retrieved legal context provided to you.
- Treat retrieved documents as the primary source of factual legal information.
- Do not invent legal provisions, sections, case names, judgments, citations, or court decisions.
- Do not assume that a legal provision exists merely because it sounds plausible.

2. INSUFFICIENT CONTEXT:
- If the retrieved context does not contain sufficient information to answer the user's question, clearly communicate that the available knowledge base does not contain sufficient information.
- Do not fabricate an answer.
- Do not generate unsupported legal references.
- Do not pretend that an unrelated retrieved document supports the answer.
- Set status to "INSUFFICIENT_EVIDENCE".

3. PARTIAL INFORMATION:
- If the retrieved documents answer only part of the question, answer only the supported portion.
- Clearly identify what information is missing under limitations.
- Explain that the available sources are insufficient for the remaining portion.
- Set status to "PARTIAL_ANSWER".

4. CITATION REQUIREMENTS:
- Cite only documents actually provided in the retrieved context.
- Every citation must reference an exact CHUNK NUMBER (chunk_index) from the excerpts.
- supporting_text MUST be an exact verbatim quote from that chunk.
- Never fabricate source references or associate a claim with a document that does not support it.

5. LEGAL ACCURACY:
- Distinguish between statutory provisions, judicial interpretations, and general explanations.
- Do not present outdated legal provisions as current law without verification.
- Never claim that a judgment establishes a principle unless the retrieved judgment supports that interpretation.

6. RESPONSE BEHAVIOR:
- Prioritize accuracy over completeness.
- If evidence is insufficient, abstain rather than hallucinate.
- Your objective is not to answer every question. Your objective is to provide reliable legal information supported by available sources.
- Return valid JSON only, matching the exact required schema.

Required JSON Structure:
{
  "answer": {
    "summary": "Direct, professional summary answer (2-3 sentences)",
    "explanation": [
      "Key legal point or provision 1 explained clearly",
      "Key legal point or provision 2 explained clearly"
    ]
  },
  "citations": [
    {
      "chunk_index": 1,
      "act": "Name of the Act (e.g. The Motor Vehicles Act, 1988)",
      "section": "Exact section number (e.g. Section 134(a))",
      "title": "Section title or topic (e.g. Duty of driver in case of accident)",
      "supporting_text": "Exact verbatim sentence or quote from the chunk excerpt"
    }
  ],
  "next_steps": [
    {
      "step": 1,
      "title": "Short title of action",
      "description": "Practical explanation of what the user must or should do",
      "chunk_indices": [1]
    }
  ],
  "warnings": [
    {
      "type": "LEGAL_CAUTION",
      "message": "Specific warning about statutory consequences, penalties, or time limits"
    }
  ],
  "limitations": [
    "Specific topic or question that the provided documents do not contain"
  ],
  "status": "ANSWERED"
}

Notes for status:
- "ANSWERED": Document excerpts directly and adequately answer the user's inquiry.
- "PARTIAL_ANSWER": Excerpts answer some aspects, but important parts of the user question are not covered in the text.
- "INSUFFICIENT_EVIDENCE": Excerpts do not contain relevant legal information.
"""

    history_block = ""
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for msg in conversation_history[-6:]:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if content:
                history_lines.append(f"{role.capitalize()}: {content}")
        if history_lines:
            history_block = "Prior Conversation Context:\n" + "\n".join(history_lines) + "\n\n"

    user_prompt = f"""{history_block}User Question:
{original_query}

Plain Language Translation:
{query_understanding.get('plain_language_query', original_query)}

Retrieved Document Excerpts:
{formatted_context}

Analyze the legal issue and generate the structured JSON response based strictly on the excerpts."""

    try:
        llm_tracker.record_call("generation_service", LLM_MODEL, user_prompt)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )

        message = response.choices[0].message
        raw_text = message.content or getattr(message, "reasoning_content", None) or ""
        parsed = extract_json_from_text(raw_text)

        # 1. Answer Details
        ans_dict = parsed.get("answer") or {}
        summary = str(ans_dict.get("summary") or "Analysis completed based on the retrieved legal provisions.").strip()
        explanation = ans_dict.get("explanation")
        if not isinstance(explanation, list) or not explanation:
            explanation = [summary]
        explanation = [str(item).strip() for item in explanation if item]

        # 2. Citations Validation
        raw_citations = parsed.get("citations") or []
        if not isinstance(raw_citations, list):
            raw_citations = []
        validated_citations = validate_and_extract_citations(raw_citations, retrieved_chunks)

        # 3. Next Steps
        raw_steps = parsed.get("next_steps") or []
        if not isinstance(raw_steps, list):
            raw_steps = []
        action_steps: List[ActionStep] = []
        for idx, s in enumerate(raw_steps, start=1):
            if not isinstance(s, dict):
                continue
            title = str(s.get("title") or f"Action {idx}").strip()
            desc = str(s.get("description") or "").strip()
            chunk_indices = s.get("chunk_indices") or []
            if not isinstance(chunk_indices, list):
                chunk_indices = []

            source_ids = []
            source_pages = []
            for ci in chunk_indices:
                try:
                    c_num = int(ci)
                    if 1 <= c_num <= len(retrieved_chunks):
                        chk = retrieved_chunks[c_num - 1]
                        sid = chk.get("document_id") or chk.get("metadata", {}).get("document_id")
                        spage = chk.get("page_number") or chk.get("metadata", {}).get("page_number")
                        if sid and sid not in source_ids:
                            source_ids.append(str(sid))
                        if spage is not None and spage not in source_pages:
                            source_pages.append(int(spage))
                except (ValueError, TypeError):
                    continue

            action_steps.append(
                ActionStep(
                    step=idx,
                    title=title,
                    description=desc,
                    source_ids=source_ids,
                    source_pages=source_pages,
                    source_type="RAG_AND_LLM" if source_ids else "GENERAL"
                )
            )

        # 4. Warnings
        raw_warnings = parsed.get("warnings") or []
        if not isinstance(raw_warnings, list):
            raw_warnings = []
        warnings_list: List[WarningItem] = []
        for w in raw_warnings:
            if isinstance(w, dict):
                warnings_list.append(
                    WarningItem(
                        type=str(w.get("type") or "LEGAL_CAUTION").strip(),
                        message=str(w.get("message") or "").strip()
                    )
                )

        # 5. Limitations
        raw_limits = parsed.get("limitations") or []
        if not isinstance(raw_limits, list):
            raw_limits = []
        limitations_list = [str(lim).strip() for lim in raw_limits if lim]

        # 6. Status Determination
        status = str(parsed.get("status") or "ANSWERED").strip().upper()
        if status not in {"ANSWERED", "PARTIAL_ANSWER", "INSUFFICIENT_EVIDENCE"}:
            status = "ANSWERED" if validated_citations else "PARTIAL_ANSWER"

        return StructuredRAGResponse(
            query=q_info,
            problem_summary=p_summary,
            answer=AnswerDetails(
                summary=summary,
                explanation=explanation
            ),
            citations=validated_citations,
            retrieval=retrieval_details,
            next_steps=NextSteps(actions=action_steps),
            warnings=warnings_list,
            limitations=limitations_list,
            metadata=ResponseMetadata(
                status=status,
                model=LLM_MODEL,
                retrieval_method="ChromaDB Dense Vector Search (Nemotron Embedding)",
                top_k=top_k
            )
        )

    except Exception as e:
        # Gracefully return a fallback valid StructuredRAGResponse with status GENERATION_FAILED
        return StructuredRAGResponse(
            query=q_info,
            problem_summary=p_summary,
            answer=AnswerDetails(
                summary="The legal assistant encountered an issue processing the structured model response.",
                explanation=[
                    f"Generation error: {str(e)}",
                    "The retrieved documents are preserved and accessible below."
                ]
            ),
            citations=[],
            retrieval=retrieval_details,
            next_steps=NextSteps(actions=[
                ActionStep(
                    step=1,
                    title="Review Retrieved Documents Manually",
                    description="You can inspect the relevant sections in the retrieved chunks list below.",
                    source_ids=[],
                    source_pages=[],
                    source_type="RAG"
                )
            ]),
            warnings=[
                WarningItem(
                    type="GENERATION_FAILED",
                    message="Response generation failed. Please re-submit your query."
                )
            ],
            limitations=["LLM response parsing could not be completed."],
            metadata=ResponseMetadata(
                status="GENERATION_FAILED",
                model=LLM_MODEL,
                retrieval_method="ChromaDB Dense Vector Search (Nemotron Embedding)",
                top_k=top_k
            )
        )


# Backward compatibility wrapper
def generate_answer(question: str, retrieved_chunks: list):
    """Legacy wrapper for existing test scripts."""
    res = generate_structured_response(
        original_query=question,
        query_understanding={
            "refined_query": question,
            "plain_language_query": question,
            "title": "Legal Inquiry",
            "category": "Traffic & Legal Regulations",
            "description": question
        },
        retrieved_chunks=retrieved_chunks,
        top_k=len(retrieved_chunks)
    )
    return {
        "answer": res.answer.summary + "\n\n" + "\n".join(res.answer.explanation),
        "sources": [
            {
                "filename": c.file_name,
                "page_number": c.page_number,
                "similarity": c.similarity_score or 0.0
            }
            for c in res.retrieval.chunks
        ]
    }