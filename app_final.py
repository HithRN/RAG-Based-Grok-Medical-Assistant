# app_fixed.py
from flask import Flask, render_template, jsonify, request, Response
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import logging
import traceback
from typing import Any, Dict, List



load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Environment keys (make sure these are set in your environment or .env)
#PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. LLM calls will fail unless provided.")

# create Flask app
app = Flask(__name__)

# Embeddings and vector store setup (wrap in try to make failures obvious)
embeddings = None
docsearch = None
retriever = None
llm = None
rag_chain = None

try:
    embeddings = download_hugging_face_embeddings()
    index = "medbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(
        temperature=0.4,
        max_tokens=500,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("Vector store, retriever, LLm and RAG chain initialized.")
except Exception as e:
    logger.exception("Initialization error (embeddings/vectorstore/llm). Continue but /health will show issues.")
    # Keep app running so you can see meaningful errors in /get and /health
    rag_chain = None


def _safe_get_text_from_resp(resp: Any) -> str:
    """
    Accepts many response shapes and attempts to extract the 'answer' text.
    """
    if resp is None:
        return ""
    # If resp is already a string
    if isinstance(resp, str):
        return resp
    # Common keys
    for key in ("answer", "output_text", "text", "response", "result"):
        if isinstance(resp, dict) and key in resp and resp[key]:
            val = resp[key]
            if isinstance(val, str):
                return val
            # sometimes it's a list or nested dict -> coerce to string
            try:
                return str(val)
            except Exception:
                continue
    # If resp has 'choices' as in OpenAI style
    if isinstance(resp, dict) and "choices" in resp:
        choices = resp.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            # typical shape: {"text": "..."}
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            try:
                return str(first)
            except Exception:
                pass
    # fallback to stringifying the whole object
    try:
        return str(resp)
    except Exception:
        return ""


def _extract_sources(resp: Any) -> List[Dict]:
    """
    Look for common keys that contain source documents. Accepts both lists of Document objects
    or lists of dicts. Returns list of {"title":..., "snippet":..., "meta":...}
    """
    if resp is None:
        return []

    candidate_keys = (
        "source_documents",
        "source_docs",
        "sources",
        "retrieved_documents",
        "documents",
        "docs",
    )
    docs = None
    if isinstance(resp, dict):
        for k in candidate_keys:
            if k in resp and resp[k]:
                docs = resp[k]
                break

    # If no explicit key found, maybe the chain returned a tuple or object with attribute
    if docs is None:
        # try resp.get("raw", {}) variants
        if isinstance(resp, dict) and "raw" in resp and isinstance(resp["raw"], dict):
            for k in candidate_keys:
                if k in resp["raw"] and resp["raw"][k]:
                    docs = resp["raw"][k]
                    break

    # If still not found, try attributes (some LangChain outputs)
    if docs is None and hasattr(resp, "__dict__"):
        for k in candidate_keys:
            if hasattr(resp, k):
                docs = getattr(resp, k)
                break

    # If nothing found, return empty
    if not docs:
        return []

    out = []
    for s in docs:
        try:
            # Document-like object with .page_content and .metadata
            if hasattr(s, "page_content") or hasattr(s, "metadata"):
                page_content = getattr(s, "page_content", None)
                meta = getattr(s, "metadata", {}) or {}
                title = meta.get("title") or meta.get("source") or meta.get("filename") or meta.get("url") or meta.get("doc_id") or "doc"
                snippet = (page_content[:250] + "...") if page_content else ""
                out.append({"title": title, "snippet": snippet, "meta": dict(meta)})
            elif isinstance(s, dict):
                # dict-like doc structure
                page_content = s.get("page_content") or s.get("content") or s.get("text") or ""
                meta = s.get("metadata") or {}
                title = meta.get("title") or meta.get("source") or s.get("id") or "doc"
                snippet = (page_content[:250] + "...") if page_content else ""
                out.append({"title": title, "snippet": snippet, "meta": dict(meta)})
            else:
                # fallback for unknown shapes
                out.append({"title": "doc", "snippet": str(s)[:250] + "...", "meta": {}})
        except Exception:
            logger.debug("Failed to parse a source doc: %s", traceback.format_exc())
            # still append a safe string representation
            try:
                out.append({"title": "doc", "snippet": str(s)[:250] + "...", "meta": {}})
            except Exception:
                pass
    return out


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    problems = []
    ok = True
    if rag_chain is None:
        ok = False
        problems.append("RAG chain not initialized (check embeddings/vectorstore/LLM keys).")
    # you can add more checks here (e.g., ping vector DB)
    return jsonify({"ok": ok, "problems": problems})


@app.route("/get", methods=["POST"])
def chat():
    if rag_chain is None:
        return Response("RAG chain not initialized.", mimetype="text/plain"), 500

    try:
        # Get message safely
        if request.is_json:
            data = request.get_json()
            msg = data.get("msg") or data.get("input") or ""
        else:
            msg = request.form.get("msg") or request.args.get("msg") or ""

        if not msg.strip():
            return Response("Empty message.", mimetype="text/plain"), 400

        # Call chain
        try:
            resp = rag_chain.invoke({"input": msg})
        except TypeError:
            resp = rag_chain.invoke(msg)

        # Extract answer robustly
        answer = ""

        if isinstance(resp, str):
            answer = resp

        elif isinstance(resp, dict):
            for key in ["answer", "result", "output_text", "text", "response"]:
                if key in resp:
                    answer = str(resp[key])
                    break

            # fallback if nothing matched
            if not answer:
                answer = str(resp)

        else:
            answer = str(resp)

        # Return ONLY STRING
        return Response(answer, mimetype="text/plain")

    except Exception as e:
        return Response(f"Internal Error: {str(e)}", mimetype="text/plain"), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
