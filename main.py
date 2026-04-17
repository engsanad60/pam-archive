import os
import sys
print(f"[STARTUP] PORT env = {os.environ.get('PORT', 'NOT SET')} | Python {sys.version}", flush=True)

import asyncio
import hashlib
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from archive_logic import (
    ArchiveManager, CANCELLED_FILES,
    normalize_numbers, load_stored_text, chunk_text,
)
from chatbot_logic import extract_decree_from_question

logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
from chatbot_logic import ChatbotService  # noqa: E402 (after archive imports)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
load_dotenv(BASE_DIR / ".env")

# Ensure required runtime directories exist (important for Railway ephemeral FS)
for _d in ["uploads", "data", "data/chromadb", "static"]:
    (BASE_DIR / _d).mkdir(parents=True, exist_ok=True)


# Simple in-memory response cache (last 50 questions)
_response_cache: Dict[str, Any] = {}

def _cache_key(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()

def _get_cached(question: str) -> Optional[Dict[str, Any]]:
    return _response_cache.get(_cache_key(question))

def _set_cached(question: str, value: Dict[str, Any]) -> None:
    key = _cache_key(question)
    if len(_response_cache) >= 50:
        # Remove oldest entry
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[key] = value


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: re-index any documents missing from ChromaDB, then pre-warm Claude."""
    loop = asyncio.get_event_loop()
    try:
        count = await loop.run_in_executor(None, archive_manager.reindex_missing_documents)
        stats = archive_manager.get_index_stats()
        if count:
            logger.info("✅ تم إعادة فهرسة %d وثيقة عند بدء التشغيل", count)
        logger.info(
            "📚 ChromaDB جاهز: %d وثيقة في %d إدارة",
            stats.get("ready", stats.get("documents_count", 0)),
            stats.get("departments_count", 0),
        )
    except Exception as exc:
        logger.warning("Startup reindex failed (non-fatal): %s", exc)

    # Pre-warm Claude Haiku so first user request is fast
    try:
        await loop.run_in_executor(None, _prewarm_claude)
        logger.info("🔥 Claude pre-warm completed")
    except Exception as exc:
        logger.warning("Claude pre-warm failed (non-fatal): %s", exc)

    yield


def _prewarm_claude() -> None:
    """Make a tiny Claude call so the model is cached for first real request."""
    chatbot_service.raw_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": "hi"}],
    )


app = FastAPI(title="Arabic Archive + RAG Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
archive_manager = ArchiveManager(base_dir=BASE_DIR)
chatbot_service = ChatbotService(base_dir=BASE_DIR)

app.mount("/uploads", StaticFiles(directory=str(BASE_DIR / "uploads")), name="uploads")

_static_dir = BASE_DIR / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    source: Dict[str, Any] = {}
    show_source: bool = False
    contains_violation: bool = False


class SectionPayload(BaseModel):
    name_ar: str
    name_en: str


class YearPayload(BaseModel):
    year: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """User portal - main entry point for all users."""
    return templates.TemplateResponse(
        request=request,
        name="user_portal.html",
        context={"request": request},
    )


@app.get("/archive", include_in_schema=False)
async def archive_redirect() -> RedirectResponse:
    """Legacy redirect: old /archive URL → new admin archive page."""
    return RedirectResponse(url="/admin/archive")


@app.get("/chat", include_in_schema=False)
async def chat_redirect() -> RedirectResponse:
    return RedirectResponse(url="/")


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request) -> HTMLResponse:
    """Admin dashboard - home for administrators."""
    return templates.TemplateResponse(
        request=request,
        name="admin_dashboard.html",
        context={"request": request},
    )


@app.get("/admin/archive", response_class=HTMLResponse)
async def admin_archive_page(request: Request) -> HTMLResponse:
    """Admin archive management page."""
    return templates.TemplateResponse(
        request=request,
        name="admin_archive.html",
        context={"request": request},
    )


@app.get("/admin/departments", response_class=HTMLResponse)
async def admin_departments_page(request: Request) -> HTMLResponse:
    """Admin departments & sections management page."""
    return templates.TemplateResponse(
        request=request,
        name="admin_departments.html",
        context={"request": request},
    )


@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings_page(request: Request) -> HTMLResponse:
    """Admin settings page."""
    return templates.TemplateResponse(
        request=request,
        name="admin_settings.html",
        context={"request": request},
    )


@app.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="admin_logs.html",
        context={"request": request},
    )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={"request": request},
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        n = int(archive_manager.chroma_collection.count())
    except Exception:
        n = 0
    return {"status": "ok", "documents_indexed": n}


@app.get("/api/config")
async def get_config() -> Dict[str, str]:
    return archive_manager.get_config()


@app.put("/api/config")
async def update_config(payload: Dict[str, str]) -> Dict[str, str]:
    system_name_ar = payload.get("system_name_ar", "").strip()
    system_name_en = payload.get("system_name_en", "").strip()
    if not system_name_ar or not system_name_en:
        raise HTTPException(status_code=400, detail="Both Arabic and English names are required")
    return archive_manager.update_config(system_name_ar=system_name_ar, system_name_en=system_name_en)


@app.get("/api/departments")
async def get_departments() -> Dict[str, Any]:
    return {"departments": archive_manager.get_departments()}


@app.get("/api/years/predefined")
async def predefined_years() -> Dict[str, List[str]]:
    return {"years": archive_manager.get_predefined_years()}


@app.post("/api/years/predefined")
async def add_predefined_year(payload: YearPayload) -> Dict[str, Any]:
    year = payload.year.strip()
    if not re.match(r"^\d{4}$", year):
        raise HTTPException(status_code=400, detail="السنة يجب أن تكون 4 أرقام")
    years = archive_manager.get_predefined_years()
    if year not in years:
        years.append(year)
        years.sort()
        archive_manager._save_predefined_years(years)
    return {"message": f"تم إضافة سنة {year}", "years": years}


@app.post("/api/departments")
async def add_department(payload: Dict[str, str]) -> Dict[str, Any]:
    name_ar = payload.get("name_ar", "").strip()
    name_en = payload.get("name_en", "").strip()
    if not name_ar or not name_en:
        raise HTTPException(status_code=400, detail="Arabic and English names are required")
    department = archive_manager.add_department(name_ar=name_ar, name_en=name_en)
    return {"message": "Department added successfully", "department": department}


@app.put("/api/departments/{department_id}")
async def update_department(department_id: str, payload: Dict[str, str]) -> Dict[str, Any]:
    name_ar = payload.get("name_ar", "").strip()
    name_en = payload.get("name_en", "").strip()
    try:
        department = archive_manager.update_department(
            department_id=department_id,
            name_ar=name_ar,
            name_en=name_en,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Department updated successfully", "department": department}


@app.delete("/api/departments/{department_id}")
async def delete_department(department_id: str) -> Dict[str, str]:
    try:
        archive_manager.delete_department(department_id=department_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "Department deleted successfully"}


@app.post("/api/departments/{department_id}/sections")
async def add_section(department_id: str, payload: SectionPayload) -> Dict[str, Any]:
    try:
        sec = archive_manager.add_section(
            department_id=department_id,
            name_ar=payload.name_ar,
            name_en=payload.name_en,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Section added", "section": sec}


@app.put("/api/departments/{department_id}/sections/{section_id}")
async def update_section(department_id: str, section_id: str, payload: SectionPayload) -> Dict[str, Any]:
    try:
        sec = archive_manager.update_section(
            department_id=department_id,
            section_id=section_id,
            name_ar=payload.name_ar,
            name_en=payload.name_en,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Section updated", "section": sec}


@app.delete("/api/departments/{department_id}/sections/{section_id}")
async def delete_section(department_id: str, section_id: str) -> Dict[str, str]:
    try:
        archive_manager.delete_section(department_id=department_id, section_id=section_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "Section deleted"}


@app.post("/api/departments/{department_id}/sections/{section_id}/years")
async def add_section_year(department_id: str, section_id: str, payload: YearPayload) -> Dict[str, str]:
    try:
        archive_manager.add_year_to_section(
            department_id=department_id,
            section_id=section_id,
            year=payload.year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Year folder added"}


def _record_to_api_dict(r: Any) -> Dict[str, Any]:
    return {
        "id": r.id,
        "file_name": r.file_name,
        "department_id": r.department_id,
        "department_name_ar": r.department_name_ar,
        "department_name_en": r.department_name_en,
        "section_id": r.section_id,
        "section_name_ar": r.section_name_ar,
        "section_name_en": r.section_name_en,
        "year": r.year,
        "decree_number": r.decree_number,
        "pages_count": r.pages_count,
        "confidence": r.confidence,
        "document_type": r.document_type,
        "main_topic": r.main_topic,
        "summary": r.summary,
        "language": r.language,
        "upload_date": r.upload_date,
        "relative_path": r.relative_path,
        "ocr_used": r.ocr_used,
        "extraction_method": r.extraction_method,
        "status": r.status,
        "doc_type": r.doc_type,
        "structured_data": r.structured_data,
        "error_message": r.error_message,
        "original_filename": r.file_name,
    }


@app.get("/api/files")
async def list_files() -> Dict[str, Any]:
    files_data = archive_manager.list_files_by_department()
    by_dept: Dict[str, List[Dict[str, Any]]] = {}
    for dep_id, records in files_data.items():
        by_dept[dep_id] = [_record_to_api_dict(r) for r in records]
    flat = [_record_to_api_dict(r) for r in archive_manager.list_all_files()]
    return {"files_by_department": by_dept, "files": flat}


@app.get("/api/files/status")
async def files_status() -> Dict[str, Any]:
    """Lightweight polling endpoint - returns status of all files."""
    raw = archive_manager._load_files_list()
    result = []
    for f in raw:
        result.append({
            "id": f.get("id", ""),
            "status": f.get("status", "ready"),
            "original_filename": f.get("original_filename", f.get("file_name", "")),
            "department_name_ar": f.get("department_name_ar", f.get("department", "")),
            "section_name_ar": f.get("section_name_ar", f.get("section", "")),
            "year": f.get("year", ""),
            "document_type": f.get("document_type", ""),
            "doc_type": f.get("doc_type", ""),
            "decree_number": f.get("decree_number", ""),
            "confidence": f.get("confidence", ""),
            "extraction_method": f.get("extraction_method", ""),
            "summary": f.get("summary", ""),
            "error_message": f.get("error_message", ""),
            "pages_count": f.get("pages_count", "0"),
            "language": f.get("language", ""),
            "structured_data": f.get("structured_data"),
        })
    return {"files": result}


@app.get("/api/chat/stats")
async def chat_stats() -> Dict[str, Any]:
    return archive_manager.get_index_stats()


@app.get("/api/chat/logs")
async def chat_logs() -> Dict[str, Any]:
    return {"logs": chatbot_service.get_logs(), "stats": chatbot_service.get_logs_stats()}


@app.get("/api/chat/logs/export")
async def chat_logs_export() -> Response:
    csv_data = chatbot_service.get_logs_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_logs.csv"},
    )


@app.get("/api/analytics")
async def analytics_data() -> Dict[str, Any]:
    return archive_manager.get_analytics_data()


@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department_id: str = Form(...),
    section_id: str = Form(...),
    year: str = Form(""),
) -> Dict[str, Any]:
    """Fast upload: save file immediately, process in background."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="يسمح فقط برفع ملفات PDF")

    year = (year or "").strip()
    if not year or not re.match(r"^\d{4}$", year):
        raise HTTPException(status_code=400, detail="السنة مطلوبة (4 أرقام)")

    content = await file.read()

    try:
        file_id, file_path = archive_manager.save_file_immediately(
            file_content=content,
            department_id=department_id,
            section_id=section_id,
            year=year,
            original_filename=file.filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(
        archive_manager.process_file_background,
        file_id,
        file_path,
        department_id,
        section_id,
        year,
    )

    return {
        "success": True,
        "file_id": file_id,
        "message": "تم رفع الملف وجاري المعالجة في الخلفية...",
    }


@app.post("/api/files/{file_id}/reprocess")
async def reprocess_file(file_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Re-trigger background processing for a failed file."""
    raw = archive_manager._load_files_list()
    target = next((x for x in raw if x.get("id") == file_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="الملف غير موجود")

    rel = target.get("relative_path", "")
    file_path = BASE_DIR / rel if rel else None
    if not file_path or not file_path.is_file():
        raise HTTPException(status_code=404, detail="ملف PDF غير موجود على القرص")

    archive_manager.update_file_metadata(file_id, {"status": "processing", "error_message": ""})

    background_tasks.add_task(
        archive_manager.process_file_background,
        file_id,
        file_path,
        target.get("department_id", ""),
        target.get("section_id", ""),
        target.get("year", ""),
    )

    return {"success": True, "message": "جاري إعادة المعالجة..."}


@app.delete("/api/files/by-id/{file_id}")
async def delete_file_by_id(file_id: str) -> Dict[str, str]:
    try:
        archive_manager.delete_file_by_id(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "تم حذف الملف بنجاح"}


@app.delete("/api/files/{file_id}/cancel")
async def cancel_file_processing(file_id: str) -> Dict[str, str]:
    """Cancel a file that is currently being processed."""
    # Add to cancellation set so background task stops
    CANCELLED_FILES.add(file_id)
    ok = archive_manager.cancel_file(file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    return {"message": "تم إلغاء المعالجة وحذف الملف"}


@app.patch("/api/files/{file_id}/visibility")
async def set_file_visibility(file_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Toggle public/private visibility for a file."""
    is_public = bool(body.get("is_public", False))
    ok = archive_manager.set_file_visibility(file_id, is_public)
    if not ok:
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    return {"success": True, "is_public": is_public}


@app.get("/api/files/by-ids")
async def get_files_by_ids(ids: str = "") -> Dict[str, Any]:
    """Return metadata for specific file IDs (comma-separated). Public files only."""
    if not ids.strip():
        return {"files": []}
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    all_files = archive_manager._load_files_list()
    matched = [
        f for f in all_files
        if f.get("id") in id_list and f.get("status", "ready") == "ready"
    ]
    return {"files": matched}


@app.delete("/api/files/{department_id}/{file_name}")
async def delete_file_legacy(department_id: str, file_name: str) -> Dict[str, str]:
    try:
        archive_manager.delete_file(department_id=department_id, file_name=file_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "تم حذف الملف بنجاح"}


def _resolve_file_path(file_id: str):
    """Shared helper: find metadata + resolve absolute path for a file_id."""
    raw = archive_manager._load_files_list()
    target = next((x for x in raw if x.get("id") == file_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    rel = target.get("relative_path", "")
    file_path = BASE_DIR / rel if rel else None
    if not file_path or not file_path.is_file():
        raise HTTPException(status_code=404, detail="ملف PDF غير موجود على القرص")
    return target, file_path


@app.get("/files/{file_id}/view")
async def view_file(file_id: str) -> FileResponse:
    """Serve the PDF inline in the browser (for viewing)."""
    target, file_path = _resolve_file_path(file_id)
    filename = target.get("original_filename", "file.pdf")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
        filename=filename,
    )


@app.get("/files/{file_id}/download")
async def download_file(file_id: str) -> FileResponse:
    """Force-download the PDF file."""
    target, file_path = _resolve_file_path(file_id)
    filename = target.get("original_filename", "file.pdf")
    encoded_name = quote(filename, safe="")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
        filename=filename,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(payload: ChatRequest, request: Request) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="السؤال مطلوب")

    ip = request.client.host if request.client else "unknown"
    session_id = request.headers.get("X-Session-ID", "")

    try:
        answer, source = chatbot_service.ask(question, ip=ip, session_id=session_id)
    except Exception as exc:
        logger.exception("chatbot_service.ask failed")
        raise HTTPException(status_code=500, detail=f"Chat error: {exc!s}") from exc

    return ChatResponse(
        answer=answer,
        source=source if isinstance(source, dict) else {},
        show_source=bool(source.get("show_source", False)) if isinstance(source, dict) else False,
        contains_violation=bool(isinstance(answer, str) and answer.startswith("عذراً، لا يمكنني الرد")),
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events streaming endpoint for real-time chat responses."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="السؤال مطلوب")

    ip = request.client.host if request.client else "unknown"
    session_id = request.headers.get("X-Session-ID", "")

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_sync_gen():
            try:
                for chunk_data in chatbot_service.stream_ask(question, ip=ip, session_id=session_id):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk_data), loop)
            except Exception as exc:
                logger.exception("stream_ask generator error: %s", exc)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"chunk": "عذراً، حدث خطأ أثناء المعالجة.", "done": True, "source": {}}),
                    loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, run_sync_gen)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Debug / Admin utility endpoints ──────────────────────────────────────────

@app.get("/api/debug/chromadb")
async def debug_chromadb(q: str = "قرار") -> Dict[str, Any]:
    """Inspect ChromaDB contents and test vector search."""
    try:
        col = chatbot_service.collection
        total = col.count()
        norm_q = normalize_numbers(q)
        results = col.query(
            query_texts=[norm_q],
            n_results=min(5, total) if total else 1,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            {
                "preview": (doc or "")[:200],
                "metadata": meta,
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return {"total_documents_in_chromadb": total, "query": q, "normalized_query": norm_q, "results": hits}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/debug/metadata-search")
async def debug_metadata_search(decree: str, year: str = "") -> Dict[str, Any]:
    """Test the metadata fallback search by decree number."""
    from chatbot_logic import extract_decree_from_question as _edq
    matched_docs, matched_metas = chatbot_service._metadata_keyword_search(
        f"قرار رقم {decree}" + (f" لسنة {year}" if year else "")
    )
    return {
        "decree_searched": decree,
        "year_searched": year,
        "files_found": len(matched_docs),
        "previews": [(d or "")[:200] for d in matched_docs],
        "metadatas": matched_metas,
    }


@app.post("/api/admin/reindex-all")
async def reindex_all(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Clear ChromaDB and re-index every ready file with the current chunking logic."""
    background_tasks.add_task(_reindex_all_task)
    return {"message": "إعادة الفهرسة بدأت في الخلفية. تحقق من السجلات للمتابعة."}


def _reindex_all_task() -> None:
    """Background task: clear ChromaDB and re-index all ready files."""
    import logging as _log
    _logger = _log.getLogger("reindex_all")
    try:
        col = chatbot_service.collection
        # Delete every document (ChromaDB requires a where filter)
        try:
            col.delete(where={"file_id": {"$ne": "__none__"}})
            _logger.info("ChromaDB cleared successfully.")
        except Exception as exc:
            _logger.warning("Could not bulk-clear ChromaDB (will overwrite): %s", exc)

        meta_path = BASE_DIR / "data" / "files_metadata.json"
        if not meta_path.exists():
            _logger.warning("files_metadata.json not found — nothing to re-index.")
            return

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        files = data.get("files", data) if isinstance(data, dict) else data
        ready = [f for f in files if f.get("status") == "ready"]
        _logger.info("Re-indexing %d ready files…", len(ready))

        for f in ready:
            file_id = f.get("id") or f.get("file_id", "")
            if not file_id:
                continue
            try:
                # Prefer previously saved clean text; fall back to pdfplumber
                text = load_stored_text(file_id, BASE_DIR)
                if not text:
                    rel = f.get("relative_path", "")
                    file_path = BASE_DIR / rel if rel else None
                    if file_path and file_path.is_file():
                        text = archive_manager._extract_text_pdfplumber(file_path)
                if not text:
                    text = " ".join(filter(None, [
                        f.get("summary", ""), f.get("main_topic", ""),
                        f.get("document_type", ""), f.get("original_filename", ""),
                    ]))
                if not text:
                    _logger.warning("No text for file_id=%s — skipping.", file_id)
                    continue

                chroma_meta = {
                    "file_id": file_id,
                    "original_filename": f.get("original_filename", ""),
                    "file_name": f.get("file_name", ""),
                    "department": f.get("department_name_ar", ""),
                    "department_name_ar": f.get("department_name_ar", ""),
                    "department_name_en": f.get("department_name_en", ""),
                    "section": f.get("section_name_ar", ""),
                    "section_name_ar": f.get("section_name_ar", ""),
                    "section_name_en": f.get("section_name_en", ""),
                    "year": normalize_numbers(str(f.get("year", ""))),
                    "decree_number": normalize_numbers(str(f.get("decree_number", ""))),
                    "doc_type": f.get("doc_type", ""),
                    "document_type": f.get("document_type", ""),
                    "main_topic": f.get("main_topic", ""),
                    "summary": f.get("summary", ""),
                    "language": f.get("language", ""),
                    "confidence": f.get("confidence", ""),
                    "upload_date": f.get("upload_date", ""),
                    "department_id": f.get("department_id", ""),
                    "section_id": f.get("section_id", ""),
                    "relative_path": f.get("relative_path", ""),
                    "ocr_used": str(f.get("ocr_used", "false")).lower(),
                    "extraction_method": f.get("extraction_method", ""),
                }
                archive_manager._index_document(file_id, text, chroma_meta)
                _logger.info("Re-indexed: %s", f.get("original_filename", file_id))
            except Exception as exc:
                _logger.error("Failed to re-index file_id=%s: %s", file_id, exc)

        _logger.info("✅ Re-indexing complete. %d files processed.", len(ready))
    except Exception as exc:
        _logger.exception("_reindex_all_task failed: %s", exc)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
