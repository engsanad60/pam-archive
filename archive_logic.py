import base64
import json
import logging
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import fitz
import pdfplumber
from anthropic import Anthropic
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

# Global set for tracking files that should be cancelled during background processing
CANCELLED_FILES: set = set()

# ── Arabic/Western numeral helpers ────────────────────────────────────────────

_AR2W = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_W2AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

def normalize_numbers(text: str) -> str:
    """Convert Arabic-Indic numerals (٠-٩) to Western (0-9) in a string."""
    if not text:
        return text
    return str(text).translate(_AR2W)

def get_number_variants(number) -> List[str]:
    """Return both Western and Arabic-Indic forms of a number, plus parenthesised variants."""
    western = normalize_numbers(str(number))
    arabic = western.translate(_W2AR)
    return [western, arabic, f"({western})", f"( {western} )", f"({arabic})", f"( {arabic} )"]

# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-based chunks for better retrieval."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks if chunks else [text]

# ── Stored text helpers ───────────────────────────────────────────────────────

def _texts_dir(base_dir: Path) -> Path:
    d = base_dir / "data" / "texts"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_stored_text(file_id: str, text: str, base_dir: Path) -> str:
    """Persist extracted/cleaned text to disk and return the path."""
    path = _texts_dir(base_dir) / f"{file_id}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)

def load_stored_text(file_id: str, base_dir: Optional[Path] = None) -> str:
    """Load previously saved extracted text from disk."""
    search_dirs = [Path("data/texts"), Path("./data/texts")]
    if base_dir:
        search_dirs.insert(0, base_dir / "data" / "texts")
    for d in search_dirs:
        p = d / f"{file_id}.txt"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""

DEPARTMENTS_FILE = "departments.json"
CONFIG_FILE = "system_config.json"
FILES_METADATA_FILE = "files_metadata.json"

MIN_DIRECT_TEXT_CHARS = 100

DEFAULT_DEPARTMENTS = [
    {"id": "hr", "name_ar": "الموارد البشرية", "name_en": "HR", "sections": [], "section_years": {}},
    {"id": "legal", "name_ar": "الشؤون القانونية", "name_en": "Legal", "sections": [], "section_years": {}},
    {"id": "finance", "name_ar": "المالية", "name_en": "Finance", "sections": [], "section_years": {}},
    {"id": "operations", "name_ar": "العمليات", "name_en": "Operations", "sections": [], "section_years": {}},
]

DEFAULT_CONFIG = {
    "system_name_ar": "الهيئة العامة للقوى العاملة",
    "system_name_en": "Public Authority for Manpower",
}

PREDEFINED_YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]

COMBINED_ANALYSIS_PROMPT = """Analyze this document and return ONLY a single JSON object with exactly two keys: "verification" and "structured_data".

"verification" must contain:
{
  "document_type": "type of document in its original language",
  "main_topic": "one line description in document language",
  "decree_number": "number if found, else empty string",
  "year": "year if found, else empty string",
  "pages_read": number,
  "confidence": "Clear or Partially Clear or Unclear",
  "summary": "3-4 lines summary in document language",
  "language": "Arabic or English or Other"
}

"structured_data" must contain:
First identify doc_type as: purchase_order, invoice, contract, circular, or other.

For purchase_order:
{"doc_type":"purchase_order","po_number":null,"date":null,"vendor_name":null,"vendor_code":null,"total_amount":null,"currency":null,"items":[],"cost_center":null,"payment_terms":null,"department":null}

For invoice:
{"doc_type":"invoice","invoice_number":null,"date":null,"customer_name":null,"vendor_name":null,"contract_reference":null,"total_amount":null,"currency":null,"items":[],"bank_details":null}

For contract:
{"doc_type":"contract","contract_title":null,"contract_number":null,"date":null,"start_date":null,"end_date":null,"duration_months":null,"party_one":null,"party_two":null,"total_value":null,"currency":null,"payment_terms":null,"covered_systems":[],"governing_law":null}

For circular:
{"doc_type":"circular","circular_number":null,"year":null,"issuing_authority":null,"subject":null,"effective_date":null,"summary":null}

For other:
{"doc_type":"other","description":null}

Always extract amounts as numbers not strings. Use null for missing fields.
Return ONLY the JSON object, no markdown, no backticks, no explanation."""


@dataclass
class ArchiveRecord:
    id: str
    file_name: str
    department_id: str
    department_name_ar: str
    department_name_en: str
    section_id: str
    section_name_ar: str
    section_name_en: str
    year: str
    decree_number: str
    pages_count: int
    confidence: str
    document_type: str
    main_topic: str
    summary: str
    language: str
    upload_date: str
    relative_path: str
    ocr_used: bool
    extraction_method: str
    status: str = "ready"
    doc_type: str = ""
    structured_data: Optional[Dict[str, Any]] = None
    error_message: str = ""


class ArchiveManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.uploads_dir = self.base_dir / "uploads"
        self.data_dir = self.base_dir / "data"
        self.departments_path = self.base_dir / DEPARTMENTS_FILE
        self.config_path = self.base_dir / CONFIG_FILE
        self.files_metadata_path = self.data_dir / FILES_METADATA_FILE

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_and_ensure_defaults()
        load_dotenv()
        load_dotenv(self.base_dir / ".env")
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

        self.chroma_client = chromadb.PersistentClient(path=str(self.data_dir))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="archive_documents"
        )
        self.embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def _migrate_and_ensure_defaults(self) -> None:
        self._ensure_default_config()
        self._migrate_departments_file()
        self._migrate_files_metadata_file()

    def _ensure_default_config(self) -> None:
        if not self.config_path.exists():
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)

    def _migrate_departments_file(self) -> None:
        if not self.departments_path.exists():
            self._save_departments_wrapper({"departments": DEFAULT_DEPARTMENTS.copy()})
            return

        with self.departments_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "departments" in raw:
            depts = raw["departments"]
        elif isinstance(raw, list):
            depts = raw
        else:
            depts = []

        normalized: List[Dict[str, Any]] = []
        if depts and isinstance(depts[0], str):
            for name in depts:
                slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or uuid.uuid4().hex[:8]
                normalized.append(
                    {
                        "id": slug,
                        "name_ar": name,
                        "name_en": name,
                        "sections": [],
                        "section_years": {},
                    }
                )
        else:
            for item in depts:
                if not isinstance(item, dict):
                    continue
                dep_id = str(item.get("id", "")).strip() or uuid.uuid4().hex[:8]
                name_ar = str(item.get("name_ar", "")).strip() or str(item.get("name_en", "")).strip() or dep_id
                name_en = str(item.get("name_en", "")).strip() or str(item.get("name_ar", "")).strip() or dep_id
                sections = item.get("sections")
                if not isinstance(sections, list):
                    sections = []
                clean_sections = []
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    clean_sections.append(
                        {
                            "id": str(sec.get("id", "")).strip() or uuid.uuid4().hex[:8],
                            "name_ar": str(sec.get("name_ar", "")).strip(),
                            "name_en": str(sec.get("name_en", "")).strip(),
                        }
                    )
                section_years = item.get("section_years")
                if not isinstance(section_years, dict):
                    section_years = {}
                normalized.append(
                    {
                        "id": dep_id,
                        "name_ar": name_ar,
                        "name_en": name_en,
                        "sections": clean_sections,
                        "section_years": section_years,
                    }
                )

        existing = {d["id"] for d in normalized}
        for dep in DEFAULT_DEPARTMENTS:
            if dep["id"] not in existing:
                normalized.append(dict(dep))
        self._save_departments_wrapper({"departments": normalized})

    def _save_departments_wrapper(self, wrapper: Dict[str, Any]) -> None:
        with self.departments_path.open("w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)

    def _load_departments_wrapper(self) -> Dict[str, Any]:
        with self.departments_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "departments" in data:
            return data
        return {"departments": []}

    def get_departments(self) -> List[Dict[str, Any]]:
        return self._load_departments_wrapper().get("departments", [])

    def _save_departments_list(self, departments: List[Dict[str, Any]]) -> None:
        self._save_departments_wrapper({"departments": departments})

    def add_department(self, name_ar: str, name_en: str) -> Dict[str, Any]:
        name_ar, name_en = name_ar.strip(), name_en.strip()
        if not name_ar or not name_en:
            raise ValueError("اسم الإدارة غير صالح")
        departments = self.get_departments()
        new_id = uuid.uuid4().hex[:10]
        item = {
            "id": new_id,
            "name_ar": name_ar,
            "name_en": name_en,
            "sections": [],
            "section_years": {},
        }
        departments.append(item)
        self._save_departments_list(departments)
        return item

    def update_department(self, department_id: str, name_ar: str, name_en: str) -> Dict[str, Any]:
        name_ar, name_en = name_ar.strip(), name_en.strip()
        if not name_ar or not name_en:
            raise ValueError("الاسم العربي والإنجليزي مطلوبان")
        departments = self.get_departments()
        for dep in departments:
            if dep["id"] == department_id:
                dep["name_ar"] = name_ar
                dep["name_en"] = name_en
                self._save_departments_list(departments)
                return dep
        raise ValueError("الإدارة غير موجودة")

    def delete_department(self, department_id: str) -> None:
        departments = self.get_departments()
        target_dep = next((d for d in departments if d["id"] == department_id), None)
        if not target_dep:
            raise ValueError("الإدارة غير موجودة")
        dept_folder = self.uploads_dir / self._sanitize_segment(target_dep["name_ar"])
        for f in list(self._load_files_list()):
            if f.get("department_id") == department_id:
                try:
                    self.delete_file_by_id(f.get("id", ""))
                except Exception:
                    pass
        departments = [d for d in departments if d["id"] != department_id]
        self._save_departments_list(departments)
        if dept_folder.is_dir():
            shutil.rmtree(dept_folder, ignore_errors=True)

    def add_section(self, department_id: str, name_ar: str, name_en: str) -> Dict[str, Any]:
        name_ar, name_en = name_ar.strip(), name_en.strip()
        if not name_ar or not name_en:
            raise ValueError("اسم القسم مطلوب")
        departments = self.get_departments()
        for dep in departments:
            if dep["id"] == department_id:
                sec_id = uuid.uuid4().hex[:10]
                sec = {"id": sec_id, "name_ar": name_ar, "name_en": name_en}
                dep.setdefault("sections", []).append(sec)
                dep.setdefault("section_years", {})
                dep["section_years"].setdefault(sec_id, [])
                self._save_departments_list(departments)
                return sec
        raise ValueError("الإدارة غير موجودة")

    def update_section(self, department_id: str, section_id: str, name_ar: str, name_en: str) -> Dict[str, Any]:
        name_ar, name_en = name_ar.strip(), name_en.strip()
        departments = self.get_departments()
        for dep in departments:
            if dep["id"] != department_id:
                continue
            for sec in dep.get("sections", []):
                if sec["id"] == section_id:
                    sec["name_ar"] = name_ar
                    sec["name_en"] = name_en
                    self._save_departments_list(departments)
                    return sec
        raise ValueError("القسم غير موجود")

    def delete_section(self, department_id: str, section_id: str) -> None:
        departments = self.get_departments()
        for dep in departments:
            if dep["id"] != department_id:
                continue
            dep["sections"] = [s for s in dep.get("sections", []) if s["id"] != section_id]
            sy = dep.get("section_years", {})
            if section_id in sy:
                del sy[section_id]
            self._save_departments_list(departments)
            for f in list(self._load_files_list()):
                if f.get("department_id") == department_id and f.get("section_id") == section_id:
                    try:
                        self.delete_file_by_id(f["id"])
                    except Exception:
                        pass
            return
        raise ValueError("الإدارة غير موجودة")

    def add_year_to_section(self, department_id: str, section_id: str, year: str) -> None:
        year = str(year).strip()
        if not year or not re.match(r"^\d{4}$", year):
            raise ValueError("سنة غير صالحة")
        departments = self.get_departments()
        for dep in departments:
            if dep["id"] != department_id:
                continue
            if not any(s["id"] == section_id for s in dep.get("sections", [])):
                raise ValueError("القسم غير موجود")
            sy = dep.setdefault("section_years", {})
            lst = sy.setdefault(section_id, [])
            if year not in lst:
                lst.append(year)
                lst.sort()
            self._save_departments_list(departments)
            (self._section_year_path(dep, section_id, year)).mkdir(parents=True, exist_ok=True)
            return
        raise ValueError("الإدارة غير موجودة")

    def _section_by_ids(self, department_id: str, section_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        for dep in self.get_departments():
            if dep["id"] != department_id:
                continue
            for sec in dep.get("sections", []):
                if sec["id"] == section_id:
                    return dep, sec
        raise ValueError("الإدارة أو القسم غير موجود")

    def _sanitize_segment(self, name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]+', "", name).strip() or "folder"

    def _section_year_path(self, dep: Dict[str, Any], section_id: str, year: str) -> Path:
        sec = next((s for s in dep.get("sections", []) if s["id"] == section_id), None)
        if not sec:
            raise ValueError("القسم غير موجود")
        d_ar = self._sanitize_segment(dep["name_ar"])
        s_ar = self._sanitize_segment(sec["name_ar"])
        y = self._sanitize_segment(year)
        return self.uploads_dir / d_ar / s_ar / y

    def _ensure_year_registered(self, department_id: str, section_id: str, year: str) -> None:
        try:
            self.add_year_to_section(department_id, section_id, year)
        except ValueError:
            pass

    def _migrate_files_metadata_file(self) -> None:
        if not self.files_metadata_path.exists():
            self._save_files_wrapper({"files": []})
            return
        with self.files_metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        files: List[Dict[str, Any]] = []
        if isinstance(data, dict) and "files" in data:
            files = data["files"] if isinstance(data["files"], list) else []
        elif isinstance(data, list):
            files = data

        migrated: List[Dict[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                item["id"] = uuid.uuid4().hex
            item.setdefault("original_filename", item.get("file_name", ""))
            item.setdefault("department", item.get("department_name_ar", ""))
            item.setdefault("section", item.get("section_name_ar", "عام"))
            item.setdefault("section_id", item.get("section_id", "legacy"))
            item.setdefault("year", item.get("year", "") or "")
            item.setdefault("ocr_used", str(item.get("ocr_used", "false")).lower() == "true")
            item.setdefault("extraction_method", item.get("extraction_method", "legacy"))
            item.setdefault("file_path", "/" + item.get("relative_path", "").replace("\\", "/"))
            item.setdefault("status", "ready")
            item.setdefault("doc_type", "")
            item.setdefault("structured_data", None)
            item.setdefault("error_message", "")
            migrated.append(item)
        self._save_files_wrapper({"files": migrated})

    def _save_files_wrapper(self, wrapper: Dict[str, Any]) -> None:
        with self.files_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)

    def _load_files_list(self) -> List[Dict[str, Any]]:
        if not self.files_metadata_path.exists():
            self._save_files_wrapper({"files": []})
        with self.files_metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "files" in data:
            return data["files"] if isinstance(data["files"], list) else []
        if isinstance(data, list):
            return data
        return []

    def _normalize_arabic_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
        normalized = normalized.replace("ى", "ي").replace("ة", "ه")
        return normalized

    def _extract_decree_number(self, text: str) -> str:
        for pattern in [
            r"(?:قرار|القرار)\s*(?:رقم)?\s*[:\-]?\s*(\d+)",
            r"(?:No\.?|Number)\s*[:\-]?\s*(\d+)",
        ]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return ""

    def _extract_year(self, text: str) -> str:
        m = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
        return m.group(1) if m else ""

    def _extract_text_pdfplumber(self, pdf_path: Path) -> str:
        parts: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    def _extract_pages_count(self, pdf_path: Path) -> int:
        doc = fitz.open(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()

    def _extract_text_claude_vision(self, pdf_path: Path) -> str:
        doc = fitz.open(str(pdf_path))
        all_text: List[str] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode()
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "اقرأ كل النص الموجود في هذه الصورة بدقة تامة. اكتب النص كما هو بدون أي تعديل أو تلخيص.",
                                },
                            ],
                        }
                    ],
                )
                page_text = response.content[0].text if response.content else ""
                all_text.append(f"\n--- صفحة {page_num + 1} ---\n{page_text}")
        finally:
            doc.close()
        return "\n".join(all_text).strip()

    def _clean_ocr_text(self, raw_text: str) -> str:
        """Send raw OCR text to Claude for correction of Arabic OCR errors."""
        if not raw_text or len(raw_text.strip()) < 20:
            return raw_text
        try:
            resp = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "هذا نص مستخرج من مستند عربي ممسوح ضوئياً وفيه أخطاء OCR.\n"
                            "قم بتصحيح الأخطاء الإملائية والكلمات المشوهة وأعد كتابة النص "
                            "بشكل صحيح مع الحفاظ على المعنى الأصلي والأرقام والتواريخ.\n"
                            "أعد النص المصحح فقط بدون أي تعليق.\n\n"
                            f"النص:\n{raw_text}"
                        ),
                    }
                ],
            )
            cleaned = resp.content[0].text if resp.content else raw_text
            return cleaned.strip() if cleaned.strip() else raw_text
        except Exception as exc:
            logger.warning("OCR text cleaning failed (non-fatal): %s", exc)
            return raw_text

    def extract_full_text_pipeline(self, pdf_path: Path) -> Dict[str, Any]:
        text = self._extract_text_pdfplumber(pdf_path)
        if len(text.strip()) > MIN_DIRECT_TEXT_CHARS:
            return {"text": text, "extraction_method": "direct", "ocr_used": False}
        vision_text = self._extract_text_claude_vision(pdf_path)
        # Clean OCR errors from scanned PDF text
        cleaned_text = self._clean_ocr_text(vision_text)
        return {
            "text": cleaned_text.strip(),
            "extraction_method": "claude_vision",
            "ocr_used": True,
        }

    def _extraction_method_label(self, method: str) -> str:
        if method == "direct":
            return "طريقة القراءة: نص مباشر ✅"
        return "طريقة القراءة: Claude Vision 🤖 (ممسوح ضوئياً)"

    def _parse_claude_json(self, raw: str) -> Dict[str, Any]:
        """Robustly parse JSON from Claude output, stripping markdown fences."""
        content = raw.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _call_combined_analysis(self, text: str, pages_count: int) -> Dict[str, Any]:
        """Single Claude call returning both verification info and structured financial data."""
        text_for_model = text[:15000] if text else "No readable text extracted."
        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                temperature=0,
                system=COMBINED_ANALYSIS_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Pages: {pages_count}\nDocument text:\n{text_for_model}",
                    }
                ],
            )
            raw = message.content[0].text.strip() if message.content else "{}"
            return self._parse_claude_json(raw)
        except Exception as exc:
            logger.warning("_call_combined_analysis failed: %s", exc)
            return {}

    def verify_pdf_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Legacy method kept for API compatibility. Used by old verify flow."""
        pages_count = self._extract_pages_count(pdf_path)
        pipe = self.extract_full_text_pipeline(pdf_path)
        text = pipe["text"]
        extraction_method = pipe["extraction_method"]
        ocr_used = pipe["ocr_used"]

        combined = self._call_combined_analysis(text, pages_count)
        parsed = combined.get("verification", {})

        decree = str(parsed.get("decree_number", "")).strip()
        year = str(parsed.get("year", "")).strip()
        if not decree:
            decree = self._extract_decree_number(self._normalize_arabic_text(text))
        if not year:
            year = self._extract_year(text)

        return {
            "document_type": str(parsed.get("document_type", "")).strip(),
            "main_topic": str(parsed.get("main_topic", "")).strip(),
            "decree_number": decree,
            "year": year,
            "pages_read": int(parsed.get("pages_read") or pages_count),
            "confidence": str(parsed.get("confidence", "Unclear")).strip() or "Unclear",
            "summary": str(parsed.get("summary", "")).strip(),
            "language": str(parsed.get("language", "Other")).strip() or "Other",
            "raw_text": text,
            "extraction_method": extraction_method,
            "extraction_method_label": self._extraction_method_label(extraction_method),
            "ocr_used": ocr_used,
        }

    def _index_document(self, file_id: str, text: str, metadata: Dict[str, str]) -> None:
        """Delete old chunks, save text to disk, then store overlapping chunks in ChromaDB."""
        # 1. Remove previous chunks for this file
        try:
            self.chroma_collection.delete(where={"file_id": file_id})
        except Exception:
            pass

        # 2. Normalize numbers in the text and key metadata fields
        norm_text = normalize_numbers(text or "")
        norm_meta = dict(metadata)
        for key in ("decree_number", "year"):
            if norm_meta.get(key):
                norm_meta[key] = normalize_numbers(str(norm_meta[key]))

        # 3. Save cleaned text to disk for later full-text fallback
        try:
            save_stored_text(file_id, norm_text, self.base_dir)
            norm_meta["text_path"] = str(self.base_dir / "data" / "texts" / f"{file_id}.txt")
        except Exception as exc:
            logger.warning("Could not save text to disk for %s: %s", file_id, exc)

        # 4. Split into overlapping chunks and store directly in ChromaDB
        chunks = chunk_text(norm_text)
        total = len(chunks)
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_meta = dict(norm_meta)
            chunk_meta["chunk_index"] = str(i)
            chunk_meta["total_chunks"] = str(total)
            ids.append(f"{file_id}_chunk_{i}")
            docs.append(chunk)
            metas.append(chunk_meta)

        if ids:
            self.chroma_collection.add(documents=docs, metadatas=metas, ids=ids)
        logger.info("Indexed %d chunk(s) for file_id=%s", total, file_id)

    # ── Background Processing (new fast upload flow) ──────────────────────────

    def save_file_immediately(
        self,
        file_content: bytes,
        department_id: str,
        section_id: str,
        year: str,
        original_filename: str,
    ) -> Tuple[str, Path]:
        """Save file to disk instantly and create a 'processing' metadata entry."""
        dep, sec = self._section_by_ids(department_id, section_id)
        year = str(year).strip()
        if not year or not re.match(r"^\d{4}$", year):
            raise ValueError("السنة مطلوبة وبصيغة 4 أرقام")

        self._ensure_year_registered(department_id, section_id, year)

        saved_name = (original_filename or "file.pdf").strip() or "file.pdf"
        dest_dir = self._section_year_path(dep, section_id, year)
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / saved_name

        with open(destination, "wb") as fh:
            fh.write(file_content)

        file_uuid = uuid.uuid4().hex
        relative_path = destination.relative_to(self.base_dir).as_posix()
        file_path = "/" + relative_path
        upload_date = datetime.utcnow().date().isoformat()

        row: Dict[str, Any] = {
            "id": file_uuid,
            "file_id": file_uuid,
            "original_filename": saved_name,
            "file_name": saved_name,
            "department": dep["name_ar"],
            "department_id": department_id,
            "department_name_ar": dep["name_ar"],
            "department_name_en": dep["name_en"],
            "section": sec["name_ar"],
            "section_id": section_id,
            "section_name_ar": sec["name_ar"],
            "section_name_en": sec["name_en"],
            "year": year,
            "upload_date": upload_date,
            "status": "processing",
            "file_path": file_path,
            "relative_path": relative_path,
            "decree_number": "",
            "pages_count": "0",
            "confidence": "",
            "document_type": "",
            "main_topic": "",
            "summary": "",
            "language": "",
            "ocr_used": False,
            "extraction_method": "",
            "doc_type": "",
            "structured_data": None,
            "error_message": "",
            "is_public": False,
            "visibility_note": "",
        }
        self._append_file_metadata(row)
        return file_uuid, destination

    def update_file_metadata(self, file_id: str, updates: Dict[str, Any]) -> None:
        """Update specific fields of a file's metadata entry."""
        data = self._load_files_list()
        for item in data:
            if item.get("id") == file_id or item.get("file_id") == file_id:
                item.update(updates)
                break
        self._save_files_wrapper({"files": data})

    def cancel_file(self, file_id: str) -> bool:
        """Cancel a processing file: mark it, add to cancelled set, delete physical file."""
        CANCELLED_FILES.add(file_id)
        data = self._load_files_list()
        target = next((x for x in data if x.get("id") == file_id), None)
        if not target:
            return False
        rel = target.get("relative_path", "")
        if rel:
            file_path = self.base_dir / rel
            try:
                if file_path.is_file():
                    file_path.unlink()
            except Exception as exc:
                logger.warning("Failed to delete cancelled file %s: %s", file_path, exc)
        # Remove from ChromaDB if partially indexed
        try:
            self.chroma_collection.delete(where={"file_id": file_id})
        except Exception:
            pass
        self.update_file_metadata(file_id, {"status": "cancelled", "error_message": "تم الإلغاء من قبل المستخدم"})
        return True

    def set_file_visibility(self, file_id: str, is_public: bool) -> bool:
        """Toggle a file's public/private visibility."""
        data = self._load_files_list()
        target = next((x for x in data if x.get("id") == file_id), None)
        if not target:
            return False
        self.update_file_metadata(file_id, {
            "is_public": is_public,
            "visibility_note": "للجمهور" if is_public else "للشات بوت فقط",
        })
        return True

    def process_file_background(
        self,
        file_id: str,
        file_path: Path,
        department_id: str,
        section_id: str,
        year: str,
    ) -> None:
        """Full document processing pipeline - runs in background after instant upload."""
        try:
            # Cancellation check before starting
            if file_id in CANCELLED_FILES:
                CANCELLED_FILES.discard(file_id)
                logger.info("File %s was cancelled before processing started", file_id)
                return

            logger.info("Background processing started for file_id=%s", file_id)

            # Step 1: Extract text
            pipe = self.extract_full_text_pipeline(file_path)
            text = pipe["text"]
            extraction_method = pipe["extraction_method"]
            ocr_used = pipe["ocr_used"]

            # Cancellation check after text extraction
            if file_id in CANCELLED_FILES:
                CANCELLED_FILES.discard(file_id)
                logger.info("File %s cancelled after text extraction", file_id)
                return

            pages_count = self._extract_pages_count(file_path)
            normalized = self._normalize_arabic_text(text)

            # Step 2: Combined Claude analysis (verification + structured data)
            combined = self._call_combined_analysis(text, pages_count)
            ver = combined.get("verification", {})
            structured_data = combined.get("structured_data", {})

            # Cancellation check after Claude analysis
            if file_id in CANCELLED_FILES:
                CANCELLED_FILES.discard(file_id)
                logger.info("File %s cancelled after Claude analysis", file_id)
                return

            # Fallback extraction
            decree_number = str(ver.get("decree_number", "")).strip() or self._extract_decree_number(normalized)
            doc_year = str(ver.get("year", "")).strip() or self._extract_year(normalized)
            document_type = str(ver.get("document_type", "")).strip()
            main_topic = str(ver.get("main_topic", "")).strip()
            summary = str(ver.get("summary", "")).strip()
            language = str(ver.get("language", "Other")).strip() or "Other"
            confidence = str(ver.get("confidence", "Unclear")).strip() or "Unclear"
            doc_type = str(structured_data.get("doc_type", "other")).strip() if structured_data else "other"

            # Step 3: Load current metadata to get department/section labels
            current_data = self._load_files_list()
            target = next((x for x in current_data if x.get("id") == file_id), None)
            if not target:
                logger.error("file_id=%s not found in metadata during background processing", file_id)
                return

            saved_name = target.get("original_filename", "")
            upload_date = target.get("upload_date", datetime.utcnow().date().isoformat())
            relative_path = target.get("relative_path", "")

            # Step 4: Index in ChromaDB (normalize numbers in key fields)
            norm_decree = normalize_numbers(str(decree_number or ""))
            norm_year = normalize_numbers(str(year or ""))
            chroma_metadata = {
                "file_id": file_id,
                "original_filename": saved_name,
                "file_name": saved_name,
                "department_id": department_id,
                "department": target.get("department_name_ar", ""),
                "department_name_ar": target.get("department_name_ar", ""),
                "department_name_en": target.get("department_name_en", ""),
                "section_id": section_id,
                "section": target.get("section_name_ar", ""),
                "section_name_ar": target.get("section_name_ar", ""),
                "section_name_en": target.get("section_name_en", ""),
                "year": norm_year,
                "decree_number": norm_decree,
                "pages_count": str(pages_count),
                "confidence": confidence,
                "document_type": document_type,
                "main_topic": main_topic,
                "summary": summary,
                "language": language,
                "upload_date": upload_date,
                "relative_path": relative_path,
                "ocr_used": str(ocr_used).lower(),
                "extraction_method": extraction_method,
                "doc_type": doc_type,
            }
            index_text = text if text else f"محتوى غير متاح في الملف {saved_name}"
            self._index_document(file_id, index_text, chroma_metadata)

            # Step 5: Update metadata with full data + ready status
            self.update_file_metadata(file_id, {
                "status": "ready",
                "decree_number": decree_number,
                "pages_count": str(pages_count),
                "confidence": confidence,
                "document_type": document_type,
                "main_topic": main_topic,
                "summary": summary,
                "language": language,
                "ocr_used": ocr_used,
                "extraction_method": extraction_method,
                "doc_type": doc_type,
                "structured_data": structured_data if structured_data else None,
                "error_message": "",
            })
            logger.info("Background processing completed for file_id=%s doc_type=%s", file_id, doc_type)

        except Exception as exc:
            logger.exception("Background processing failed for file_id=%s", file_id)
            self.update_file_metadata(file_id, {
                "status": "error",
                "error_message": str(exc),
            })

    def reprocess_file(self, file_id: str) -> bool:
        """Reset status to processing and re-trigger background processing setup."""
        data = self._load_files_list()
        target = next((x for x in data if x.get("id") == file_id), None)
        if not target:
            return False
        rel = target.get("relative_path", "")
        file_path = self.base_dir / rel if rel else None
        if not file_path or not file_path.is_file():
            return False
        self.update_file_metadata(file_id, {"status": "processing", "error_message": ""})
        return True

    def reindex_missing_documents(self) -> int:
        """
        Startup recovery: re-index any 'ready' documents missing from ChromaDB.
        Uses fast pdfplumber extraction for direct-text PDFs.
        For scanned PDFs (claude_vision), falls back to stored metadata text
        to avoid expensive re-Vision calls on every server restart.
        Returns the number of documents re-indexed.
        """
        files = self._load_files_list()
        ready_files = [f for f in files if f.get("status", "ready") == "ready"]
        if not ready_files:
            return 0

        # Get all file_ids currently in ChromaDB
        try:
            chroma_ids_result = self.chroma_collection.get(include=["metadatas"])
            indexed_file_ids: set = set()
            for meta in (chroma_ids_result.get("metadatas") or []):
                fid = (meta or {}).get("file_id", "")
                if fid:
                    indexed_file_ids.add(fid)
        except Exception as exc:
            logger.warning("reindex_missing_documents: could not query ChromaDB: %s", exc)
            indexed_file_ids = set()

        reindexed = 0
        for f in ready_files:
            file_id = f.get("id") or f.get("file_id", "")
            if not file_id or file_id in indexed_file_ids:
                continue  # already in ChromaDB

            rel = f.get("relative_path", "")
            file_path = self.base_dir / rel if rel else None
            if not file_path or not file_path.is_file():
                logger.warning("reindex: PDF not found on disk for file_id=%s (%s)", file_id, rel)
                continue

            try:
                # Prefer previously saved cleaned text (avoids re-OCR)
                text = load_stored_text(file_id, self.base_dir)
                if not text.strip():
                    # Fast extraction path — never calls Claude Vision on startup
                    text = self._extract_text_pdfplumber(file_path)
                if not text.strip():
                    # Fallback: reconstruct searchable text from stored metadata
                    text = " ".join(filter(None, [
                        f.get("document_type", ""),
                        f.get("main_topic", ""),
                        f.get("summary", ""),
                        normalize_numbers(str(f.get("decree_number", ""))),
                        f.get("original_filename", ""),
                    ]))
                if not text.strip():
                    text = f"ملف {f.get('original_filename', file_id)}"

                chroma_metadata = {
                    "file_id": file_id,
                    "original_filename": f.get("original_filename", ""),
                    "file_name": f.get("file_name", ""),
                    "department_id": f.get("department_id", ""),
                    "department": f.get("department_name_ar", ""),
                    "department_name_ar": f.get("department_name_ar", ""),
                    "department_name_en": f.get("department_name_en", ""),
                    "section_id": f.get("section_id", ""),
                    "section": f.get("section_name_ar", ""),
                    "section_name_ar": f.get("section_name_ar", ""),
                    "section_name_en": f.get("section_name_en", ""),
                    "year": normalize_numbers(str(f.get("year", ""))),
                    "decree_number": normalize_numbers(str(f.get("decree_number", ""))),
                    "pages_count": str(f.get("pages_count", "")),
                    "confidence": f.get("confidence", ""),
                    "document_type": f.get("document_type", ""),
                    "main_topic": f.get("main_topic", ""),
                    "summary": f.get("summary", ""),
                    "language": f.get("language", ""),
                    "upload_date": f.get("upload_date", ""),
                    "relative_path": f.get("relative_path", ""),
                    "ocr_used": str(f.get("ocr_used", "false")).lower(),
                    "extraction_method": f.get("extraction_method", ""),
                    "doc_type": f.get("doc_type", ""),
                }
                self._index_document(file_id, text, chroma_metadata)
                reindexed += 1
                logger.info("Re-indexed on startup: %s", f.get("original_filename", file_id))
            except Exception as exc:
                logger.warning("reindex: failed for file_id=%s: %s", file_id, exc)

        return reindexed

    # ── Legacy save_pdf (kept for backward compatibility) ─────────────────────

    def save_pdf(
        self,
        source_file: Path,
        department_id: str,
        section_id: str,
        year: str,
        verification: Optional[Dict[str, Any]] = None,
        original_filename: Optional[str] = None,
    ) -> "ArchiveRecord":
        dep, sec = self._section_by_ids(department_id, section_id)
        year = str(year).strip()
        if not year or not re.match(r"^\d{4}$", year):
            raise ValueError("السنة مطلوبة وبصيغة 4 أرقام")

        self._ensure_year_registered(department_id, section_id, year)

        saved_name = (original_filename or source_file.name).strip() or source_file.name
        dest_dir = self._section_year_path(dep, section_id, year)
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / saved_name
        shutil.copy2(source_file, destination)

        ver = verification or {}
        raw = ver.get("raw_text")
        if raw is not None and str(raw).strip():
            text = str(raw).strip()
        else:
            text = self._extract_text_pdfplumber(destination)

        if len(text) <= MIN_DIRECT_TEXT_CHARS:
            pipe = self.extract_full_text_pipeline(destination)
            text = pipe["text"]
            extraction_method = pipe["extraction_method"]
            ocr_used = pipe["ocr_used"]
        else:
            extraction_method = str(ver.get("extraction_method", "direct"))
            ocr_used = bool(ver.get("ocr_used", False))

        pages_count = int(ver.get("pages_read") or self._extract_pages_count(destination))
        normalized = self._normalize_arabic_text(text)
        decree_number = str(ver.get("decree_number", "")).strip() or self._extract_decree_number(normalized)
        document_type = str(ver.get("document_type", "")).strip()
        main_topic = str(ver.get("main_topic", "")).strip()
        summary = str(ver.get("summary", "")).strip()
        language = str(ver.get("language", "Other")).strip() or "Other"
        confidence = str(ver.get("confidence", "Unclear")).strip() or "Unclear"
        upload_date = datetime.utcnow().date().isoformat()

        file_uuid = uuid.uuid4().hex
        relative_path = destination.relative_to(self.base_dir).as_posix()
        file_path = "/" + relative_path.replace("\\", "/")

        metadata = {
            "file_id": file_uuid,
            "original_filename": saved_name,
            "file_name": saved_name,
            "department_id": department_id,
            "department": dep["name_ar"],
            "department_name_ar": dep["name_ar"],
            "department_name_en": dep["name_en"],
            "section_id": section_id,
            "section": sec["name_ar"],
            "section_name_ar": sec["name_ar"],
            "section_name_en": sec["name_en"],
            "year": year,
            "decree_number": decree_number,
            "pages_count": str(pages_count),
            "confidence": confidence,
            "document_type": document_type,
            "main_topic": main_topic,
            "summary": summary,
            "language": language,
            "upload_date": upload_date,
            "relative_path": relative_path,
            "ocr_used": str(ocr_used).lower(),
            "extraction_method": extraction_method,
        }
        index_text = text if text else f"محتوى غير متاح في الملف {saved_name}"
        self._index_document(file_uuid, index_text, metadata)

        row = {
            "id": file_uuid,
            "file_id": file_uuid,
            "original_filename": saved_name,
            "department": dep["name_ar"],
            "department_id": department_id,
            "department_name_ar": dep["name_ar"],
            "department_name_en": dep["name_en"],
            "section": sec["name_ar"],
            "section_id": section_id,
            "section_name_ar": sec["name_ar"],
            "section_name_en": sec["name_en"],
            "year": year,
            "decree_number": decree_number,
            "upload_date": upload_date,
            "confidence": confidence,
            "ocr_used": ocr_used,
            "extraction_method": extraction_method,
            "file_path": file_path,
            "relative_path": relative_path,
            "file_name": saved_name,
            "pages_count": str(pages_count),
            "document_type": document_type,
            "main_topic": main_topic,
            "summary": summary,
            "language": language,
            "status": "ready",
            "doc_type": "",
            "structured_data": None,
            "error_message": "",
        }
        self._append_file_metadata(row)

        return ArchiveRecord(
            id=file_uuid,
            file_name=saved_name,
            department_id=department_id,
            department_name_ar=dep["name_ar"],
            department_name_en=dep["name_en"],
            section_id=section_id,
            section_name_ar=sec["name_ar"],
            section_name_en=sec["name_en"],
            year=year,
            decree_number=decree_number,
            pages_count=pages_count,
            confidence=confidence,
            document_type=document_type,
            main_topic=main_topic,
            summary=summary,
            language=language,
            upload_date=upload_date,
            relative_path=relative_path,
            ocr_used=ocr_used,
            extraction_method=extraction_method,
            status="ready",
        )

    def _append_file_metadata(self, row: Dict[str, Any]) -> None:
        data = self._load_files_list()
        data.append(row)
        self._save_files_wrapper({"files": data})

    def _record_to_archive_record(self, item: Dict[str, Any]) -> ArchiveRecord:
        return ArchiveRecord(
            id=item.get("id", ""),
            file_name=item.get("file_name", item.get("original_filename", "")),
            department_id=item.get("department_id", ""),
            department_name_ar=item.get("department_name_ar", item.get("department", "")),
            department_name_en=item.get("department_name_en", ""),
            section_id=item.get("section_id", ""),
            section_name_ar=item.get("section_name_ar", item.get("section", "")),
            section_name_en=item.get("section_name_en", ""),
            year=item.get("year", ""),
            decree_number=item.get("decree_number", ""),
            pages_count=int(item.get("pages_count", 0) or 0),
            confidence=item.get("confidence", ""),
            document_type=item.get("document_type", ""),
            main_topic=item.get("main_topic", ""),
            summary=item.get("summary", ""),
            language=item.get("language", ""),
            upload_date=item.get("upload_date", ""),
            relative_path=item.get("relative_path", ""),
            ocr_used=bool(item.get("ocr_used", False)),
            extraction_method=item.get("extraction_method", ""),
            status=item.get("status", "ready"),
            doc_type=item.get("doc_type", ""),
            structured_data=item.get("structured_data"),
            error_message=item.get("error_message", ""),
        )

    def list_all_files(self) -> List[ArchiveRecord]:
        return [self._record_to_archive_record(x) for x in self._load_files_list()]

    def list_files_by_department(self, department_id: Optional[str] = None) -> Dict[str, List[ArchiveRecord]]:
        out: Dict[str, List[ArchiveRecord]] = {d["id"]: [] for d in self.get_departments()}
        for rec in self.list_all_files():
            if department_id and rec.department_id != department_id:
                continue
            out.setdefault(rec.department_id, []).append(rec)
        for k in out:
            out[k] = sorted(out[k], key=lambda r: r.file_name.lower())
        return out

    def get_index_stats(self) -> Dict[str, Any]:
        files = self._load_files_list()
        dept_ids = {f.get("department_id") for f in files if f.get("department_id")}
        ready = sum(1 for f in files if f.get("status", "ready") == "ready")
        processing = sum(1 for f in files if f.get("status") == "processing")
        error = sum(1 for f in files if f.get("status") == "error")
        low_quality = sum(
            1 for f in files
            if f.get("status", "ready") == "ready"
            and "unclear" in str(f.get("confidence", "")).lower()
        )
        return {
            "documents_count": len(files),
            "departments_count": len(dept_ids),
            "ready": ready,
            "processing": processing,
            "error": error,
            "low_quality": low_quality,
        }

    def get_analytics_data(self) -> Dict[str, Any]:
        """Aggregate structured financial data for the analytics dashboard."""
        files = self._load_files_list()
        ready_files = [f for f in files if f.get("status", "ready") == "ready"]

        type_counts: Dict[str, int] = {}
        type_amounts: Dict[str, float] = {}
        by_vendor: Dict[str, float] = {}
        monthly: Dict[str, Dict[str, float]] = {}

        purchase_orders = []
        invoices = []
        contracts = []
        circulars = []

        for f in ready_files:
            dt = f.get("doc_type") or "other"
            sd: Dict[str, Any] = f.get("structured_data") or {}
            type_counts[dt] = type_counts.get(dt, 0) + 1

            amount_raw = sd.get("total_amount") or sd.get("total_value") or 0
            try:
                amount = float(amount_raw) if amount_raw else 0.0
            except (ValueError, TypeError):
                amount = 0.0
            type_amounts[dt] = type_amounts.get(dt, 0.0) + amount

            # Vendor aggregation
            vendor = (
                sd.get("vendor_name")
                or sd.get("party_two")
                or sd.get("customer_name")
                or ""
            )
            if vendor and amount:
                by_vendor[vendor] = by_vendor.get(vendor, 0.0) + amount

            # Monthly aggregation
            date_str = sd.get("date") or sd.get("start_date") or f.get("upload_date") or ""
            yr = f.get("year") or ""
            month = ""
            if date_str:
                parts = re.findall(r"\d+", str(date_str))
                if len(parts) >= 2:
                    month = parts[1].zfill(2) if len(parts[0]) != 4 else parts[1].zfill(2)
                    yr = parts[0] if len(parts[0]) == 4 else yr
            if yr and month and amount:
                monthly.setdefault(yr, {})
                monthly[yr][month] = monthly[yr].get(month, 0.0) + amount

            # Build typed lists
            if dt == "purchase_order":
                purchase_orders.append({
                    "file_id": f.get("id"),
                    "filename": f.get("original_filename"),
                    "po_number": sd.get("po_number"),
                    "date": sd.get("date"),
                    "vendor_name": sd.get("vendor_name"),
                    "description": (sd.get("items") or [{}])[0].get("description", "") if sd.get("items") else "",
                    "total_amount": sd.get("total_amount"),
                    "currency": sd.get("currency", "KWD"),
                    "cost_center": sd.get("cost_center"),
                    "department": f.get("department_name_ar"),
                    "year": f.get("year"),
                })
            elif dt == "invoice":
                invoices.append({
                    "file_id": f.get("id"),
                    "filename": f.get("original_filename"),
                    "invoice_number": sd.get("invoice_number"),
                    "date": sd.get("date"),
                    "customer_name": sd.get("customer_name"),
                    "vendor_name": sd.get("vendor_name"),
                    "total_amount": sd.get("total_amount"),
                    "currency": sd.get("currency", "KWD"),
                    "contract_reference": sd.get("contract_reference"),
                    "department": f.get("department_name_ar"),
                    "year": f.get("year"),
                })
            elif dt == "contract":
                invoices_ref = sd.get("total_value") or sd.get("total_amount")
                contracts.append({
                    "file_id": f.get("id"),
                    "filename": f.get("original_filename"),
                    "contract_title": sd.get("contract_title"),
                    "party_one": sd.get("party_one"),
                    "party_two": sd.get("party_two"),
                    "start_date": sd.get("start_date"),
                    "end_date": sd.get("end_date"),
                    "duration_months": sd.get("duration_months"),
                    "total_value": sd.get("total_value"),
                    "currency": sd.get("currency", "KWD"),
                    "payment_terms": sd.get("payment_terms"),
                    "covered_systems": sd.get("covered_systems", []),
                    "department": f.get("department_name_ar"),
                    "year": f.get("year"),
                })
            elif dt == "circular":
                circulars.append({
                    "file_id": f.get("id"),
                    "filename": f.get("original_filename"),
                    "circular_number": sd.get("circular_number"),
                    "year": sd.get("year") or f.get("year"),
                    "issuing_authority": sd.get("issuing_authority"),
                    "subject": sd.get("subject"),
                    "effective_date": sd.get("effective_date"),
                    "department": f.get("department_name_ar"),
                })

        total_amount = sum(type_amounts.values())

        return {
            "summary": {
                "contracts": type_counts.get("contract", 0),
                "purchase_orders": type_counts.get("purchase_order", 0),
                "invoices": type_counts.get("invoice", 0),
                "circulars": type_counts.get("circular", 0),
                "other": type_counts.get("other", 0),
                "total_amount": round(total_amount, 3),
                "currency": "KWD",
            },
            "purchase_orders": purchase_orders,
            "invoices": invoices,
            "contracts": contracts,
            "circulars": circulars,
            "type_amounts": type_amounts,
            "by_vendor": by_vendor,
            "monthly": monthly,
        }

    def delete_file_by_id(self, file_id: str) -> None:
        data = self._load_files_list()
        target = next((x for x in data if x.get("id") == file_id or x.get("file_id") == file_id), None)
        if not target:
            raise FileNotFoundError("الملف غير موجود")
        rel = target.get("relative_path", "")
        file_path = self.base_dir / rel if rel else None
        if file_path and file_path.is_file():
            file_path.unlink()
        chroma_id = target.get("file_id") or target.get("id")
        self.chroma_collection.delete(where={"file_id": chroma_id})
        data = [x for x in data if x.get("id") != target.get("id")]
        self._save_files_wrapper({"files": data})

    def delete_file(self, department_id: str, file_name: str) -> None:
        for item in self._load_files_list():
            if item.get("department_id") == department_id and item.get("file_name") == file_name:
                self.delete_file_by_id(item["id"])
                return
        raise FileNotFoundError("الملف غير موجود")

    def get_config(self) -> Dict[str, str]:
        if not self.config_path.exists():
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        with self.config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "system_name_ar": str(data.get("system_name_ar", DEFAULT_CONFIG["system_name_ar"])),
            "system_name_en": str(data.get("system_name_en", DEFAULT_CONFIG["system_name_en"])),
        }

    def update_config(self, system_name_ar: str, system_name_en: str) -> Dict[str, str]:
        system_name_ar = system_name_ar.strip()
        system_name_en = system_name_en.strip()
        if not system_name_ar or not system_name_en:
            raise ValueError("اسم النظام بالعربية والإنجليزية مطلوب")
        payload = {"system_name_ar": system_name_ar, "system_name_en": system_name_en}
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def get_predefined_years(self) -> List[str]:
        years_path = self.data_dir / "predefined_years.json"
        if years_path.exists():
            try:
                with years_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return sorted(data)
            except Exception:
                pass
        return list(PREDEFINED_YEARS)

    def _save_predefined_years(self, years: List[str]) -> None:
        years_path = self.data_dir / "predefined_years.json"
        with years_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(years), f, ensure_ascii=False, indent=2)
