import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import chromadb
from anthropic import Anthropic as RawAnthropic
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Best model for Arabic legal accuracy
OPUS_MODEL = "claude-opus-4-5"
# Keep aliases so existing references resolve without change
HAIKU_MODEL = OPUS_MODEL
SONNET_MODEL = OPUS_MODEL

NOT_FOUND_AR = "عذراً، لم أجد معلومات كافية للإجابة على سؤالك في الوثائق المتاحة."
NOT_FOUND_EN = "Sorry, I could not find sufficient information to answer your question in the available documents."
VIOLATION_AR = "عذراً، لا يمكنني الرد على هذا النوع من الرسائل. يرجى استخدام لغة لائقة."

LEGAL_PERSONA_PROMPT = f"""أنت حاصل على درجة الدكتوراه في اللغة العربية وفقه اللغة،
ومستشار قانوني معتمد متخصص في القانون الكويتي وقوانين العمل.

لديك:
- دكتوراه في اللغة العربية وآدابها
- خبرة 20 عاماً في الاستشارات القانونية الكويتية
- إلمام تام بقانون العمل في القطاعين العام والخاص
- قدرة على الحساب الدقيق لمكافآت نهاية الخدمة والحقوق العمالية
- فهم عميق للمصطلحات القانونية والإدارية الكويتية

عند الإجابة على الأسئلة:
- استخدم عقلك ومعرفتك القانونية العامة بالإضافة للوثائق
- يمكنك الإجابة على أسئلة قانون العمل العامة من خبرتك
- يمكنك إجراء حسابات مكافأة نهاية الخدمة بدقة
- لا تطلب وثائق لإجابة أسئلة قانونية معروفة
- اجمع بين معرفتك العامة والوثائق المرفوعة معاً

للحسابات العمالية استخدم قانون العمل الكويتي رقم 6 لسنة 2010:
- مكافأة نهاية الخدمة للقطاع الخاص:
  * أول 5 سنوات: 15 يوم راتب عن كل سنة
  * ما زاد عن 5 سنوات: 30 يوم راتب عن كل سنة
  * الحساب: (الراتب ÷ 30) × عدد الأيام المستحقة
- إذا كانت الاستقالة من الموظف تُطبق نسب مختلفة:
  * أقل من 5 سنوات: لا مكافأة
  * 5 إلى 10 سنوات: 50% من المكافأة
  * 10 سنوات فأكثر: 100% من المكافأة

أنت مستشار قانوني وخبير في اللوائح والقرارات الحكومية الكويتية،
متخصص في قوانين العمل وقرارات الهيئة العامة للقوى العاملة.

خلفيتك وخبرتك:
- خبير في القانون الكويتي وأنظمة العمل
- متمرس في قراءة وتفسير القرارات الوزارية والتعاميم الإدارية
- تفهم المصطلحات القانونية العربية بدقة
- تعرف كيف تتداخل القرارات وكيف يلغي الجديد القديم
- تفهم الاستثناءات والحالات الخاصة في القانون

مهاراتك في قراءة القرارات:
1. تميز بين القرار الحالي والقرارات المُستشهد بها كمراجع
2. تفهم أن "استثناءً من القرار رقم X" يعني أن القرار الجديد يخالف القديم وهذا الجديد هو الحاكم
3. تستخرج الشروط والاستثناءات والمدد الزمنية بدقة
4. تفهم المصطلحات: يُسمح، يُحظر، استثناء، مع عدم الإخلال، اعتباراً من، حتى تاريخ، بشرط موافقة
5. تفهم التسلسل الهرمي للقرارات: الأحدث يلغي الأقدم

قواعد الإجابة:
- أجب كمستشار قانوني محترف بلغة واضحة ومبسطة
- اذكر رقم القرار والسنة دائماً
- اذكر الشروط المطلوبة بوضوح
- اذكر المدة الزمنية للسريان إن وجدت
- إذا كان هناك استثناء مؤقت، وضح ذلك بجلاء
- استخدم نقاط أو أرقام لتوضيح الشروط المتعددة
- إذا لم تجد معلومات كافية قل فقط: '{NOT_FOUND_AR}'

قواعد القراءة القانونية الحرجة:
- المستند المقدم لك هو القرار الحالي الحاكم
- أي قرارات مذكورة داخله هي مراجع قديمة فقط
- كلمة "استثناءً من" تعني أن الحكم الجديد يختلف لصالح المستخدم
- الأحدث دائماً يطغى على الأقدم
- اقرأ المادة الأولى والثانية بعناية - فيهما جوهر القرار

قاعدة اللغة:
- إذا كان السؤال بالعربية أجب بالعربية
- إذا كان السؤال بالإنجليزية أجب بالإنجليزية

مهم: في نهاية إجابتك اكتب في سطر منفصل:
SOURCES_USED: [أرقام المستندات التي استخدمتها مفصولة بفاصلة، مثال: 1 أو 1,2]
إذا لم تستخدم أي مستند اكتب: SOURCES_USED: none"""

# ── Arabic/Western numeral helpers (mirrored from archive_logic) ──────────────
_AR2W = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_numbers(text: str) -> str:
    """Convert Arabic-Indic numerals to Western numerals."""
    if not text:
        return text
    return str(text).translate(_AR2W)

def extract_decree_from_question(question: str) -> dict:
    """Extract decree number and year from a user question."""
    normalized = normalize_numbers(question)
    patterns = [
        r'رقم\s*[\(\s]*(\d+)',
        r'القرار\s+(\d+)',
        r'تعميم\s+(\d+)',
        r'وزاري\s+(\d+)',
        r'number\s+(\d+)',
        r'no\.?\s*(\d+)',
        r'#\s*(\d+)',
    ]
    decree_num = None
    for pattern in patterns:
        m = re.search(pattern, normalized, re.IGNORECASE)
        if m:
            decree_num = m.group(1)
            break
    year_m = re.search(r'(20\d{2})', normalized)
    year = year_m.group(1) if year_m else None
    return {"decree_number": decree_num, "year": year}

def load_stored_text_chatbot(file_id: str, base_dir: Optional[Path] = None) -> str:
    """Load persisted extracted text for a file (written during indexing)."""
    candidates = []
    if base_dir:
        candidates.append(base_dir / "data" / "texts" / f"{file_id}.txt")
    candidates += [Path("data/texts") / f"{file_id}.txt", Path("./data/texts") / f"{file_id}.txt"]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""

def load_stored_summary_chatbot(file_id: str, base_dir: Optional[Path] = None) -> str:
    """Load persisted smart summary for a file (written during indexing)."""
    candidates = []
    if base_dir:
        candidates.append(base_dir / "data" / "texts" / f"{file_id}_summary.txt")
    candidates += [
        Path("data/texts") / f"{file_id}_summary.txt",
        Path("./data/texts") / f"{file_id}_summary.txt",
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""

GREETING_KEYWORDS = [
    "السلام", "سلام", "مرحبا", "مرحباً", "هلا", "صباح", "مساء",
    "أهلا", "أهلاً", "تحية", "hello", "hi", "hey", "good morning",
    "good afternoon", "greetings", "howdy",
]

FINANCIAL_KEYWORDS = [
    "مبلغ", "قيمة", "إجمالي", "فاتورة", "فواتير", "أمر شراء", "أوامر شراء",
    "عقد", "عقود", "مصروف", "دفع", "دينار", "KWD", "كم عدد", "كم مبلغ",
    "كم قيمة", "إجمالي الفواتير", "إجمالي أوامر", "مجموع",
    "purchase order", "invoice", "contract", "total amount", "payment",
]

# Keywords that indicate the question is about HR / labour law (not financial ops)
_HR_LAW_KEYWORDS = [
    "مكافأة", "نهاية خدمة", "راتب", "إجازة", "فصل", "استقالة",
    "عقد عمل", "قانون العمل", "تعويض", "أجر", "عمالة", "توظيف",
    "إجازة سنوية", "بدل", "مستحقات", "قرار وزاري", "تعميم",
    "indemnity", "salary", "resignation", "labor law", "annual leave",
    "labour", "termination", "employment contract",
]
_FINANCIAL_DOC_TYPES = {"purchase_order", "invoice", "financial"}


def _is_source_relevant(question: str, source_meta: dict) -> bool:
    """Return False when a financial document is matched to an HR/law question.

    This prevents purchase-order source cards from appearing under answers
    about ministerial decrees or labour-law topics.
    """
    doc_type = (source_meta.get("doc_type") or "").lower()
    if doc_type not in _FINANCIAL_DOC_TYPES:
        return True  # non-financial docs are always relevant
    q_lower = question.lower()
    return not any(kw in q_lower for kw in _HR_LAW_KEYWORDS)


class ChatbotService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        load_dotenv()
        load_dotenv(self.base_dir / ".env")
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.data_dir / "chat_logs.json"
        self.files_metadata_path = self.data_dir / "files_metadata.json"
        self._ensure_logs_store()

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.raw_client = RawAnthropic(api_key=api_key)

        _chroma_path = str(self.data_dir / "chromadb")
        os.makedirs(_chroma_path, exist_ok=True)
        self._onnx_ef = ONNXMiniLM_L6_V2()
        self.chroma_client = chromadb.PersistentClient(path=_chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self._onnx_ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_logs_store(self) -> None:
        if not self.logs_path.exists():
            with self.logs_path.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _load_logs(self) -> List[Dict[str, str]]:
        self._ensure_logs_store()
        with self.logs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _save_logs(self, logs: List[Dict[str, str]]) -> None:
        with self.logs_path.open("w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def _log_chat(
        self,
        user_message: str,
        bot_response: str,
        language: str,
        contains_violation: bool,
        violation_type: str,
        ip_address: str = "unknown",
        session_id: str = "",
        response_time_ms: int = 0,
    ) -> None:
        logs = self._load_logs()
        now = datetime.utcnow()
        logs.append(
            {
                "id": uuid.uuid4().hex,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "ip_address": ip_address,
                "session_id": session_id,
                "user_message": user_message,
                "bot_response": bot_response,
                "language": language,
                "response_time_ms": response_time_ms,
                "contains_violation": contains_violation,
                "violation_type": violation_type,
            }
        )
        self._save_logs(logs)

    def get_logs(self) -> List[Dict[str, str]]:
        return list(reversed(self._load_logs()))

    def get_logs_csv(self) -> str:
        logs = self._load_logs()
        rows = ["id,date,time,ip_address,session_id,user_message,bot_response,language,response_time_ms,contains_violation,violation_type"]
        for row in logs:
            vals = [
                str(row.get("id", "")),
                str(row.get("date", row.get("timestamp", "")[:10])),
                str(row.get("time", "")),
                str(row.get("ip_address", "unknown")),
                str(row.get("session_id", "")),
                str(row.get("user_message", "")).replace('"', '""'),
                str(row.get("bot_response", "")).replace('"', '""'),
                str(row.get("language", "")),
                str(row.get("response_time_ms", 0)),
                str(row.get("contains_violation", False)).lower(),
                str(row.get("violation_type", "")),
            ]
            rows.append(",".join(f'"{v}"' for v in vals))
        return "\n".join(rows)

    def get_logs_stats(self) -> Dict[str, str]:
        logs = self._load_logs()
        total = len(logs)
        violations = sum(1 for l in logs if l.get("contains_violation"))
        lang_count: Dict[str, int] = {}
        for l in logs:
            lang = l.get("language", "Other")
            lang_count[lang] = lang_count.get(lang, 0) + 1
        top_lang = max(lang_count, key=lang_count.get) if lang_count else "N/A"
        return {"total": total, "violations": violations, "top_language": top_lang}

    def _load_files_metadata(self) -> List[Dict[str, Any]]:
        if not self.files_metadata_path.exists():
            return []
        try:
            with self.files_metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "files" in data:
                return data["files"] if isinstance(data["files"], list) else []
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _is_financial_query(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in FINANCIAL_KEYWORDS)

    def _answer_financial_query(self, question: str) -> Tuple[str, bool]:
        """
        Try to answer financial/analytical questions directly from structured metadata.
        Returns (answer, was_answered). If was_answered is False, fall through to RAG.
        """
        files = self._load_files_metadata()
        ready_files = [f for f in files if f.get("status", "ready") == "ready"]
        if not ready_files:
            return "", False

        # Build a compact structured data summary for Claude
        structured_summary = []
        for f in ready_files:
            sd = f.get("structured_data")
            if not sd:
                continue
            dt = f.get("doc_type") or sd.get("doc_type") or "other"
            entry: Dict[str, Any] = {
                "doc_type": dt,
                "filename": f.get("original_filename", ""),
                "year": f.get("year", ""),
                "department": f.get("department_name_ar", ""),
            }
            if dt == "purchase_order":
                entry.update({
                    "po_number": sd.get("po_number"),
                    "date": sd.get("date"),
                    "vendor_name": sd.get("vendor_name"),
                    "total_amount": sd.get("total_amount"),
                    "currency": sd.get("currency", "KWD"),
                    "cost_center": sd.get("cost_center"),
                    "items": sd.get("items", []),
                })
            elif dt == "invoice":
                entry.update({
                    "invoice_number": sd.get("invoice_number"),
                    "date": sd.get("date"),
                    "customer_name": sd.get("customer_name"),
                    "vendor_name": sd.get("vendor_name"),
                    "total_amount": sd.get("total_amount"),
                    "currency": sd.get("currency", "KWD"),
                    "contract_reference": sd.get("contract_reference"),
                })
            elif dt == "contract":
                entry.update({
                    "contract_title": sd.get("contract_title"),
                    "party_one": sd.get("party_one"),
                    "party_two": sd.get("party_two"),
                    "start_date": sd.get("start_date"),
                    "end_date": sd.get("end_date"),
                    "duration_months": sd.get("duration_months"),
                    "total_value": sd.get("total_value"),
                    "currency": sd.get("currency", "KWD"),
                    "payment_terms": sd.get("payment_terms"),
                })
            elif dt == "circular":
                entry.update({
                    "circular_number": sd.get("circular_number"),
                    "circular_year": sd.get("year"),
                    "issuing_authority": sd.get("issuing_authority"),
                    "subject": sd.get("subject"),
                })
            structured_summary.append(entry)

        if not structured_summary:
            return "", False

        summary_json = json.dumps(structured_summary, ensure_ascii=False, indent=2)

        system_prompt = (
            "أنت مساعد مالي وإداري ذكي. لديك وصول إلى بيانات منظمة مستخرجة من الوثائق.\n\n"
            "عند الإجابة على الأسئلة المالية:\n"
            "- اعمل حسابات دقيقة من البيانات المتوفرة\n"
            "- اذكر الأرقام المحددة مع فاصلة الألوف والعملة\n"
            "- اذكر مصادر المعلومات (رقم الأمر/الفاتورة/العقد)\n"
            "- إذا طُلب إجمالي، اجمع المبالغ بدقة\n"
            "- إذا طُلب عدد، عدّ العناصر المطابقة\n\n"
            "أمثلة على الإجابات:\n"
            "- 'إجمالي أوامر الشراء لإنوفا في 2026 هو 4,830 د.ك (أمر رقم 4359 بتاريخ 05/04/2026)'\n"
            "- 'يوجد فاتورة واحدة لسنة 2026 بقيمة 2,100 د.ك (SU-INN-25-0088)'\n\n"
            f"البيانات المتوفرة:\n{summary_json}"
        )

        try:
            response = self.raw_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=700,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            answer = response.content[0].text.strip() if response.content else ""
            if answer and NOT_FOUND_AR not in answer and NOT_FOUND_EN not in answer:
                return answer, True
        except Exception as exc:
            logger.warning("_answer_financial_query failed: %s", exc)

        return "", False

    def _is_collection_empty(self) -> bool:
        return self.collection.count() == 0

    def _is_arabic(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def _is_greeting(self, text: str) -> bool:
        t = text.strip().lower()
        # Fast local check - no API call needed
        if any(g in t for g in GREETING_KEYWORDS):
            return True
        # Also check very short messages as likely greetings
        if len(t) < 8 and not any(c.isdigit() for c in t):
            return True
        return False

    def _greeting_response(self, lang: str) -> str:
        if lang == "en":
            return "Hello! I'm the official smart assistant of the Public Authority for Manpower, Kuwait. How can I help you today?"
        return "وعليكم السلام ورحمة الله وبركاته! 👋\n\nأهلاً وسهلاً بك في نظام الأرشفة الذكي للهيئة العامة للقوى العاملة.\n\nيمكنني مساعدتك في الاطلاع على الوثائق والقرارات واللوائح المؤسسية. كيف أستطيع مساعدتك اليوم؟"

    def _detect_violation(self, message: str) -> Dict[str, str]:
        prompt = (
            "Does this message contain profanity, insults, hate speech, or "
            'inappropriate content? Reply with JSON only: '
            '{"contains_violation": true/false, "violation_type": "profanity/hate/inappropriate/none"}'
        )
        response = self.raw_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=120,
            temperature=0,
            system=prompt,
            messages=[{"role": "user", "content": message}],
        )
        raw = response.content[0].text.strip() if response.content else "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"contains_violation": False, "violation_type": "none"}
        return {
            "contains_violation": bool(parsed.get("contains_violation", False)),
            "violation_type": str(parsed.get("violation_type", "none")),
        }

    def _correct_typos(self, question: str) -> str:
        text = (question or "").strip()
        if not text:
            return text
        try:
            if self._is_arabic(text):
                user_content = (
                    "صحح الأخطاء الإملائية في هذا النص فقط، لا تغير المعنى، "
                    "أرجع النص المصحح فقط بدون أي كلام آخر:\n"
                    f"{text}"
                )
            else:
                user_content = (
                    "Fix only spelling and typos in the following text. "
                    "Do not change meaning. Return only the corrected text, nothing else:\n"
                    f"{text}"
                )
            response = self.raw_client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": user_content}],
            )
            corrected = (response.content[0].text if response.content else "").strip()
            return corrected if corrected else text
        except Exception:
            logger.exception("_correct_typos failed; using original question")
            return text

    def _build_source(self, metadata: Dict[str, Any], lang: str) -> Dict[str, Any]:
        def _s(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return str(val).strip()

        original_filename = _s(metadata.get("original_filename") or metadata.get("file_name", ""))
        department = (
            _s(metadata.get("department_name_ar", ""))
            if lang == "ar"
            else _s(metadata.get("department_name_en", ""))
        ) or _s(metadata.get("department", ""))
        section = (
            _s(metadata.get("section_name_ar", ""))
            if lang == "ar"
            else _s(metadata.get("section_name_en", ""))
        ) or _s(metadata.get("section", ""))
        decree_number = _s(metadata.get("decree_number", ""))
        year = _s(metadata.get("year", ""))

        if lang == "ar":
            sentence_parts: List[str] = []
            if decree_number:
                dec = f"بناءً على قرار رقم {decree_number}"
                if year:
                    dec += f" لسنة {year}"
                sentence_parts.append(dec)
            elif year:
                sentence_parts.append(f"لسنة {year}")
            if section:
                sentence_parts.append(section)
            if department:
                sentence_parts.append(f"إدارة {department}")
            summary_line = "، ".join(sentence_parts) if sentence_parts else ""
            if original_filename:
                label = f"📄 {original_filename}" + (f" | {summary_line}" if summary_line else "")
            else:
                label = summary_line or ""
        else:
            parts: List[str] = []
            if original_filename:
                parts.append(f"📄 {original_filename}")
            if section:
                parts.append(section)
            if department:
                parts.append(f"Department {department}")
            if decree_number:
                dec = f"Decree {decree_number}"
                if year:
                    dec += f" ({year})"
                parts.append(dec)
            elif year:
                parts.append(f"Year {year}")
            label = " | ".join(parts)

        file_id = _s(metadata.get("file_id", ""))
        is_public = bool(metadata.get("is_public", True))
        view_url = f"/files/{file_id}/view" if file_id else ""
        show_source = bool(department or decree_number or section or original_filename)
        return {
            "document": original_filename,
            "department": department,
            "section": section,
            "decree_number": decree_number,
            "year": year,
            "source_label": label,
            "show_source": show_source,
            "file_id": file_id,
            "is_public": is_public,
            "view_url": view_url,
        }

    @staticmethod
    def _normalize_arabic_numbers(text: str) -> str:
        """Convert Arabic-Indic numerals (٠-٩) to Western numerals (0-9)."""
        return normalize_numbers(text)

    @staticmethod
    def _build_query_variations(question: str) -> List[str]:
        """Generate query variations to maximise ChromaDB recall."""
        q = normalize_numbers(question)
        variations = [q]

        # Decree number normalisation: "قرار وزاري رقم 9 لسنة 2026" variations
        decree_patterns = [
            (r"القرار الوزاري رقم", "قرار وزاري رقم"),
            (r"قرار وزاري رقم", "القرار الوزاري رقم"),
            (r"قرار رقم", "قرار وزاري رقم"),
            (r"التعميم رقم", "تعميم رقم"),
            (r"تعميم رقم", "التعميم رقم"),
            (r"أمر شراء رقم", "أمر الشراء رقم"),
            (r"فاتورة رقم", "الفاتورة رقم"),
        ]
        for pat, repl in decree_patterns:
            alt = re.sub(pat, repl, q)
            if alt != q:
                variations.append(alt)

        # Extract key terms (decree number + year) as a short focused query
        numbers = re.findall(r"\d+", q)
        if numbers:
            variations.append(" ".join(numbers))

        return list(dict.fromkeys(variations))  # deduplicate preserving order

    def _metadata_keyword_search(self, question: str) -> Tuple[List[str], List[Dict]]:
        """
        Fallback: parse decree/year from the question, search files_metadata.json,
        then load the stored text from disk (avoids depending on ChromaDB chunks).
        """
        q = normalize_numbers(question)
        decree_info = extract_decree_from_question(q)
        decree_num = decree_info.get("decree_number")
        year = decree_info.get("year")

        # Also gather all numbers in case pattern extraction missed them
        all_numbers = re.findall(r"\d+", q)
        year_candidates = {n for n in all_numbers if len(n) == 4}
        decree_candidates = {n for n in all_numbers if 1 <= len(n) <= 4}
        if decree_num:
            decree_candidates.add(decree_num)
        if year:
            year_candidates.add(year)

        if not decree_candidates and not year_candidates:
            return [], []

        try:
            meta_path = self.base_dir / "data" / "files_metadata.json"
            if not meta_path.exists():
                return [], []
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            files = data.get("files", data) if isinstance(data, dict) else data

            matched: List[Dict] = []
            for f in files:
                if f.get("status", "ready") != "ready":
                    continue

                # Normalize stored decree/year numbers
                sd = f.get("structured_data") or {}
                file_decree = normalize_numbers(str(
                    sd.get("decree_number") or sd.get("circular_number") or
                    f.get("decree_number") or ""
                )).strip()
                file_year = normalize_numbers(str(
                    sd.get("year") or f.get("year") or ""
                )).strip()

                decree_match = bool(decree_candidates and file_decree and file_decree in decree_candidates)
                year_match = bool(year_candidates and file_year and file_year in year_candidates)

                if decree_match and year_match:
                    matched.append(f)
                elif decree_match and not year_candidates:
                    matched.append(f)
                elif year_match and not decree_candidates:
                    matched.append(f)

            if not matched:
                return [], []

            docs, metas = [], []
            for f in matched[:5]:
                file_id = f.get("id") or f.get("file_id", "")
                # Prefer smart summary → raw text → metadata fallback
                text = load_stored_summary_chatbot(file_id, self.base_dir)
                if not text:
                    text = load_stored_text_chatbot(file_id, self.base_dir)
                if not text:
                    text = " ".join(filter(None, [
                        f.get("summary", ""),
                        f.get("main_topic", ""),
                        f.get("document_type", ""),
                        f.get("original_filename", ""),
                    ]))
                if not text:
                    continue
                meta = {
                    "file_id": file_id,
                    "original_filename": f.get("original_filename", ""),
                    "file_name": f.get("file_name", ""),
                    "department_name_ar": f.get("department_name_ar", ""),
                    "department_name_en": f.get("department_name_en", ""),
                    "section_name_ar": f.get("section_name_ar", ""),
                    "section_name_en": f.get("section_name_en", ""),
                    "year": file_year,
                    "decree_number": file_decree,
                    "doc_type": f.get("doc_type", ""),
                    "is_public": str(f.get("is_public", True)),
                }
                docs.append(text[:4000])  # cap to avoid huge context
                metas.append(meta)

            return docs, metas
        except Exception as exc:
            logger.warning("Metadata keyword search failed: %s", exc)
            return [], []

    def _direct_chromadb_search(self, query: str, n_results: int = 8) -> Tuple[List[str], List[Dict]]:
        """
        Multi-query ChromaDB search with 2-pass prioritization:
        smart summaries are returned first, then raw chunks.
        """
        norm_query = normalize_numbers(query)
        variations = self._build_query_variations(norm_query)
        # doc_key -> (doc, meta, distance)
        seen_docs: Dict[str, tuple] = {}

        for variation in variations:
            try:
                results = self.collection.query(
                    query_texts=[variation],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                docs  = results.get("documents",  [[]])[0] or []
                metas = results.get("metadatas",   [[]])[0] or []
                dists = results.get("distances",   [[]])[0] or []
                for doc, meta, dist in zip(docs, metas, dists):
                    if dist > 1.5:
                        continue
                    key = (doc or "")[:80]
                    if key and key not in seen_docs:
                        seen_docs[key] = (doc, meta, dist)
            except Exception as exc:
                logger.warning("ChromaDB variation search failed: %s", exc)

        # Sort by distance ascending
        ranked = sorted(seen_docs.values(), key=lambda x: x[2])

        if not ranked:
            # Keyword/metadata fallback when vector search finds nothing
            kb_docs, kb_metas = self._metadata_keyword_search(query)
            for doc, meta in zip(kb_docs, kb_metas):
                key = (doc or "")[:80]
                if key and key not in seen_docs:
                    ranked.append((doc, meta, 0.0))

        # 2-pass: smart summaries first, then raw chunks
        summary_results = [(d, m, dist) for d, m, dist in ranked if m.get("content_type") == "smart_summary"]
        chunk_results   = [(d, m, dist) for d, m, dist in ranked if m.get("content_type") != "smart_summary"]
        ordered = summary_results + chunk_results

        final_docs  = [r[0] for r in ordered[:n_results]]
        final_metas = [r[1] for r in ordered[:n_results]]
        return final_docs, final_metas

    def stream_ask(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        ip: str = "unknown",
        session_id: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous generator for streaming chat responses.
        Yields dicts: {"chunk": str, "done": bool, "source": dict, "contains_violation": bool}
        Pass conversation_history as [{role, content}, ...] to enable multi-turn context.
        """
        start_ms = int(time.time() * 1000)
        q_raw = (question or "").strip()
        if not q_raw:
            yield {"chunk": NOT_FOUND_AR, "done": True, "source": {"show_source": False}, "contains_violation": False}
            return

        lang = "Arabic" if self._is_arabic(q_raw) else "English"
        lang_code = "ar" if lang == "Arabic" else "en"
        fallback = NOT_FOUND_AR if lang == "Arabic" else NOT_FOUND_EN

        # 1. Fast greeting detection (no API call)
        if self._is_greeting(q_raw):
            greeting = self._greeting_response(lang_code)
            self._log_chat(q_raw, greeting, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": greeting, "done": True, "source": {"show_source": False, "source_label": ""}, "contains_violation": False}
            return

        # 2. Violation check with fast Haiku model
        try:
            violation = self._detect_violation(q_raw)
        except Exception:
            violation = {"contains_violation": False, "violation_type": "none"}

        if violation.get("contains_violation"):
            self._log_chat(q_raw, VIOLATION_AR, lang, True, str(violation.get("violation_type", "none")), ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": VIOLATION_AR, "done": True, "source": {"show_source": False}, "contains_violation": True}
            return

        # 3. Financial queries answered from structured data (Sonnet for accuracy)
        if self._is_financial_query(q_raw):
            fin_answer, was_answered = self._answer_financial_query(q_raw)
            if was_answered and fin_answer:
                self._log_chat(q_raw, fin_answer, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
                yield {"chunk": fin_answer, "done": True, "source": {"show_source": True, "source_label": "📊 البيانات المالية المنظمة"}, "contains_violation": False}
                return

        # 4. Empty collection fallback
        if self._is_collection_empty():
            self._log_chat(q_raw, fallback, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": fallback, "done": True, "source": {"show_source": False}, "contains_violation": False}
            return

        # 5. Typo correction (skip for short texts - saves one API round trip)
        search_question = q_raw
        if len(q_raw) > 15:
            try:
                corrected = self._correct_typos(q_raw)
                if corrected and corrected.strip():
                    search_question = corrected
            except Exception:
                pass

        # 6. Direct ChromaDB search (smart summaries prioritized) + streaming response
        try:
            docs, metas = self._direct_chromadb_search(search_question, n_results=8)

            if not docs:
                self._log_chat(q_raw, fallback, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
                yield {"chunk": fallback, "done": True, "source": {"show_source": False}, "contains_violation": False, "source_file_ids": []}
                return

            # ── Build numbered context grouped by file_id ──────────────────────
            # Group chunks/summaries by file_id, preserving order of first appearance
            file_order: List[str] = []
            file_to_meta: Dict[str, Any] = {}
            file_to_docs: Dict[str, List[str]] = {}
            for doc, meta in zip(docs, metas):
                fid = meta.get("file_id", "")
                if not fid:
                    continue
                if fid not in file_to_meta:
                    file_order.append(fid)
                    file_to_meta[fid] = meta
                    file_to_docs[fid] = []
                file_to_docs[fid].append(doc)

            # Build source_files_list (ordered, for SOURCES_USED parsing)
            source_files_list: List[Dict[str, Any]] = []
            numbered_parts: List[str] = []
            for idx, fid in enumerate(file_order[:5], 1):
                meta = file_to_meta[fid]
                # Prefer smart summary chunk; take at most 2 chunks
                chunks_for_file = file_to_docs[fid]
                summary_chunks = [c for c, m in zip(docs, metas)
                                  if m.get("file_id") == fid and m.get("content_type") == "smart_summary"]
                raw_chunks = [c for c, m in zip(docs, metas)
                              if m.get("file_id") == fid and m.get("content_type") != "smart_summary"]
                combined = (summary_chunks + raw_chunks)[:2]
                combined_text = "\n\n".join(combined)
                numbered_parts.append(f"[مستند {idx}]:\n{combined_text}")

                s = self._build_source(meta, lang_code)
                source_files_list.append({
                    "file_id": fid,
                    "filename": s["document"],
                    "department": s["department"],
                    "section": s["section"],
                    "decree_number": s["decree_number"],
                    "year": s["year"],
                    "view_url": s["view_url"],
                    "is_public": s["is_public"],
                    "doc_type": meta.get("doc_type", ""),
                })

            numbered_context = "\n\n---\n\n".join(numbered_parts)
            source = self._build_source(file_to_meta.get(file_order[0], {}) if file_order else {}, lang_code)

            system_prompt = LEGAL_PERSONA_PROMPT + f"\n\nالوثائق المتاحة:\n{numbered_context}"

            # Build messages array: up to 6 previous turns + current question
            _hist_turns: List[Dict[str, Any]] = []
            for _turn in (conversation_history or [])[-6:]:
                _role = _turn.get("role", "")
                _content = str(_turn.get("content", "")).strip()
                if _role in ("user", "assistant") and _content:
                    _hist_turns.append({"role": _role, "content": _content})
            _hist_turns.append({"role": "user", "content": search_question})

            # ── Stream response, buffering to strip SOURCES_USED: from output ──
            # Use just "SOURCES_USED:" (no leading newline) for robust matching
            SOURCES_MARKER = "SOURCES_USED:"
            # Buffer slightly more than marker length so partial tokens don't slip through
            BUFFER_WINDOW = len(SOURCES_MARKER) + 5
            pending_buf = ""
            full_answer = ""
            sources_text = ""
            marker_found = False

            try:
                with self.raw_client.messages.stream(
                    model=HAIKU_MODEL,
                    max_tokens=600,
                    system=system_prompt,
                    messages=_hist_turns,
                ) as stream:
                    for text in stream.text_stream:
                        if marker_found:
                            # After marker: accumulate source indices, yield nothing
                            sources_text += text
                            continue

                        pending_buf += text
                        marker_pos = pending_buf.find(SOURCES_MARKER)
                        if marker_pos != -1:
                            marker_found = True
                            # Strip trailing whitespace/newlines before the marker
                            visible = pending_buf[:marker_pos].rstrip("\n ")
                            sources_text = pending_buf[marker_pos + len(SOURCES_MARKER):]
                            if visible:
                                full_answer += visible
                                yield {"chunk": visible, "done": False, "source": {}, "contains_violation": False}
                            pending_buf = ""
                        else:
                            # Safe to stream: keep BUFFER_WINDOW chars buffered
                            safe_len = max(0, len(pending_buf) - BUFFER_WINDOW)
                            if safe_len > 0:
                                safe = pending_buf[:safe_len]
                                full_answer += safe
                                yield {"chunk": safe, "done": False, "source": {}, "contains_violation": False}
                                pending_buf = pending_buf[safe_len:]

                # Flush remaining buffer after stream ends
                if pending_buf and not marker_found:
                    marker_pos = pending_buf.find(SOURCES_MARKER)
                    if marker_pos != -1:
                        visible = pending_buf[:marker_pos].rstrip("\n ")
                        sources_text += pending_buf[marker_pos + len(SOURCES_MARKER):]
                        if visible:
                            full_answer += visible
                            yield {"chunk": visible, "done": False, "source": {}, "contains_violation": False}
                    else:
                        full_answer += pending_buf
                        yield {"chunk": pending_buf, "done": False, "source": {}, "contains_violation": False}

            except Exception as stream_exc:
                logger.warning("Streaming failed, trying non-streaming: %s", stream_exc)
                resp = self.raw_client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=700,
                    system=system_prompt,
                    messages=_hist_turns,
                )
                raw_text = resp.content[0].text if resp.content else fallback
                marker_pos = raw_text.find(SOURCES_MARKER)
                if marker_pos != -1:
                    full_answer = raw_text[:marker_pos].rstrip("\n ").strip()
                    sources_text = raw_text[marker_pos + len(SOURCES_MARKER):]
                else:
                    full_answer = raw_text
                yield {"chunk": full_answer, "done": False, "source": {}, "contains_violation": False}

            # ── Parse which sources were actually used ─────────────────────────
            sources_text = sources_text.strip()
            verified_sources: List[Dict[str, Any]] = []
            if sources_text and sources_text.lower() != "none":
                try:
                    used_indices = [
                        int(x.strip()) - 1
                        for x in sources_text.replace("،", ",").split(",")
                        if x.strip().isdigit()
                    ]
                    seen_verified: set = set()
                    for i in used_indices:
                        if 0 <= i < len(source_files_list):
                            fid = source_files_list[i].get("file_id", "")
                            if fid and fid not in seen_verified:
                                seen_verified.add(fid)
                                verified_sources.append(source_files_list[i])
                except Exception:
                    pass

            # Strip sources if answer is a "not found" phrase
            NO_ANSWER_PHRASES = [NOT_FOUND_AR, NOT_FOUND_EN, "عذراً، لم أجد", "لا تتوفر", "لم أتمكن"]
            if any(phrase in full_answer for phrase in NO_ANSWER_PHRASES):
                verified_sources = []

            # Remove sources whose doc_type is irrelevant to the question topic
            verified_sources = [s for s in verified_sources if _is_source_relevant(q_raw, s)]

            source_file_ids = [s["file_id"] for s in verified_sources]
            self._log_chat(q_raw, full_answer, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": "", "done": True, "source": source, "sources": verified_sources, "contains_violation": False, "source_file_ids": source_file_ids}

        except Exception as exc:
            logger.exception("stream_ask: ChromaDB/Claude query failed: %s", exc)
            err = "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."
            self._log_chat(q_raw, err, lang, False, "error", ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": err, "done": True, "source": {"show_source": False}, "contains_violation": False, "source_file_ids": []}

    def ask(self, question: str, ip: str = "unknown", session_id: str = "") -> Tuple[str, Dict[str, Any]]:
        start_ms = int(time.time() * 1000)
        q_raw = (question or "").strip()
        if not q_raw:
            return NOT_FOUND_AR, {"show_source": False, "source_label": ""}

        lang = "Arabic" if self._is_arabic(q_raw) else "English"
        lang_code = "ar" if lang == "Arabic" else "en"
        fallback = NOT_FOUND_AR if lang == "Arabic" else NOT_FOUND_EN
        err_technical = (
            "عذراً، حدث خطأ تقني أثناء معالجة سؤالك. حاول مرة أخرى."
            if lang == "Arabic"
            else "Sorry, a technical error occurred while processing your question. Please try again."
        )

        def _log(msg, resp, viol=False, vtype="none"):
            self._log_chat(msg, resp, lang, viol, vtype, ip, session_id, int(time.time() * 1000) - start_ms)

        # Fast greeting detection (no API call)
        if self._is_greeting(q_raw):
            response = self._greeting_response(lang_code)
            _log(q_raw, response)
            return response, {"show_source": False, "source_label": ""}

        try:
            violation = self._detect_violation(q_raw)
        except Exception:
            logger.exception("_detect_violation failed")
            violation = {"contains_violation": False, "violation_type": "none"}

        if violation.get("contains_violation"):
            _log(q_raw, VIOLATION_AR, True, str(violation.get("violation_type", "none")))
            return VIOLATION_AR, {"show_source": False, "source_label": ""}

        # Financial query shortcut (structured data)
        if self._is_financial_query(q_raw):
            fin_answer, was_answered = self._answer_financial_query(q_raw)
            if was_answered and fin_answer:
                _log(q_raw, fin_answer)
                return fin_answer, {"show_source": True, "source_label": "📊 البيانات المالية المنظمة"}

        if self._is_collection_empty():
            _log(q_raw, fallback)
            return fallback, {"show_source": False, "source_label": ""}

        # Typo correction (skip for short texts)
        search_question = q_raw
        if len(q_raw) > 15:
            corrected = self._correct_typos(q_raw)
            if corrected and corrected.strip():
                search_question = corrected

        try:
            query_engine = self._get_query_engine()
            org_name = "الهيئة العامة للقوى العاملة في الكويت"
            prompt = (
                f"أنت المساعد الذكي الرسمي لـ {org_name}.\n"
                "أجب على الأسئلة بناءً على الوثائق المتاحة فقط. لا تستخدم معلومات خارجية.\n"
                "FINANCIAL QUERY RULES: For amounts/counts, give exact numbers with currency.\n"
                "DOCUMENT SEARCH: Answer ONLY from the retrieved documents.\n"
                f"If not found, say: {fallback}"
            )
            response = query_engine.query(f"{prompt}\n\nUser question: {search_question}")
            answer = str(response).strip()
            source_nodes = response.source_nodes or []
        except Exception:
            logger.exception("ChromaDB / LlamaIndex query_engine.query failed")
            _log(q_raw, err_technical, vtype="error")
            return err_technical, {"show_source": False, "source_label": ""}

        if not source_nodes or not answer:
            _log(q_raw, fallback)
            return fallback, {"show_source": False, "source_label": ""}

        try:
            top_metadata: Dict[str, Any] = dict(source_nodes[0].metadata or {})
            source = self._build_source(top_metadata, lang_code)
        except Exception:
            logger.exception("_build_source failed")
            _log(q_raw, err_technical, vtype="error")
            return err_technical, {"show_source": False, "source_label": ""}

        if fallback.lower() in answer.lower():
            _log(q_raw, fallback)
            return fallback, {"show_source": False, "source_label": ""}

        _log(q_raw, answer)
        return answer, source
