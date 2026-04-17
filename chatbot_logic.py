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
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

# Model constants - Haiku for speed, Sonnet for accuracy-critical tasks
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

NOT_FOUND_AR = "عذراً، لا تتوفر لدي معلومات كافية في الوثائق المتاحة للإجابة على هذا السؤال."
NOT_FOUND_EN = "Sorry, I could not find relevant information in the available documents."
VIOLATION_AR = "عذراً، لا يمكنني الرد على هذا النوع من الرسائل. يرجى استخدام لغة لائقة."

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
        self.llm = Anthropic(
            model=HAIKU_MODEL,
            api_key=api_key,
            max_tokens=800,
            temperature=0.0,
        )
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.chroma_client = chromadb.PersistentClient(path=str(self.data_dir))
        self.collection = self.chroma_client.get_or_create_collection(name="archive_documents")
        self._cached_query_engine: Optional[RetrieverQueryEngine] = None

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

    def _build_query_engine(self) -> RetrieverQueryEngine:
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        return RetrieverQueryEngine.from_args(retriever=retriever, llm=self.llm)

    def _get_query_engine(self) -> RetrieverQueryEngine:
        """Lazy-initialize and cache the query engine."""
        if self._cached_query_engine is None:
            self._cached_query_engine = self._build_query_engine()
        return self._cached_query_engine

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

        show_source = bool(department or decree_number or section)
        return {
            "document": original_filename,
            "department": department,
            "section": section,
            "decree_number": decree_number,
            "year": year,
            "source_label": label,
            "show_source": show_source,
        }

    def _direct_chromadb_search(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        """Direct ChromaDB query - faster than going through LlamaIndex."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or []
            return docs, metas
        except Exception as exc:
            logger.warning("Direct ChromaDB search failed: %s", exc)
            return [], []

    def stream_ask(
        self,
        question: str,
        ip: str = "unknown",
        session_id: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous generator for streaming chat responses.
        Yields dicts: {"chunk": str, "done": bool, "source": dict, "contains_violation": bool}
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

        # 6. Direct ChromaDB search + streaming Claude Haiku response
        try:
            docs, metas = self._direct_chromadb_search(search_question, n_results=3)

            if not docs:
                self._log_chat(q_raw, fallback, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
                yield {"chunk": fallback, "done": True, "source": {"show_source": False}, "contains_violation": False, "source_file_ids": []}
                return

            context = "\n\n---\n\n".join(docs[:3])
            source = self._build_source(metas[0] if metas else {}, lang_code)
            # Collect source file IDs for the user portal
            source_file_ids = [m.get("file_id", "") for m in metas if m.get("file_id")]

            org_name = "الهيئة العامة للقوى العاملة - دولة الكويت"
            system_prompt = (
                f"أنت المساعد الذكي الرسمي لـ {org_name}.\n"
                "مهمتك مساعدة الموظفين والمراجعين في الاطلاع على الوثائق والقرارات واللوائح.\n"
                "أجب فقط بناءً على الوثائق المتاحة أدناه. لا تستخدم معلومات خارجية.\n"
                f"إذا لم تجد الإجابة، قل: {fallback}\n\n"
                f"الوثائق المتاحة:\n{context}"
            )

            full_response = ""
            try:
                with self.raw_client.messages.stream(
                    model=HAIKU_MODEL,
                    max_tokens=500,
                    system=system_prompt,
                    messages=[{"role": "user", "content": search_question}],
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        yield {"chunk": text, "done": False, "source": {}, "contains_violation": False}
            except Exception as stream_exc:
                logger.warning("Streaming failed, trying non-streaming: %s", stream_exc)
                # Fallback to non-streaming
                resp = self.raw_client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": search_question}],
                )
                full_response = resp.content[0].text if resp.content else fallback
                yield {"chunk": full_response, "done": False, "source": {}, "contains_violation": False}

            self._log_chat(q_raw, full_response, lang, False, "none", ip, session_id, int(time.time() * 1000) - start_ms)
            yield {"chunk": "", "done": True, "source": source, "contains_violation": False, "source_file_ids": source_file_ids}

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
