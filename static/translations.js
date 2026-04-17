// ═══════════════════════════════════════════════════════════
// PAM Smart Archive System – Centralized Translations
// الهيئة العامة للقوى العاملة – ملف الترجمات المركزي
// ═══════════════════════════════════════════════════════════

const TRANSLATIONS = {
  ar: {
    app_name: "نظام الأرشفة الذكي",
    app_subtitle: "الهيئة العامة للقوى العاملة",
    app_full: "الهيئة العامة للقوى العاملة",
    app_tagline: "نظام الأرشفة والمساعد الذكي",
    nav_documents: "الوثائق المتاحة",
    nav_reports: "التقارير والإحصاءات",
    nav_search: "البحث",
    nav_history: "محادثاتي السابقة",
    nav_about: "عن النظام",
    nav_dashboard: "الرئيسية",
    nav_archive: "إدارة الأرشيف",
    nav_departments: "الإدارات والأقسام",
    nav_analytics: "التحليلات",
    nav_logs: "سجل المحادثات",
    nav_settings: "الإعدادات",
    nav_notifications: "الإشعارات",
    admin_panel: "لوحة الإدارة الرسمية",
    upload_title: "رفع الملفات",
    upload_drag: "اسحب ملفات PDF هنا أو اضغط للاختيار (متعدد)",
    upload_manual: "تصنيف يدوي",
    upload_auto: "تصنيف تلقائي بالذكاء الاصطناعي 🤖",
    upload_department: "الإدارة",
    upload_section: "القسم",
    upload_year: "السنة",
    upload_btn: "رفع الكل",
    upload_cancel: "إلغاء",
    upload_confirm: "تأكيد الحفظ",
    upload_queue_ready: "جاهز ✅",
    upload_queue_uploading: "جاري الرفع 📤",
    upload_queue_done: "تم ✅",
    upload_queue_error: "خطأ ❌",
    chat_placeholder: "اكتب سؤالك هنا...",
    chat_send: "إرسال",
    chat_welcome: "السلام عليكم! 👋 أنا مساعدك الذكي الرسمي للهيئة العامة للقوى العاملة. كيف يمكنني مساعدتك اليوم؟",
    chat_title: "المساعد الذكي",
    chat_online: "متصل",
    chat_thinking: "جاري البحث في الوثائق...",
    chat_history_title: "محادثاتي السابقة",
    chat_history_clear: "مسح السجل",
    chat_history_empty: "لا توجد محادثات سابقة",
    chat_new: "محادثة جديدة",
    docs_title: "نتائج بحثك",
    docs_empty: "ابدأ بسؤال المساعد الذكي للاطلاع على الوثائق ذات الصلة",
    docs_view: "عرض الملف",
    docs_details: "تفاصيل",
    docs_clear: "مسح النتائج",
    docs_public: "عام 🌐",
    docs_private: "خاص 🔒",
    type_purchase_order: "أمر شراء",
    type_invoice: "فاتورة",
    type_contract: "عقد",
    type_circular: "تعميم",
    type_decree: "قرار",
    type_other: "أخرى",
    status_processing: "جاري المعالجة",
    status_ready: "جاهز",
    status_error: "خطأ",
    status_cancelled: "ملغي",
    btn_save: "حفظ",
    btn_cancel: "إلغاء",
    btn_cancel_processing: "❌ إلغاء",
    btn_delete: "حذف",
    btn_edit: "تعديل",
    btn_view: "عرض",
    btn_download: "تحميل",
    btn_add: "إضافة",
    btn_confirm: "تأكيد",
    btn_close: "إغلاق",
    btn_retry: "إعادة المحاولة",
    dept_title: "الإدارات والأقسام",
    dept_add: "إضافة إدارة",
    dept_section_add: "إضافة قسم",
    dept_name_ar: "الاسم بالعربي",
    dept_name_en: "الاسم بالإنجليزي",
    dept_files_count: "ملف",
    dept_sections_count: "قسم",
    logs_title: "سجل المحادثات",
    logs_date: "التاريخ",
    logs_time: "الوقت",
    logs_ip: "عنوان IP",
    logs_message: "رسالة المستخدم",
    logs_response: "الرد",
    logs_status: "الحالة",
    logs_violation: "مخالفة",
    logs_clean: "سليم ✅",
    logs_filter_all: "الكل",
    logs_filter_violations: "⚠️ المخالفات",
    logs_filter_today: "اليوم",
    logs_filter_week: "هذا الأسبوع",
    logs_copy_ip: "نسخ",
    analytics_total_docs: "إجمالي الوثائق",
    analytics_total_contracts: "العقود",
    analytics_total_pos: "أوامر الشراء",
    analytics_total_invoices: "الفواتير",
    analytics_dept_breakdown: "توزيع الوثائق حسب الإدارة",
    analytics_type_breakdown: "توزيع حسب نوع الوثيقة",
    notif_title: "الإشعارات",
    notif_empty: "لا توجد إشعارات",
    notif_clear: "مسح الكل",
    notif_mark_read: "تعليم الكل كمقروء",
    notif_success: "تم رفع وفهرسة الملف بنجاح",
    notif_error: "فشل معالجة الملف",
    notif_warning: "جودة قراءة منخفضة",
    visibility_public: "🌐 عام - يظهر للمستخدمين",
    visibility_private: "🔒 خاص - للشات بوت فقط",
    visibility_make_public: "جعل عام 🌐",
    visibility_make_private: "جعل خاص 🔒",
    settings_title: "الإعدادات",
    settings_system_name: "اسم النظام",
    settings_language: "اللغة الافتراضية",
    settings_save: "حفظ الإعدادات",
    welcome_title: "مرحباً بك في نظام الأرشفة الذكي",
    welcome_subtitle: "الهيئة العامة للقوى العاملة - دولة الكويت",
    welcome_desc: "يمكنك البحث في الوثائق والقرارات واللوائح المؤسسية باستخدام المساعد الذكي المدعوم بالذكاء الاصطناعي",
    footer_rights: "جميع الحقوق محفوظة © 2026",
    footer_org: "الهيئة العامة للقوى العاملة - دولة الكويت",
    footer_website: "www.pam.gov.kw",
    loading: "جاري التحميل...",
    search_placeholder: "ابحث في الوثائق...",
    no_results: "لا توجد نتائج",
    confirm_delete: "هل أنت متأكد من الحذف؟",
    error_generic: "حدث خطأ، يرجى المحاولة مرة أخرى",
    indexed_count: "الوثائق المفهرسة",
    new_chat: "محادثة جديدة",
    admin_link: "⚙️ لوحة الإدارة",
    about_title: "عن نظام الأرشفة الذكي",
  },
  en: {
    app_name: "Smart Archive System",
    app_subtitle: "Public Authority for Manpower",
    app_full: "Public Authority for Manpower",
    app_tagline: "Smart Archive & AI Assistant",
    nav_documents: "Available Documents",
    nav_reports: "Reports & Analytics",
    nav_search: "Search",
    nav_history: "My Chat History",
    nav_about: "About",
    nav_dashboard: "Dashboard",
    nav_archive: "Archive Management",
    nav_departments: "Departments & Sections",
    nav_analytics: "Analytics",
    nav_logs: "Chat Logs",
    nav_settings: "Settings",
    nav_notifications: "Notifications",
    admin_panel: "Official Admin Panel",
    upload_title: "Upload Files",
    upload_drag: "Drag PDF files here or click to select (multiple)",
    upload_manual: "Manual Classification",
    upload_auto: "Auto-classify with AI 🤖",
    upload_department: "Department",
    upload_section: "Section",
    upload_year: "Year",
    upload_btn: "Upload All",
    upload_cancel: "Cancel",
    upload_confirm: "Confirm Save",
    upload_queue_ready: "Ready ✅",
    upload_queue_uploading: "Uploading 📤",
    upload_queue_done: "Done ✅",
    upload_queue_error: "Error ❌",
    chat_placeholder: "Type your question here...",
    chat_send: "Send",
    chat_welcome: "Hello! 👋 I'm the official smart assistant of the Public Authority for Manpower, Kuwait. How can I help you today?",
    chat_title: "Smart Assistant",
    chat_online: "Online",
    chat_thinking: "Searching documents...",
    chat_history_title: "My Chat History",
    chat_history_clear: "Clear History",
    chat_history_empty: "No previous conversations",
    chat_new: "New Chat",
    docs_title: "Your Search Results",
    docs_empty: "Start a conversation with the smart assistant to see relevant documents",
    docs_view: "View File",
    docs_details: "Details",
    docs_clear: "Clear Results",
    docs_public: "Public 🌐",
    docs_private: "Private 🔒",
    type_purchase_order: "Purchase Order",
    type_invoice: "Invoice",
    type_contract: "Contract",
    type_circular: "Circular",
    type_decree: "Decree",
    type_other: "Other",
    status_processing: "Processing",
    status_ready: "Ready",
    status_error: "Error",
    status_cancelled: "Cancelled",
    btn_save: "Save",
    btn_cancel: "Cancel",
    btn_cancel_processing: "❌ Cancel",
    btn_delete: "Delete",
    btn_edit: "Edit",
    btn_view: "View",
    btn_download: "Download",
    btn_add: "Add",
    btn_confirm: "Confirm",
    btn_close: "Close",
    btn_retry: "Retry",
    dept_title: "Departments & Sections",
    dept_add: "Add Department",
    dept_section_add: "Add Section",
    dept_name_ar: "Name in Arabic",
    dept_name_en: "Name in English",
    dept_files_count: "files",
    dept_sections_count: "sections",
    logs_title: "Chat Logs",
    logs_date: "Date",
    logs_time: "Time",
    logs_ip: "IP Address",
    logs_message: "User Message",
    logs_response: "Response",
    logs_status: "Status",
    logs_violation: "Violation",
    logs_clean: "Clean ✅",
    logs_filter_all: "All",
    logs_filter_violations: "⚠️ Violations Only",
    logs_filter_today: "Today",
    logs_filter_week: "This Week",
    logs_copy_ip: "Copy",
    analytics_total_docs: "Total Documents",
    analytics_total_contracts: "Contracts",
    analytics_total_pos: "Purchase Orders",
    analytics_total_invoices: "Invoices",
    analytics_dept_breakdown: "Documents by Department",
    analytics_type_breakdown: "Documents by Type",
    notif_title: "Notifications",
    notif_empty: "No notifications",
    notif_clear: "Clear All",
    notif_mark_read: "Mark All as Read",
    notif_success: "File uploaded and indexed successfully",
    notif_error: "File processing failed",
    notif_warning: "Low reading quality",
    visibility_public: "🌐 Public - Visible to users",
    visibility_private: "🔒 Private - Chatbot only",
    visibility_make_public: "Make Public 🌐",
    visibility_make_private: "Make Private 🔒",
    settings_title: "Settings",
    settings_system_name: "System Name",
    settings_language: "Default Language",
    settings_save: "Save Settings",
    welcome_title: "Welcome to the Smart Archive System",
    welcome_subtitle: "Public Authority for Manpower - State of Kuwait",
    welcome_desc: "Search documents, decrees and institutional regulations using the AI-powered smart assistant",
    footer_rights: "All Rights Reserved © 2026",
    footer_org: "Public Authority for Manpower - State of Kuwait",
    footer_website: "www.pam.gov.kw",
    loading: "Loading...",
    search_placeholder: "Search documents...",
    no_results: "No results found",
    confirm_delete: "Are you sure you want to delete?",
    error_generic: "An error occurred, please try again",
    indexed_count: "Indexed Documents",
    new_chat: "New Chat",
    admin_link: "⚙️ Admin Panel",
    about_title: "About Smart Archive System",
  }
};

// ═══════════════════════════════════════════════════════════
// Language Manager
// ═══════════════════════════════════════════════════════════
const LangManager = {
  current: localStorage.getItem('pam_language') || 'ar',

  init() {
    this.apply(this.current);
    this._updateToggle();
    // Listen for storage changes (cross-tab sync)
    window.addEventListener('storage', (e) => {
      if (e.key === 'pam_language' && e.newValue && e.newValue !== this.current) {
        this.current = e.newValue;
        this.apply(this.current);
        this._updateToggle();
      }
    });
  },

  toggle() {
    this.current = this.current === 'ar' ? 'en' : 'ar';
    localStorage.setItem('pam_language', this.current);
    this.apply(this.current);
    this._updateToggle();
  },

  apply(lang) {
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.lang = lang;
    document.body.style.fontFamily = lang === 'ar'
      ? "'Cairo', sans-serif"
      : "'Inter', 'Cairo', sans-serif";

    // Translate all marked elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const val = (TRANSLATIONS[lang] || {})[key];
      if (val !== undefined) el.textContent = val;
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      const val = (TRANSLATIONS[lang] || {})[key];
      if (val !== undefined) el.placeholder = val;
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      const key = el.getAttribute('data-i18n-title');
      const val = (TRANSLATIONS[lang] || {})[key];
      if (val !== undefined) el.title = val;
    });

    // Fire event so other scripts can update dynamic content
    document.dispatchEvent(new CustomEvent('languageChanged', { detail: { lang } }));
  },

  get(key) {
    return (TRANSLATIONS[this.current] || {})[key] || (TRANSLATIONS['ar'] || {})[key] || key;
  },

  _updateToggle() {
    document.querySelectorAll('[data-lang-toggle]').forEach(btn => {
      btn.textContent = this.current === 'ar' ? 'EN' : 'عربي';
      btn.title = this.current === 'ar' ? 'Switch to English' : 'التبديل للعربية';
    });
    // Also update legacy id-based buttons
    const btn = document.getElementById('langToggle') || document.getElementById('langBtn');
    if (btn) {
      btn.textContent = this.current === 'ar' ? 'EN' : 'عربي';
    }
  }
};

// ═══════════════════════════════════════════════════════════
// Unified Chat History Manager
// ═══════════════════════════════════════════════════════════
const ChatHistory = {
  STORAGE_KEY: 'pam_chat_history',
  MAX_ITEMS: 50,

  save(question, answer, sources) {
    if (!question || !answer) return;
    let history = this.load();
    history.unshift({
      id: Date.now(),
      timestamp: new Date().toISOString(),
      question: String(question).slice(0, 300),
      answer: String(answer).slice(0, 1000),
      sources: sources || null,
    });
    if (history.length > this.MAX_ITEMS) history = history.slice(0, this.MAX_ITEMS);
    try { localStorage.setItem(this.STORAGE_KEY, JSON.stringify(history)); } catch {}
    this.updateUI();
    // Dispatch event for other components
    document.dispatchEvent(new CustomEvent('chatHistoryUpdated', { detail: { history } }));
  },

  load() {
    try { return JSON.parse(localStorage.getItem(this.STORAGE_KEY) || '[]'); } catch { return []; }
  },

  clear() {
    localStorage.removeItem(this.STORAGE_KEY);
    this.updateUI();
    document.dispatchEvent(new CustomEvent('chatHistoryUpdated', { detail: { history: [] } }));
  },

  updateUI() {
    const container = document.getElementById('chatHistoryList') || document.getElementById('historyContainer');
    if (!container) return;
    const history = this.load();
    if (!history.length) {
      container.innerHTML = `<div style="text-align:center;color:#6b7280;padding:32px 16px;font-size:13px">
        <div style="font-size:40px;margin-bottom:8px">💬</div>
        <div data-i18n="chat_history_empty">${LangManager.get('chat_history_empty')}</div>
      </div>`;
      return;
    }
    container.innerHTML = `<div class="history-list">` +
      history.slice(0, 20).map(item => {
        const time = new Date(item.timestamp).toLocaleString(LangManager.current === 'ar' ? 'ar-KW' : 'en-US', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' });
        const shortQ = item.question.slice(0, 55) + (item.question.length > 55 ? '...' : '');
        const shortA = item.answer.slice(0, 80) + (item.answer.length > 80 ? '...' : '');
        return `<div class="history-item" style="background:var(--card,#fff);border:1px solid var(--border,#e5e7eb);border-radius:10px;padding:12px;margin-bottom:8px;cursor:pointer;transition:all .15s"
            onmouseover="this.style.borderColor='#c9a84c'" onmouseout="this.style.borderColor=''"
            onclick="ChatHistory.openInPanel(${item.id})">
          <div style="font-weight:700;font-size:13px;color:var(--primary,#1e3a5f);margin-bottom:4px">🙋 ${_escHtml(shortQ)}</div>
          <div style="font-size:12px;color:var(--muted,#6b7280);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">🤖 ${_escHtml(shortA)}</div>
          <div style="font-size:11px;color:#9ca3af;margin-top:6px">${time}</div>
        </div>`;
      }).join('') +
      `<div style="text-align:center;margin-top:8px">
        <button onclick="ChatHistory.clear()" style="border:1px solid #e5e7eb;background:transparent;color:#6b7280;padding:6px 16px;border-radius:999px;cursor:pointer;font-family:inherit;font-size:12px"
          data-i18n="chat_history_clear">${LangManager.get('chat_history_clear')}</button>
      </div>` +
      `</div>`;
  },

  openInPanel(id) {
    const history = this.load();
    const item = history.find(h => h.id == id);
    if (!item) return;
    // Open right chat panel if available
    if (typeof openChatPanel === 'function') openChatPanel();
    // Pre-populate chat with this conversation
    document.dispatchEvent(new CustomEvent('restoreChat', { detail: item }));
  }
};

// Helper used by ChatHistory
function _escHtml(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Auto-init when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { LangManager.init(); ChatHistory.updateUI(); });
} else {
  LangManager.init();
  ChatHistory.updateUI();
}
