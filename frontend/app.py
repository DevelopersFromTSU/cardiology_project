import sys
import os
import requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Настройка путей и переменных окружения
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / ".env.txt"
load_dotenv(dotenv_path=env_path)

from pipeline.pipeline3_retrieve.retriever import rewrite_patient_query, hybrid_search

CURRENT_PROFILE = {
    "role_name": "кардиолог",
    "assistant_name": "кардио-ассистент",
    "clinic_name": "Кардиологический центр"
}

st.set_page_config(
    page_title="Кардио-Анамнез",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def call_gemini(messages_history, system_prompt, temperature=0.2):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ Ошибка: Не задан GOOGLE_API_KEY в .env"

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    contents = []
    for msg in messages_history:
        if "content" in msg and msg["content"]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    while contents and contents[0]["role"] == "model":
        contents.pop(0)

    if not contents:
        return "⚠️ Ошибка: нет сообщений от пользователя."

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": temperature}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=90)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Ошибка вызова Gemini: {e}"


# ==========================================
# 2. СТРУКТУРА ВОПРОСОВ (МАШИНА СОСТОЯНИЙ)
# ==========================================
QUESTIONS = [
    {
        "id": "q_intro",
        "title": "Паспортные данные",
        "text": "Пожалуйста, укажите ваше ФИО, возраст (а также рост и вес, если помните):",
        "type": "textarea"
    },
    {
        "id": "q_pain",
        "title": "Боли в груди",
        "text": "Беспокоят ли вас давящие, жгучие или сжимающие боли за грудиной?",
        "type": "radio",
        "options": ["Да", "Нет", "Не знаю / Не обращал внимания"]
    },
    {
        "id": "q_pain_details",
        "title": "Характер боли (уточнение)",
        "text": "Опишите боль подробнее: как она ощущается, что ее провоцирует (например, ходьба, стресс), сколько она обычно длится и чем вы ее снимаете?",
        "type": "textarea",
        "depends_on": ("q_pain", "Да")
    },
    {
        "id": "q_breath",
        "title": "Одышка и отеки",
        "text": "Бывает ли у вас одышка при обычной ходьбе или когда вы ложитесь горизонтально? Отекают ли у вас ноги к вечеру?",
        "type": "radio",
        "options": ["Да", "Нет", "Иногда"]
    },
    {
        "id": "q_breath_details",
        "title": "Одышка (уточнение)",
        "text": "Расскажите подробнее: в какие моменты тяжело дышать и насколько сильно отекают ноги?",
        "type": "textarea",
        "depends_on": ("q_breath", "Да")
    },
    {
        "id": "q_rhythm",
        "title": "Перебои в сердце",
        "text": "Бывают ли приступы внезапного сильного или неритмичного сердцебиения, ощущение «замирания» или «кувыркания» сердца в груди?",
        "type": "radio",
        "options": ["Да", "Нет"]
    },
    {
        "id": "q_faint",
        "title": "Головокружения",
        "text": "Случались ли у вас внезапные приступы резкой слабости, потемнения в глазах, сильного головокружения или потери сознания?",
        "type": "radio",
        "options": ["Да", "Нет"]
    },
    {
        "id": "q_pressure_work",
        "title": "Рабочее давление",
        "text": "Какое у вас обычно артериальное давление (ваше привычное «рабочее» давление)?",
        "type": "textarea"
    },
    {
        "id": "q_pressure_max",
        "title": "Максимальное давление",
        "text": "Какое самое высокое давление у вас когда-либо фиксировали (максимальные цифры)?",
        "type": "textarea"
    },
    {
        "id": "q_pulse",
        "title": "Пульс",
        "text": "Какой у вас обычно пульс в состоянии покоя (когда вы просто сидите или лежите)?",
        "type": "textarea"
    },
    {
        "id": "q_walk",
        "title": "Темп ходьбы",
        "text": "Какой у вас обычно темп ходьбы на улице?",
        "type": "radio",
        "options": ["Медленный", "Умеренный", "Быстрый", "Затрудняюсь ответить"]
    },
    {
        "id": "q_salt",
        "title": "Привычка к соли",
        "text": "Имеете ли вы привычку подсаливать готовую еду в тарелке, даже не пробуя ее?",
        "type": "radio",
        "options": ["Да", "Нет", "Иногда"]
    },
    {
        "id": "q_smoke",
        "title": "Курение",
        "text": "Вы курите?",
        "type": "radio",
        "options": ["Да", "Нет, бросил", "Никогда не курил"]
    },
    {
        "id": "q_smoke_details",
        "title": "Курение (уточнение)",
        "text": "Расскажите подробнее: сколько лет курите (или курили), сколько сигарет в день? Если бросили — как давно?",
        "type": "textarea",
        "depends_on": ("q_smoke", ["Да", "Нет, бросил"])
    },
    {
        "id": "q_alcohol",
        "title": "Алкоголь",
        "text": "Употребляете ли вы алкоголь?",
        "type": "radio",
        "options": ["Да", "Нет"]
    },
    {
        "id": "q_alcohol_details",
        "title": "Алкоголь (уточнение)",
        "text": "Какой именно алкоголь вы употребляете, как часто и в каких количествах (например: вино 2 раза в неделю по бокалу)?",
        "type": "textarea",
        "depends_on": ("q_alcohol", "Да")
    },
    {
        "id": "q_family",
        "title": "Семейный анамнез",
        "text": "Были ли у ваших кровных родственников (родители, братья, сестры) ранние инфаркты, инсульты или гипертония?",
        "type": "radio",
        "options": ["Да", "Нет", "Не знаю"]
    },
    {
        "id": "q_allergy",
        "title": "Аллергия",
        "text": "Есть ли у вас аллергия на какие-нибудь лекарства или пищевые продукты?",
        "type": "radio",
        "options": ["Да", "Нет", "Не знаю"]
    },
    {
        "id": "q_allergy_details",
        "title": "Аллергия (уточнение)",
        "text": "Напишите, на что именно у вас аллергия и как она обычно проявляется (например, сыпь, отек, зуд)?",
        "type": "textarea",
        "depends_on": ("q_allergy", "Да")
    },
    {
        "id": "q_meds",
        "title": "Прием лекарств",
        "text": "Принимаете ли вы какие-нибудь таблетки или добавки на постоянной основе (каждый день)?",
        "type": "radio",
        "options": ["Да", "Нет"]
    },
    {
        "id": "q_meds_details",
        "title": "Лекарства (уточнение)",
        "text": "Перечислите названия препаратов и их дозировки, если помните (например: Конкор 5 мг утром, Тромбо АСС 100 мг вечером).",
        "type": "textarea",
        "depends_on": ("q_meds", "Да")
    },
    {
        "id": "q_history",
        "title": "Диагнозы",
        "text": "Отметьте, если у вас когда-либо были официально диагностированы следующие заболевания (можно выбрать несколько):",
        "type": "multiselect",
        "options": ["Гипертония", "Ишемическая болезнь сердца (ИБС)", "Инфаркт", "Инсульт", "Сахарный диабет",
                    "Ничего из перечисленного"]
    },
    {
        "id": "q_surgery",
        "title": "Операции",
        "text": "Делали ли вам когда-нибудь операции на сердце или сосудах (например, установка стентов, кардиостимулятора, шунтирование)?",
        "type": "radio",
        "options": ["Да", "Нет"]
    }
]

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ
# ==========================================
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "stage" not in st.session_state:
    st.session_state.stage = "survey"
if "current_index" not in st.session_state:
    st.session_state.current_index = 0


def get_visible_questions():
    visible = []
    for q in QUESTIONS:
        if "depends_on" in q:
            dep_id, expected_val = q["depends_on"]
            user_answer = st.session_state.answers.get(dep_id, "")
            if isinstance(expected_val, list):
                if user_answer in expected_val:
                    visible.append(q)
            else:
                if user_answer == expected_val:
                    visible.append(q)
        else:
            visible.append(q)
    return visible


def clean_orphaned_answers():
    visible_ids = [q["id"] for q in get_visible_questions()]
    keys_to_delete = [k for k in st.session_state.answers if k not in visible_ids]
    for k in keys_to_delete:
        del st.session_state.answers[k]


# ==========================================
# 4. СТИЛИЗАЦИЯ (CSS)
# ==========================================
st.markdown("""
<style>
    /* Общий светлый фон приложения */
    .stApp { background-color: #F5F0E8 !important; color: #1F2937 !important; }

    /* Верхний баннер */
    .med-banner {
        background-color: #047857 !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(4, 120, 87, 0.15);
        width: 100%;
        box-sizing: border-box;
    }
    .med-banner h2 { color: #FFFFFF !important; margin: 0 0 8px 0 !important; font-size: 26px !important; font-weight: 800 !important; }
    .med-banner p { color: #D1FAE5 !important; margin: 0 !important; font-size: 15px !important; }

    /* --- СТИЛИ КАРТОЧКИ (ФОРМЫ STREAMLIT) --- */
    div[data-testid="stForm"] {
        background-color: #047857 !important; /* Глубокий зеленый фон */
        border-radius: 16px !important;
        padding: 35px 30px !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    }

    /* Заголовки и тексты внутри зеленой карточки */
    .question-title { color: #FFFFFF !important; font-size: 26px !important; font-weight: 800 !important; margin-bottom: 15px !important; }
    .question-text { font-size: 18px !important; color: #ECFDF5 !important; margin-bottom: 25px !important; line-height: 1.6 !important; }

    /* Делаем системные лейблы и текст радиокнопок белыми */
    div[data-testid="stForm"] label, 
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] .stRadio label p,
    div[data-testid="stForm"] .stRadio label span {
        color: #FFFFFF !important; 
        font-size: 16px !important;
    }

    /* --- ПОЛЯ ВВОДА (БЕЛЫЕ С ТЕМНЫМ ТЕКСТОМ ДЛЯ ВИДИМОСТИ) --- */
    div[data-testid="stForm"] div[data-baseweb="textarea"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 2px solid #6EE7B7 !important;
    }
    div[data-testid="stForm"] div[data-baseweb="textarea"] textarea {
        background-color: transparent !important;
        color: #1F2937 !important; /* Темный текст для контраста */
        font-size: 16px !important;
        padding: 10px !important;
        -webkit-text-fill-color: #1F2937 !important; 
    }
    div[data-testid="stForm"] div[data-baseweb="textarea"]:focus-within {
        box-shadow: 0 0 0 3px #6EE7B7 !important;
    }

    /* РАДИОКНОПКИ (Кружки) */
    div[data-testid="stForm"] .stRadio div[role="radiogroup"] label div[data-baseweb="radio"] div {
        background-color: #FFFFFF !important; /* белые кружки */
        border-color: #6EE7B7 !important;
    }

    /* ВЫПАДАЮЩИЙ СПИСОК MULTISELECT */
    div[data-testid="stForm"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    div[data-testid="stForm"] div[data-baseweb="select"] span {
        color: #1F2937 !important;
    }
    div[data-testid="stForm"] div[data-baseweb="select"] div[data-baseweb="tag"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    /* --- КНОПКИ НАЗАД / ДАЛЕЕ --- */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #10B981 !important; /* Яркий изумрудный цвет */
        color: #FFFFFF !important;
        border: 2px solid #6EE7B7 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #059669 !important; /* Более темный при наведении */
        border-color: #FFFFFF !important;
        transform: translateY(-2px);
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF !important;
        font-size: 16px !important;
        margin: 0 !important;
    }

    /* --- СВОДКА ОТВЕТОВ --- */
    .summary-item { 
        margin-bottom: 15px; 
        padding: 15px 20px; 
        background-color: #FFFFFF;
        border-left: 5px solid #047857; 
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .summary-question { font-weight: 700 !important; color: #047857 !important; font-size: 15px !important; margin-bottom: 5px !important; }
    .summary-answer { font-size: 17px !important; color: #1F2937 !important; margin-top: 5px !important; }
</style>
""", unsafe_allow_html=True)

# Шапка кардиоцентра
st.markdown(
    f"""
    <div class="med-banner">
        <h2>🫀 {CURRENT_PROFILE['clinic_name']}</h2>
        <p>Электронный модуль предварительного сбора кардиологического анамнеза</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 5. БОКОВАЯ ПАНЕЛЬ (ОТСЛЕЖИВАНИЕ ПРОГРЕССА)
# ==========================================
visible_qs = get_visible_questions()

with st.sidebar:
    st.markdown("### Прогресс заполнения")
    for idx, q in enumerate(visible_qs):
        if idx < st.session_state.current_index and q["id"] in st.session_state.answers:
            icon = "🟢"
        elif idx == st.session_state.current_index and st.session_state.stage == "survey":
            icon = "🟠"
        else:
            icon = "⚪"
        st.markdown(f"{icon} {q['title']}")

# ==========================================
# 6. ОСНОВНАЯ ЛОГИКА ОТОБРАЖЕНИЯ ЭКРАНОВ
# ==========================================

# --- РЕЖИМ 1: ОПРОС ---
if st.session_state.stage == "survey":
    if st.session_state.current_index >= len(visible_qs):
        st.session_state.stage = "summary"
        st.rerun()

    current_q = visible_qs[st.session_state.current_index]

    with st.form(key=f"form_{current_q['id']}", clear_on_submit=False):
        st.markdown(f'<div class="question-title">{current_q["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="question-text">{current_q["text"]}</div>', unsafe_allow_html=True)

        answer_val = st.session_state.answers.get(current_q["id"], None)

        if current_q["type"] == "textarea":
            user_input = st.text_area("Ваш ответ:", value=answer_val if answer_val else "",
                                      label_visibility="collapsed")
        elif current_q["type"] == "radio":
            idx = current_q["options"].index(answer_val) if answer_val in current_q["options"] else 0
            user_input = st.radio("Выберите один вариант:", current_q["options"], index=idx,
                                  label_visibility="collapsed")
        elif current_q["type"] == "multiselect":
            user_input = st.multiselect("Выберите один или несколько вариантов:", current_q["options"],
                                        default=answer_val if answer_val else [], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            back_submitted = st.form_submit_button("⬅️ Назад") if st.session_state.current_index > 0 else False
        with col2:
            next_submitted = st.form_submit_button("Далее ➡️")

        if back_submitted:
            st.session_state.current_index -= 1
            st.rerun()

        if next_submitted:
            if not user_input or (isinstance(user_input, list) and len(user_input) == 0):
                st.warning("Пожалуйста, заполните поле перед переходом к следующему вопросу.")
            else:
                st.session_state.answers[current_q["id"]] = user_input
                clean_orphaned_answers()
                st.session_state.current_index += 1
                st.rerun()

# --- РЕЖИМ 2: СВОДКА ПЕРЕД ОТПРАВКОЙ ---
elif st.session_state.stage == "summary":
    st.markdown("## Проверьте ваши ответы")
    st.info(
        "Это краткая сводка собранной информации. Если вы хотите что-то исправить, нажмите кнопку 'Изменить' рядом с нужным пунктом.")

    visible_qs = get_visible_questions()

    for idx, q in enumerate(visible_qs):
        answer = st.session_state.answers.get(q["id"], "Нет ответа")
        if isinstance(answer, list):
            answer = ", ".join(answer)

        col_text, col_btn = st.columns([4, 1])
        with col_text:
            st.markdown(f"""
            <div class="summary-item">
                <div class="summary-question">{q['text']}</div>
                <div class="summary-answer">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("✏️ Изменить", key=f"edit_{q['id']}"):
                st.session_state.current_index = idx
                st.session_state.stage = "survey"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Всё верно, сформировать медицинскую карту", type="primary", use_container_width=True):
        st.session_state.stage = "analysis"
        st.rerun()

# --- РЕЖИМ 3: АНАЛИЗ (НЕЙРОСЕТЬ + RAG) ---
elif st.session_state.stage == "analysis":
    st.markdown("## ⚙️ Формирование медицинского отчета...")

    # Собираем текстовую строку из ответов для нейросети
    compiled_history = ""
    for q in get_visible_questions():
        ans = st.session_state.answers.get(q["id"], "")
        if isinstance(ans, list): ans = ", ".join(ans)
        compiled_history += f"ВОПРОС: {q['text']}\nОТВЕТ: {ans}\n\n"

    # 1. Структурирование фактов
    with st.spinner("⏳ Структурирование данных анамнеза..."):
        EXTRACTION_PROMPT = """
        Ты — медицинский дата-аналитик. Извлеки из стенограммы опроса только достоверные факты и сгруппируй строго по разделам:

        1. ПАСПОРТНЫЕ И ФИЗИКАЛЬНЫЕ ДАННЫЕ: ФИО, возраст, рост, вес (рассчитай точный ИМТ только при наличии точных роста и веса).
        2. ВЕДУЩИЕ ЖАЛОБЫ: локализация, характер, связь с нагрузкой, длительность болей в груди; одышка/ортопноэ; отеки ног; приступы сердцебиения/перебоев; головокружения/синкопе.
        3. ГЕМОДИНАМИКА: рабочее АД, максимальный пик АД, пульс в покое и при нагрузке.
        4. ФАКТОРЫ РИСКА: табакокурение (стаж, шт/день или точный срок отказа), алкоголь, питание (овощи/фрукты, досаливание), двигательная активность.
        5. АЛЛЕРГИИ И ПОСТОЯННЫЕ ЛЕКАРСТВА: конкретные препараты, дозировки и аллергические реакции.
        6. КАРДИО- И СОМАТИЧЕСКИЙ АНАМНЕЗ: подтвержденные диагнозы, сосудистые катастрофы, диабет, операции.
        7. СЕМЕЙНЫЙ АНАМНЕЗ: ранние ССЗ у родственников первой линии.
        8. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ: любые другие факты, не вошедшие в основные категории.

        КРИТИЧЕСКИЕ ПРАВИЛА ДОСТОВЕРНОСТИ:
        - СТРОГО РАЗЛИЧАЙ ОТРИЦАНИЕ И ОТСУТСТВИЕ ДАННЫХ. Если пациент прямо ответил «нет» на вопрос о наличии симптома или заболевания (например, «нет аллергий», «нет у родственников инфарктов») — пиши «Отрицает». Если пациент ответил «не знаю», «не помню», «не измерял» или вопрос был пропущен — пиши строго «Не исследовано / данных нет».
        - Внимательно извлекай ВСЕ количественные данные, названные пациентом (например, шаги, граммы алкоголя, частоту употребления, годы стажа курения). Ничего не сокращай и не упускай.
        - Запрещено додумывать: фиксируй исключительно слова пациента. Не ставь знаки приблизительности («~»), если пациент назвал точное число.
        """
        facts = call_gemini([{"role": "user", "content": compiled_history}], EXTRACTION_PROMPT, 0.1)

    # 2. Поиск в Qdrant
    with st.spinner("🔍 Поиск по клиническим рекомендациям в Qdrant..."):
        search_query = rewrite_patient_query(facts)
        try:
            retrieved_chunks = hybrid_search(search_query=search_query, top_k=20)
        except Exception:
            retrieved_chunks = []

        if retrieved_chunks:
            rag_context = "\n\n".join(
                [
                    f"--- КЛИНИЧЕСКИЙ ПРОТОКОЛ: {ch.get('book', 'Клинические рекомендации')} (Стр. {ch.get('page', 'Не указана')}) ---\n{ch.get('text', '')}"
                    for ch in retrieved_chunks]
            )
        else:
            rag_context = "Клинические протоколы Минздрава РФ по кардиологии."

    # 3. ЭМК
    with st.spinner("📝 Формирование записи в электронную медкарту (ЭМК)..."):
        EMR_PROMPT = """
        Ты — медицинский регистратор-информатик. На основе данных опроса сформируй стандартизованный протокол предварительного сбора анамнеза для Электронной медицинской карты (ЭМК):

        ### 📋 ПРЕДВАРИТЕЛЬНЫЙ АНАМНЕЗ (ДОВРАЧЕБНЫЙ ОПРОС)
        * **Жалобы**: характер ощущений в грудной клетке, одышка, перебои в работе сердца, отеки, колебания давления, общая слабость (при отсутствии жалоб указать: «Активных жалоб не предъявляет»).
        * **Гемодинамические показатели со слов пациента**: привычные и максимальные цифры АД, привычный пульс.
        * **Анамнез сердечно-сосудистых заболеваний**: ранее диагностированные болезни сердца, сосудистые кризы, инфаркты, аритмии, перенесенные вмешательства.
        * **Факторы сердечно-сосудистого риска и образ жизни**:
          - Табакокурение и употребление алкоголя (указать точные объемы и частоту, если названы);
          - Характер питания и двигательная активность (шаги, часы работы).
        * **Сопутствующие патологии**: эндокринные нарушения (диабет), болезни почек, бронхолегочные заболевания.
        * **Аллергологический анамнез**: непереносимость медикаментов или пищевых продуктов.
        * **Семейный анамнез**: ранние сосудистые катастрофы у родственников 1-й линии.
        * **Дополнительные примечания**: факты, переданные пациентом в свободном поле.

        ---
        ### ℹ️ ПАМЯТКА ДЛЯ ПАЦИЕНТА
        * Ваши ответы зафиксированы и переданы в медицинскую карту для ознакомления врачом перед приемом.
        * **Рекомендации к очной консультации**:
          1. Возьмите с собой дневник самоконтроля давления и пульса (если проводили измерения).
          2. Подготовьте точные названия и дозировки всех препаратов, которые принимаете постоянно или курсами.
          3. Возьмите имеющиеся пленки ЭКГ, выписки из стационаров, результаты анализов крови и УЗИ сердца (ЭхоКГ).
        """
        emr_res = call_gemini([{"role": "user", "content": f"ФАКТЫ:\n{facts}"}], EMR_PROMPT, 0.2)

    # 4. Отчет врача
    with st.spinner("🩺 Составление клинического аналитического заключения..."):
        DOCTOR_PROMPT = """
        Ты — клинический эксперт-кардиолог. Составь для очного приема врача предельно компактный аналитический бриф на основе анамнеза пациента и предоставленных клинических протоколов РКО.
        Объединяй клиническую гипотезу и тактику с прямой доказательной базой из клинических рекомендаций (RAG) по принципу: тезис -> обоснование фактом пациента -> опора на протокол.

        ФОРМАТ ВЫВОДА (СТРОГО 3 РАЗДЕЛА):

        ### 1. 📌 ПАСПОРТ АНАМНЕЗА (ФАКТЫ И ЦИФРЫ)
        * **Пациент**: ФИО, возраст, рост/вес (рассчитай ИМТ и укажи категорию массы тела без тильд «~»).
        * **Симптомы**: характер болей в груди (связь с нагрузкой), одышка/ортопноэ, отеки ног, перебои ритма, головокружения/синкопе (если жалоб нет — «Активных жалоб не предъявляет»).
        * **Гемодинамика**: рабочее АД, максимальный пик АД, пульс в покое и при нагрузке.
        * **Факторы риска и анамнез**: курение (стаж/срок отказа), алкоголь (точные дозы и частота), инфаркты/инсульты, диабет, сопутствующие болезни, семейная отягощенность, аллергии.
        * **Базовая терапия**: постоянно принимаемые препараты и дозы (если отрицает прием — «Базовую медикаментозную терапию не получает»).

        ### 2. 🩺 КЛИНИЧЕСКАЯ ГИПОТЕЗА И ТАКТИКА (RAG-КОНСИЛИУМ)
        Сформулируй ключевые клинические тезисы под доминирующий синдром пациента.
        - Для пункта 1 (диагноз) ссылки на литературу не требуются (вывод по фактам пациента).
        - Для пунктов 2, 3 и 4 опора на RAG строго обязательна.

        ПРАВИЛА ДОКАЗАТЕЛЬНОЙ БАЗЫ И ФОРМАТИРОВАНИЯ:
        1. ОБОСНОВАНИЕ ТЕЗИСА: Обосновывай каждый тезис опорой на RAG-контекст. Если возможно — используй точную цитату. Если информация в RAG представлена в виде таблицы, сложного алгоритма или фрагментов текста, сделай точный смысловой синтез (summary) с обязательным указанием источника.
        2. ГИБКОСТЬ ШКАЛ: Применяй те клинические шкалы и алгоритмы стратификации риска, которые явно присутствуют в предоставленном RAG-контексте для данного состояния (например, не только SCORE-2, но и HAS-BLED, ARC-HBR и др., если они есть в извлеченных чанках).
        3. ОФОРМЛЕНИЕ ССЫЛОК: Оформляй каждую ссылку исключительно через маркированный список «- >» на новой строке. Запрещено писать две ссылки подряд в одну строку!

        Структура вывода:
        1. **Главный диагноз или ведущий синдром**: точный клинический вывод по профилю пациента с указанием функционального класса или степени тяжести (без цитат).
        2. **Уровень риска**: категория риска по профильной шкале.
        - > [{Название руководства}, Стр. X]: «Точная цитата или смысловой синтез критерия стратификации риска...»
        3. **План лечения и цели**: стартовая стратегия (препараты/комбинации/интервенция/хирургия) и ключевые клинические цели.
        - > [{Название руководства}, Стр. X]: «Точная цитата или смысловой синтез схемы терапии и целевых ориентиров...»
        4. **Чего делать нельзя (Ограничения)**: абсолютные противопоказания, запрещенные комбинации или опасные взаимодействия под профиль пациента.
        - > [{Название руководства}, Стр. X]: «Точная цитата или смысловой синтез прямого запрета/противопоказания...» (Если в извлеченных данных нет прямых ограничений, напиши: «Прямых ограничений в предоставленном контексте не найдено»).

        ### 3. 🔍 ФОКУС ОЧНОГО ПРИЕМА (СЛЕПЫЕ ЗОНЫ И ТОЧЕЧНЫЙ СКРИНИНГ)
        * **⚠️ Слепые зоны опроса**: четко перечисли всё, о чем пациент не знает, не помнил, не измерял или ответил неполно (неизвестный пульс, скрытый алкогольный фактор, неисследованные отеки, пропущенные лабораторные данные). Строго не выдумывай "слепые зоны", если пациент дал конкретный ответ (например, если пациент отрицает наличие инфарктов у семьи - не пиши, что семейный анамнез не исследован).

        ЖЕСТКИЕ ПРАВИЛА И ОГРАНИЧЕНИЯ:
        - Никаких произвольных разделов: строго запрещено создавать пункты, не указанные в шаблоне.
        - Лимит ссылок: от 1 до 3 наиболее сильных подкреплений из RAG на каждый пункт Раздела №2.
        - Запрещено выдумывать цитаты, страницы и клинические шкалы: брать строго из переданного блока КЛИНИЧЕСКИЕ ПРОТОКОЛЫ.
        """
        doctor_res = call_gemini(
            [{"role": "user", "content": f"ФАКТЫ ПАЦИЕНТА:\n{facts}\n\nКЛИНИЧЕСКИЕ ПРОТОКОЛЫ (RAG):\n{rag_context}"}],
            DOCTOR_PROMPT, 0.2
        )

    # Вывод результатов в табах
    st.success("✅ Карта и анализ успешно сформированы!")
    tab1, tab2 = st.tabs(["📋 Выписка для ЭМК", "🩺 Аналитический отчет врача"])
    with tab1:
        st.markdown(emr_res)
    with tab2:
        st.markdown(doctor_res)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Начать новый опрос пациента", use_container_width=True):
        st.session_state.clear()
        st.rerun()