import sys
import os
import requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 1. Настройка путей
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# 2. Переменные окружения
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / ".env.txt"
load_dotenv(dotenv_path=env_path)

from pipeline.pipeline3_retrieve.retriever import rewrite_patient_query, hybrid_search

CURRENT_PROFILE = {
    "role_name": "кардиолог",
    "assistant_name": "кардио-ассистент",
    "clinic_name": "Кардиологический центр",
    "welcome_message": (
        "Здравствуйте! Я интеллектуальный ассистент кардиологического центра. "
        "Перед приемом врача я помогу собрать точные данные для вашей амбулаторной карты.\n\n"
        "**Какое у вас обычно артериальное давление (привычное рабочее и максимальное)?**"
    )
}

st.set_page_config(
    page_title="Кардио-Анамнез",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# СТИЛИЗАЦИЯ: БЕЖЕВЫЙ ФОН, ЗЕЛЕНЫЕ БЛОКИ, БЕЛЫЙ ШРИФТ
# =====================================================================
st.markdown(
    """
    <style>
        /* 1. Основной фон: бежевый */
        .stApp {
            background-color: #F5F0E8 !important;
            color: #1F2937 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .main .block-container {
            max-width: 860px !important;
            padding-top: 2rem !important;
            padding-bottom: 3.5rem !important;
            margin: 0 auto !important;
        }

        footer, header, div[data-testid="stBottom"] {
            display: none !important;
        }

        /* 2. Верхний баннер кардиоцентра */
        .med-banner {
            background-color: #047857 !important;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 22px;
            box-shadow: 0 4px 14px rgba(4, 120, 87, 0.15);
            width: 100%;
            box-sizing: border-box;
        }
        .med-banner h2 {
            color: #FFFFFF !important;
            margin: 0 0 4px 0 !important;
            font-size: 22px !important;
            font-weight: 700 !important;
        }
        .med-banner p {
            color: #E6F4EA !important;
            margin: 0 !important;
            font-size: 14px !important;
        }

        /* 3. Карточки диалога */
        .stChatMessage {
            background-color: #FFFFFF !important;
            border: 1.5px solid #047857 !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
            width: 100% !important;
            box-sizing: border-box;
        }
        .stChatMessage p, .stChatMessage span, .stChatMessage div, .stChatMessage strong {
            color: #064E3B !important;
            font-size: 15px !important;
            line-height: 1.55 !important;
        }

        /* 4. АВАТАРЫ: ЗЕЛЕНЫЙ ФОН И БЕЛЫЕ ЗНАЧКИ */
        div[data-testid="stChatMessageAvatar"],
        div[data-testid="stChatMessageAvatar"] > div,
        div[data-testid="stChatMessageAvatar"] * {
            background-color: #047857 !important;
            background: #047857 !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            color: #FFFFFF !important;
        }

        /* 5. Кнопки быстрых ответов */
        div[data-testid="column"] button,
        div[data-testid="stColumn"] button,
        .stButton button {
            width: 100% !important;
            background-color: #047857 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 4px rgba(4, 120, 87, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="column"] button p,
        div[data-testid="column"] button span,
        div[data-testid="stColumn"] button p,
        div[data-testid="stColumn"] button span,
        .stButton button p,
        .stButton button span {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        div[data-testid="column"] button:hover:not(:disabled),
        div[data-testid="stColumn"] button:hover:not(:disabled),
        .stButton button:hover:not(:disabled) {
            background-color: #065F46 !important;
            transform: translateY(-1px);
        }
        /* Стиль для заблокированных кнопок */
        div[data-testid="column"] button:disabled,
        div[data-testid="stColumn"] button:disabled,
        .stButton button:disabled {
            background-color: #6EE7B7 !important;
            opacity: 0.6 !important;
            cursor: not-allowed !important;
        }

        /* 6. Поле ввода ответа и кнопка отправки */
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #064E3B !important;
            border: 1.5px solid #047857 !important;
            border-radius: 8px !important;
            font-size: 15px !important;
            padding: 12px 14px !important;
            box-shadow: none !important;
        }
        .stTextInput input::placeholder {
            color: #059669 !important;
            opacity: 0.75 !important;
        }
        .stTextInput input:focus {
            border-color: #065F46 !important;
            box-shadow: 0 0 0 2px rgba(4, 120, 87, 0.2) !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: auto !important;
            min-width: 180px !important;
            background-color: #047857 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            box-shadow: 0 2px 5px rgba(4, 120, 87, 0.2) !important;
        }
        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        div[data-testid="stFormSubmitButton"] button:disabled {
            background-color: #6EE7B7 !important;
            opacity: 0.6 !important;
            cursor: not-allowed !important;
        }

        /* 7. ЗЕЛЕНЫЙ БЛОК ДОПОЛНИТЕЛЬНОЙ ИНФОРМАЦИИ СО СВЕТЛЫМИ БУКВАМИ */
        .extra-card {
            background-color: #047857 !important;
            border-radius: 12px !important;
            padding: 20px 24px !important;
            margin-top: 18px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 4px 14px rgba(4, 120, 87, 0.15) !important;
        }
        .extra-card h4 {
            color: #FFFFFF !important;
            margin: 0 0 8px 0 !important;
            font-size: 17px !important;
            font-weight: 700 !important;
        }
        .extra-card p {
            color: #E6F4EA !important;
            margin: 0 !important;
            font-size: 14px !important;
            line-height: 1.45 !important;
        }

        /* Текстовое поле внутри дополнительного блока */
        div[data-testid="stTextArea"] div[data-baseweb="textarea"],
        div[data-testid="stTextArea"] textarea {
            background-color: #065F46 !important;
            color: #FFFFFF !important;
            border: 1.5px solid #34D399 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            font-size: 14.5px !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #A7F3D0 !important;
            opacity: 0.8 !important;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #6EE7B7 !important;
            box-shadow: 0 0 0 2px rgba(110, 231, 183, 0.25) !important;
        }
        div[data-testid="stTextArea"] div {
            color: #E6F4EA !important;
        }

        .counter-badge {
            font-size: 13px;
            font-weight: 700;
            color: #047857;
            text-align: right;
            margin-top: 6px;
        }

        /* 8. ЧИСТОЕ АНИМИРОВАННОЕ ТРОЕТОЧИЕ */
        .typing-dots {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 5px;
            color: #047857;
            line-height: 1;
            display: inline-block;
            animation: dotsBlink 1.2s infinite ease-in-out;
            padding: 4px 0;
        }
        @keyframes dotsBlink {
            0%, 100% { opacity: 0.25; }
            50% { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Состояния сессии
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": CURRENT_PROFILE["welcome_message"]}
    ]

if "interview_completed" not in st.session_state:
    st.session_state.interview_completed = False

if "is_waiting_for_assistant" not in st.session_state:
    st.session_state.is_waiting_for_assistant = False

if "stage" not in st.session_state:
    st.session_state.stage = "interview"

if "rag_results" not in st.session_state:
    st.session_state.rag_results = None

if "extra_info_text" not in st.session_state:
    st.session_state.extra_info_text = ""

INTERVIEW_SYSTEM_INSTRUCTION = f"""
Ты — профессиональный медицинский робот-интервьюер ({CURRENT_PROFILE['assistant_name']}) кардиологического центра, 
который переходит сразу к делу без здравствуйте. Твоя задача — провести предварительный доврачебный опрос пациента по 
5 последовательным тематическим блокам:

1. ГЕМОДИНАМИКА: Рабочее и максимальное АД, привычный пульс в покое и при нагрузке.
2. ВРЕДНЫЕ ПРИВЫЧКИ: Курение (статус, стаж, количество; если бросил — сколько лет назад) и алкоголь (частота, тип напитка, объем).
3. ОБРАЗ ЖИЗНИ И АЛЛЕРГИЯ: Физическая активность (минут ходьбы в день), диета (досаливание, овощи/фрукты, сахар) и лекарственная/пищевая аллергия.
4. ЛИЧНЫЙ АНАМНЕЗ ССЗ: Повышение давления, ИБС, аритмия, сахарный диабет, перенесенные инфаркты, инсульты, операции на сердце (с возрастом или годом).
5. СЕМЕЙНЫЙ АНАМНЕЗ: Ранние инфаркты/инсульты у мужчин рода до 55 лет, женщин до 65 лет.

ПРАВИЛА ВЕДЕНИЯ ДИАЛОГА:
- Задавай вопросы БЛОКАМИ (по 1 блоку за реплику), объединяя близкие по смыслу темы. Не дроби опрос на десятки мелких фраз.
- ЕСЛИ ПАЦИЕНТ ОТВЕТИЛ КРАТКО («Да» / «Нет»): Уточни детали текущего блока перед переходом к следующему (например: «Уточните, пожалуйста, сколько сигарет в день выкуриваете?»).
- ЕСЛИ ПАЦИЕНТ ОТВЕЧАЕТ «НЕ ЗНАЮ», «НЕ ПОМНЮ», «НЕ ИЗМЕРЯЛ»: Не настаивай. Зафиксируй: «Принято, отметил» и сразу переходи к следующему блоку.
- КОГДА ВСЕ 5 БЛОКОВ ПРОЙДЕНЫ: Поблагодари пациента за ответы и в самый конец сообщения обязательно добавь метку: [ОПРОС_ЗАВЕРШЕН].
- Строго 1 сообщение за ход.
"""


def call_gemini(messages_history, system_prompt, temperature=0.2):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ Ошибка: Не задан GOOGLE_API_KEY в .env"

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
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


# 1. Шапка кардиоцентра
st.markdown(
    f"""
    <div class="med-banner">
        <h2>🫀 {CURRENT_PROFILE['clinic_name']}</h2>
        <p>Электронный модуль предварительного сбора кардиологического анамнеза</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Отрисовка истории сообщений
for msg in st.session_state.messages:
    clean_text = msg.get("content", "").replace("[ОПРОС_ЗАВЕРШЕН]", "").strip()
    with st.chat_message(msg["role"]):
        if "emr_content" in msg and "doctor_content" in msg:
            tab1, tab2 = st.tabs(["📋 Выписка для ЭМК", "🩺 Аналитический отчет врача"])
            with tab1:
                st.markdown(msg["emr_content"])
            with tab2:
                st.markdown(msg["doctor_content"])
        elif clean_text:
            st.markdown(clean_text)

# 3. Блок индикатора ожидания: строго НАД элементами ввода
if st.session_state.is_waiting_for_assistant:
    with st.chat_message("assistant"):
        st.markdown('<div class="typing-dots">. . .</div>', unsafe_allow_html=True)

# 4. Блок активного ввода (кнопки и инпут)
if st.session_state.stage == "interview" and not st.session_state.interview_completed:
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Блокировка контролов при ожидании ответа модели
    is_disabled = st.session_state.is_waiting_for_assistant

    # Кнопки быстрых ответов
    b1, b2, b3, b4 = st.columns(4)
    quick_choice = None
    if b1.button("✅ Да", disabled=is_disabled, use_container_width=True):
        quick_choice = "Да"
    if b2.button("❌ Нет", disabled=is_disabled, use_container_width=True):
        quick_choice = "Нет"
    if b3.button("🤷‍♂️ Не помню", disabled=is_disabled, use_container_width=True):
        quick_choice = "Не знаю, не помню"
    if b4.button("📊 Не измерял", disabled=is_disabled, use_container_width=True):
        quick_choice = "Никогда не измерял, данных нет"

    # Текстовое поле ввода ответа
    with st.form(key="inline_reply_form", clear_on_submit=True):
        user_reply_text = st.text_input(
            "Ваш ответ:",
            placeholder="Введите ваш ответ и нажмите Отправить...",
            disabled=is_disabled,
            label_visibility="collapsed"
        )
        submit_btn = st.form_submit_button("Отправить ответ 💬", disabled=is_disabled)

    # Проверка нового действия от пользователя
    final_reply = quick_choice or (user_reply_text.strip() if submit_btn else None)

    if final_reply and not st.session_state.is_waiting_for_assistant:
        st.session_state.messages.append({"role": "user", "content": final_reply})
        st.session_state.is_waiting_for_assistant = True
        st.rerun()

    # Если включен режим ожидания — выполняем запрос к модели
    if st.session_state.is_waiting_for_assistant:
        ai_text = call_gemini(
            messages_history=st.session_state.messages,
            system_prompt=INTERVIEW_SYSTEM_INSTRUCTION,
            temperature=0.2
        )

        if "[ОПРОС_ЗАВЕРШЕН]" in ai_text:
            st.session_state.interview_completed = True

        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.session_state.is_waiting_for_assistant = False
        st.rerun()

# 5. Зеленый блок финала опроса (до 200 символов)
if st.session_state.interview_completed and st.session_state.stage == "interview":
    st.markdown(
        """
        <div class="extra-card">
            <h4>✍️ Дополнительная информация для врача</h4>
            <p>
                Если вы хотите дополнить диалог сведениями, о которых забыли упомянуть или считаете важными, напишите их ниже (не более 200 символов):
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    extra_note = st.text_area(
        label="Поле ввода дополнительных сведений",
        max_chars=200,
        height=85,
        placeholder="Например: принимаю омега-3, к вечеру бывает отечность щиколоток...",
        key="extra_note_input",
        label_visibility="collapsed"
    )

    current_length = len(extra_note)
    st.markdown(
        f"<div class='counter-badge'>{current_length} / 200 символов</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    if st.button("🚀 Сформировать медицинскую карту и клинический анализ", type="primary", use_container_width=True):
        st.session_state.extra_info_text = extra_note.strip()
        st.session_state.stage = "analysis"
        st.rerun()

# 6. Стадия формирования ЭМК и RAG-анализа
# 6. Стадия формирования ЭМК и RAG-анализа
if st.session_state.stage == "analysis" and st.session_state.rag_results is None:
    with st.chat_message("assistant"):
        raw_history = "\n".join(
            [f"{m['role'].upper()}: {m.get('content', '').replace('[ОПРОС_ЗАВЕРШЕН]', '')}"
             for m in st.session_state.messages if "content" in m]
        )
        if st.session_state.extra_info_text:
            raw_history += f"\nДОПОЛНИТЕЛЬНО ОТ ПАЦИЕНТА: {st.session_state.extra_info_text}"

        # 1. Структурирование объективных фактов
        with st.spinner("⏳ Структурирование данных анамнеза..."):
            EXTRACTION_PROMPT = """
            Ты — медицинский дата-аналитик. Извлеки из стенограммы опроса пациента объективные клинические данные и структурируй их по разделам:

            1. ВЕДУЩИЕ ЖАЛОБЫ И СИМПТОМЫ:
               - Болевой синдром (характер, локализация, связь с нагрузкой, длительность, чем купируется);
               - Нарушения ритма / сердцебиение (приступы, перебои, внезапность начала/конца);
               - Одышка, отеки, утомляемость, непереносимость горизонтального положения (ортопноэ);
               - Головокружения, потемнение в глазах, синкопальные (обморочные) состояния.
            2. ГЕМОДИНАМИЧЕСКИЙ ПРОФИЛЬ:
               - Рабочее и максимальное АД;
               - Пульс в покое и при нагрузке/приступах.
            3. МОДИФИЦИРУЕМЫЕ ФАКТОРЫ РИСКА:
               - Статус курения (стаж, количество сигарет в день; если бросил — когда);
               - Алкоголь (частота, объем, напиток);
               - Диетические привычки (соль, животные жиры, простые углеводы/сахар, овощи/фрукты);
               - Уровень повседневной физической активности (минут ходьбы/аэробной нагрузки в день).
            4. АЛЛЕРГОЛОГИЧЕСКИЙ СТАТУС:
               - Реакции на лекарственные препараты и пищевые продукты.
            5. ЛИЧНЫЙ СЕРДЕЧНО-СОСУДИСТЫЙ И СОМАТИЧЕСКИЙ АНАМНЕЗ:
               - Диагностированные ранее ССЗ (АГ, ИБС, инфаркт, инсульт, пороки, сердечная недостаточность, аритмии);
               - Сопутствующие патологии (сахарный диабет, болезни почек, ХОБЛ/астма, щитовидная железа);
               - Перенесенные операции на сердце/сосудах (стентирование, шунтирование, РЧА, ЭКС).
            6. СЕМЕЙНЫЙ АНАМНЕЗ:
               - Ранние сосудистые катастрофы (инфаркт, инсульт, внезапная смерть) у кровных родственников (мужчины до 55 лет, женщины до 65 лет).
            7. ДОПОЛНИТЕЛЬНЫЕ СВЕДЕНИЯ:
               - Постоянно принимаемые медикаменты или добавки со слов пациента.

            Если сведений по какому-либо пункту нет или пациент ответил «не знаю / не помню / не измерял» — пиши: «Не исследовано / данных нет». Не додумывай факты.
            """
            facts = call_gemini([{"role": "user", "content": raw_history}], EXTRACTION_PROMPT, 0.1)

            # Тумблер отладки: True — показывать чанки, False — скрыть для продакшна
            DEBUG_RAG = True

            # 2. Поиск по клиническим рекомендациям в Qdrant
            with st.spinner("🔍 Поиск по клиническим рекомендациям в Qdrant..."):
                search_query = rewrite_patient_query(facts)
                try:
                    retrieved_chunks = hybrid_search(search_query=search_query, top_k=8)
                except Exception as e:
                    retrieved_chunks = []

                if retrieved_chunks:
                    rag_context = "\n\n".join(
                        [f"--- КЛИНИЧЕСКИЙ ПРОТОКОЛ (Стр. {ch.get('page', 'Не указана')}) ---\n{ch.get('text', '')}"
                         for ch in retrieved_chunks]
                    )
                else:
                    rag_context = "Клинические протоколы Минздрава РФ по кардиологии."

            # === [БЛОК ОТЛАДКИ ДЛЯ РАЗРАБОТЧИКА] ===
            if DEBUG_RAG and retrieved_chunks:
                with st.expander("🛠️ Отладка RAG: Поисковый запрос и извлеченные чанки"):
                    st.markdown(f"**Сформированный поисковый запрос:** `{search_query}`")
                    st.divider()
                    for idx, ch in enumerate(retrieved_chunks, 1):
                        st.markdown(
                            f"**Чанк #{idx}** | 📄 **Стр. {ch.get('page')}** | 🎯 **Релевантность: {ch.get('score')}%**\n\n"
                            f"```text\n{ch.get('text')[:350]}...\n```"
                        )
            # =======================================

        # 3. Формирование официальной выписки в ЭМК (восстановленный блок)
        with st.spinner("📝 Формирование записи в электронную медкарту (ЭМК)..."):
            EMR_PROMPT = """
            Ты — медицинский регистратор-информатик. На основе данных опроса сформируй стандартизованный протокол предварительного сбора анамнеза для Электронной медицинской карты (ЭМК):

            ### 📋 ПРЕДВАРИТЕЛЬНЫЙ АНАМНЕЗ (ДОВРАЧЕБНЫЙ ОПРОС)
            * **Жалобы**: характер ощущений в грудной клетке, одышка, перебои в работе сердца, отеки, колебания давления, общая слабость (при отсутствии жалоб указать: «Активных жалоб не предъявляет»).
            * **Гемодинамические показатели со слов пациента**: привычные и максимальные цифры АД, привычный пульс.
            * **Анамнез сердечно-сосудистых заболеваний**: ранее диагностированные болезни сердца, сосудистые кризы, инфаркты, аритмии, перенесенные вмешательства.
            * **Факторы сердечно-сосудистого риска и образ жизни**:
              - Табакокурение и употребление алкоголя;
              - Характер питания и двигательная активность.
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

        # 4. Формирование клинического отчета врача с бесшовными ссылками
            with st.spinner("🩺 Составление клинического аналитического заключения..."):
                DOCTOR_PROMPT = f"""
                Ты — кардиолог-консультант экспертного центра.
                Твоя задача — составить для лечащего врача структурированный клинический аналитический бриф перед очным приемом на основе фактов опроса пациента и предоставленных контекстов из клинических рекомендаций (RAG).

                СТРУКТУРА КЛИНИЧЕСКОГО БРИФА:

                ### 1. 📌 ЭКСПРЕСС-ПАСПОРТ АНАМНЕЗА
                * **Главный синдром и ключевые цифры**: ведущая жалоба пациента + гемодинамика (АД, ЧСС/пульс, ритмичность).
                * **Фоновые риски и отягощенность**: статус курения, метаболические маркеры (СД, вес, диета), коморбидность, ранний семейный кардиоваскулярный анамнез.

                ### 2. ⚠️ АНАЛИЗ СИМПТОМОВ И ОТКЛОНЕНИЙ ПО КЛИНИЧЕСКИМ ПРОТОКОЛАМ
                Сопоставь жалобы и показатели пациента со стандартами из прикрепленных клинических протоколов:
                * Выявленные отклонения (нарушения гемодинамики, ангинозные маркеры, аритмические проявления, признаки сердечной недостаточности или дислипидемии).
                * Ссылка на протокол: `> Стр. X: «Точная цитата критерия/нормы» — Клиническая интерпретация для данного пациента.`

                ### 3. 🩺 ПРЕДВАРИТЕЛЬНАЯ ДИАГНОСТИЧЕСКАЯ ГИПОТЕЗА (ДЛЯ ПРОВЕРКИ ВРАЧОМ)
                (Носит строго ориентировочный доврачебный характер, подлежит обязательной очной верификации)
                * Вероятный синдромальный диагноз или нозологическая группа (АГ / Стабильная ИБС / Нарушение ритма или проводимости / Хроническая сердечная недостаточность / Нарушение липидного обмена).
                * Категория сердечно-сосудистого риска (по SCORE2 / факторам риска / шкалам риска профильных протоколов).

                ### 4. 💊 ТЕРАПЕВТИЧЕСКАЯ ОРИЕНТИРОВКА (ПО СТАНДАРТАМ МИНЗДРАВА РФ)
                * Рекомендуемый протоколами класс стартовой или корригирующей терапии по профилю выявленной проблемы (гипотензивная, антиангинальная, антиаритмическая, пульсурежающая, квадротерапия ХСН или липидснижающая терапия).
                * Ссылка строго на блок RAG: `> Стр. X: «Цитата схемы/препаратов первого ряда» — Обоснование выбора.`
                * Немедикаментозные меры (ограничение натрия, модификация нагрузок, режим активности).

                ### 5. 🔍 ФОКУС ОЧНОГО ПРИЕМА (КЛИНИЧЕСКИЕ СЛЕПЫЕ ЗОНЫ И RED FLAGS)
                * Симптомы тревоги (Red Flags), требующие исключения неотложных состояний (ОКС, расслоение, жизнеугрожающие желудочковые тахиаритмии, отек легких).
                * Таргетные физикальные и инструментальные тесты (аускультация шумов/хрипов, ЭКГ в 12 отведениях, ЭхоКГ, суточное мониторирование ЭКГ по Холтеру, контроль липидного спектра/тропонинов/NT-proBNP).
                * Лекарственные взаимодействия и скрытые факторы (прием НПВП, капель, проаритмогенных средств).

                ТРЕБОВАНИЯ:
                - Стиль: строгий клинический язык, фактологическая плотность, отсутствие общих рассуждений («вода»).
                - Любой вывод о лечении или диагностике должен опираться на прикрепленные фрагменты рекомендаций с указанием страниц.
                """
                doctor_res = call_gemini(
                    [{"role": "user",
                      "content": f"ФАКТЫ ПАЦИЕНТА:\n{facts}\n\nКЛИНИЧЕСКИЕ ПРОТОКОЛЫ (RAG):\n{rag_context}"}],
                    DOCTOR_PROMPT,
                    0.2
                )

        # 5. Отрисовка результатов
        st.success("✅ Карта и анализ сформированы!")
        tab1, tab2 = st.tabs(["📋 Выписка для ЭМК", "🩺 Аналитический отчет врача"])
        with tab1:
            st.markdown(emr_res)
        with tab2:
            st.markdown(doctor_res)

        st.session_state.rag_results = {"emr": emr_res, "doctor": doctor_res}
        st.session_state.messages.append({
            "role": "assistant",
            "emr_content": emr_res,
            "doctor_content": doctor_res
        })

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Начать новый опрос пациента", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": CURRENT_PROFILE["welcome_message"]}]
            st.session_state.interview_completed = False
            st.session_state.is_waiting_for_assistant = False
            st.session_state.stage = "interview"
            st.session_state.rag_results = None
            st.session_state.extra_info_text = ""
            st.rerun()