import sys
import os
import requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 1. Настройка путей для импорта локальных модулей
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# 2. Загрузка переменных окружения
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / ".env.txt"
load_dotenv(dotenv_path=env_path)

# Импортируем готовые функции поиска
from pipeline.pipeline3_retrieve.retriever import rewrite_patient_query, hybrid_search

# =====================================================================
# НАСТРОЙКА МЕДИЦИНСКОГО ПРОФИЛЯ (УНИВЕРСАЛЬНОСТЬ)
# =====================================================================
MEDICAL_PROFILES = {
    "cardiology": {
        "role_name": "кардиолог",
        "assistant_name": "кардио-ассистент",
        "critical_rules": "Оценивай систолическое и диастолическое давление НЕЗАВИСИМО. Если нижнее (диастолическое) давление превышает 110-120, алгоритм должен классифицировать это как высокий риск или кризовое состояние, независимо от систолических показателей.",
        "welcome_message": "Здравствуйте! Я ваш интеллектуальный кардио-ассистент. Перед приёмом врача мне нужно провести небольшое интервью, чтобы собрать точные данные о вашем самочувствии. Расскажите, пожалуйста, что вас беспокоит, и какое у вас обычно артериальное давление?"
    },
    "endocrinology": {
        "role_name": "эндокринолог",
        "assistant_name": "эндо-ассистент",
        "critical_rules": "Особое внимание обращай на уровень глюкозы и гликированного гемоглобина. Любые упоминания симптомов гипогликемии (дрожь, потливость, сильный голод) или гипергликемии (сильная жажда, частое мочеиспускание) отмечай как высокий риск.",
        "welcome_message": "Здравствуйте! Я ваш интеллектуальный эндо-ассистент. Расскажите, что вас беспокоит? Контролируете ли вы уровень сахара в крови?"
    }
}

# Выбор активного профиля для работы приложения
CURRENT_PROFILE = MEDICAL_PROFILES["cardiology"]

# Инициализация состояний сессии Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": CURRENT_PROFILE["welcome_message"]}
    ]

if "stage" not in st.session_state:
    st.session_state.stage = "interview"  # Возможные стадии: 'interview' или 'analysis'

if "rag_results" not in st.session_state:
    st.session_state.rag_results = None

# Системный промпт для стадии сбора анамнеза (Интервьюер)
INTERVIEW_SYSTEM_INSTRUCTION = (
    f"Ты — профессиональный, вежливый и чуткий медицинский робот-интервьюер ({CURRENT_PROFILE['assistant_name']}). "
    "Твоя ЕДИНСТВЕННАЯ задача на данном этапе — собрать полный анамнез пациента в формате естественного диалога. "
    "Не задавай все вопросы сразу, общайся как живой врач: комментируй ответы, задавай по 1-2 вопроса за раз.\n\n"
    "СПИСОК ПАРАМЕТРОВ, КОТОРЫЕ ТЕБЕ НУЖНО ВЫЯВИТЬ (Опирайся на официальный опросник):\n"
    "1. Артериальное давление (привычное и максимальное в мм.рт.ст.).\n"
    "2. Пульс (привычный и максимальный ударов в минуту).\n"
    "3. Курение (курит ли сейчас, стаж, сколько сигарет в день; если бросил — сколько лет не курит).\n"
    "4. Употребление алкоголя (частота в месяц, вид напитка, количество).\n"
    "5. Наличие аллергий (на что и какая реакция).\n"
    "6. Диета и привычки: употребление овощей/фруктов, контроль жира/холестерина, привычка подсаливать еду, сахар.\n"
    "7. Физическая активность (минут в день на ходьбу).\n"
    "8. Клиническая история (ИБС, аритмия, СД, инфаркт, инсульт, операции).\n"
    "9. Семейный анамнез (ранние инфаркты/инсульты/гипертония у родственников).\n\n"
    "ПРАВИЛА ПОВЕДЕНИЯ И ЗАЩИТА ДИАЛОГА:\n"
    f"- ЗАЩИТА ОТ СМЕНЫ ТЕМЫ (КРИТИЧЕСКИ ВАЖНО): Если пациент меняет тему, извинись, напомни, что ты специализированный {CURRENT_PROFILE['assistant_name']} и твоя цель — подготовить данные для врача, после чего СРАЗУ задай следующий невыясненный вопрос.\n"
    "- Будь поддерживающим, не ставь диагнозы самостоятельно и не назначай лекарства! Твоя цель — собрать данные для базы.\n"
    "- Если пациент предоставил много информации, зафиксируй её и мягко спроси то, что осталось невыясненным.\n"
    "- Когда ты собрал все ключевые метрики, вежливо скажи пациенту, что анкета готова и он может нажать кнопку 'Запустить клинический анализ' на панели.\n"
    "- ВАЖНО: НИКОГДА НЕ ЦИТИРУЙ И НЕ ДУБЛИРУЙ историю диалога или ответы пациента в своем сообщении. Пиши только свою новую реплику.\n"
    "- КРИТИЧЕСКИ ВАЖНО: Твоя задача сейчас — сгенерировать СТРОГО ОДНУ реплику (твой следующий вопрос) и остановиться. Задай вопрос и жди, пока пациент ответит."
)

# Универсальная функция для вызова YandexGPT
def call_yandex_gpt(messages_history, system_prompt, temperature=0.3):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    api_key = os.getenv("YANDEX_API_KEY")

    if not folder_id or not api_key:
        return "⚠️ Ошибка: Не заданы YANDEX_FOLDER_ID или YANDEX_API_KEY в файле .env"

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json"
    }

    formatted_messages = [{"role": "system", "text": system_prompt}]
    for msg in messages_history:
        if "content" in msg:
            formatted_messages.append({
                "role": msg["role"],
                "text": msg["content"]
            })

    payload = {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": "2000"
        },
        "messages": formatted_messages
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        return response_data['result']['alternatives'][0]['message']['text'].strip()
    except Exception as e:
        return f"⚠️ Ошибка при обращении к YandexGPT: {e}"

# Заголовок веб-страницы
st.set_page_config(page_title="Intelligent Medical Assistant", page_icon="🩺", layout="wide")
st.title(f"🩺 Интеллектуальный {CURRENT_PROFILE['assistant_name'].capitalize()}")
st.caption("Система предварительного сбора анамнеза и RAG-анализа (Powered by YandexGPT & Qdrant)")

# Разделение интерфейса: Боковая панель
with st.sidebar:
    st.header("📋 Статус анкетирования")

    if st.session_state.stage == "interview":
        st.info("🤖 Ассистент сейчас собирает ваш анамнез. Отвечайте на вопросы в чате.")

        if st.button("🏁 Запустить клинический анализ", type="primary", use_container_width=True):
            st.session_state.stage = "analysis"
            st.rerun()

    else:
        st.success("✅ Опрос завершен! Выполняется RAG-анализ рекомендаций.")
        if st.button("🔄 Сбросить диалог и начать заново", type="secondary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant",
                 "content": CURRENT_PROFILE["welcome_message"]}
            ]
            st.session_state.stage = "interview"
            st.session_state.rag_results = None
            st.rerun()

# Отображение истории сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "patient_content" in msg and "doctor_content" in msg:
            st.markdown("✅ **Клинический анализ завершен. Подготовлены два отчета:**")
            tab1, tab2 = st.tabs(["👤 Отчет для пациента", "🩺 Отчет для врача (Клинический)"])
            with tab1:
                st.markdown(msg["patient_content"])
            with tab2:
                st.markdown(msg["doctor_content"])
        elif "content" in msg:
            st.markdown(msg["content"])

# Обработка ввода пользователя
if user_input := st.chat_input("Введите ваш ответ или вопрос здесь..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- СТАДИЯ 1: ИНТЕРВЬЮ ---
    if st.session_state.stage == "interview":
        with st.chat_message("assistant"):
            with st.spinner("Формирую вопрос..."):
                assistant_response = call_yandex_gpt(
                    messages_history=st.session_state.messages,
                    system_prompt=INTERVIEW_SYSTEM_INSTRUCTION,
                    temperature=0.3
                )
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# --- СТАДИЯ 2: КЛИНИЧЕСКИЙ АНАЛИЗ И RAG ---
if st.session_state.stage == "analysis" and st.session_state.rag_results is None:
    with st.chat_message("assistant"):

        full_dialogue_history = "\n".join(
            [f"{m['role']}: {m.get('content', '')}" for m in st.session_state.messages if "content" in m])

        # === ЭТАП: ОЧИСТКА И СТРУКТУРИРОВАНИЕ АНАМНЕЗА ===
        with st.spinner("⏳ Структурирование данных и очистка от визуального мусора..."):
            EXTRACTION_SYSTEM_INSTRUCTION = (
                "Ты — клинический дата-сайентист. Твоя задача: прочитать неструктурированную историю диалога с пациентом "
                "и извлечь из неё сухие медицинские факты, полностью игнорируя любые бытовые темы, эмоции и прочий нерелевантный текст.\n\n"
                "Сформируй отчет СТРОГО по 10 пунктам опросника. Если пациент не дал информации по пункту, напиши 'Нет данных'.\n"
                "1. Артериальное давление (привычное/максимальное):\n"
                "2. Пульс (привычный/максимальный):\n"
                "3. Курение:\n"
                "4. Алкоголь:\n"
                "5. Аллергия:\n"
                "6. Питание (овощи, соль, холестерин, сахар):\n"
                "7. Физическая активность:\n"
                "8. История болезней:\n"
                "9. Наследственность:\n"
                "10. СОПУТСТВУЮЩИЕ ФАКТОРЫ:\n\n"
                "КРИТИЧЕСКОЕ ПРАВИЛО: Верни только этот нумерованный список. Никаких вводных слов и выводов."
            )

            clean_anamnesis = call_yandex_gpt([{"role": "user", "content": full_dialogue_history}],
                                              EXTRACTION_SYSTEM_INSTRUCTION, temperature=0.1)

            with st.expander("📄 Посмотреть извлеченный структурированный анамнез"):
                st.markdown(clean_anamnesis)

        # === ЭТАП ПОИСКА ===
        with st.spinner("🔍 Идет поиск по клиническим рекомендациям в базе данных Qdrant..."):
            medical_query = rewrite_patient_query(clean_anamnesis)
            st.caption(f"**Сформированные клинические теги:** {medical_query}")

            try:
                # Безопасный вызов поиска с защитой от падения Qdrant
                retrieved_chunks = hybrid_search(search_query=medical_query, top_k=5)
            except Exception as search_error:
                st.error(f"⚠️ Ошибка подключения к базе знаний: {search_error}")
                retrieved_chunks = []

            if not retrieved_chunks:
                retrieved_context = "Клинические рекомендации по данному сочетанию симптомов в базе знаний не найдены."
            else:
                context_blocks = []
                for chunk in retrieved_chunks:
                    context_blocks.append(
                        f"--- ИСТОЧНИК: КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ, СТРАНИЦА {chunk.get('page', 'Неизвестно')} ---\n{chunk.get('text', '')}"
                    )
                retrieved_context = "\n\n".join(context_blocks)

        # --- ГЕНЕРАЦИЯ ОТЧЕТА 1: ДЛЯ ПАЦИЕНТА ---
        with st.spinner("📝 Формирую понятный отчет для пациента..."):
            PATIENT_SYSTEM_INSTRUCTION = (
                "Ты — эмпатичный и заботливый медицинский ИИ-ассистент. Твоя задача — составить краткий и понятный отчет для ПАЦИЕНТА "
                "по итогам его опроса. Опирайся на историю диалога и общие выводы из RAG-контекста, но НЕ используй сложные термины.\n\n"
                "СТРУКТУРА ОТВЕТА:\n"
                "1. **Ваша ситуация**: Опиши объективные показатели и факты из диалога. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать, что пациента что-то «беспокоит» или он на что-то «жалуется», если он сам не озвучивал активных жалоб, а просто называл цифры.\n"
                "2. **Почему мы так думаем**: Простыми словами свяжи его показатели и факторы риска с тем, как это влияет на здоровье.\n"
                "3. **Что сделать перед приемом**: Практические советы (например: вести дневник метрик, взять на прием старые исследования, сдать анализы).\n"
                "4. **Дисклеймер**: Напоминание, что этот отчет — предварительный анализ, а не диагноз.\n\n"
                "КРИТИЧЕСКОЕ ПРАВИЛО: Пиши максимально просто и дружелюбно. КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ цитировать куски текста из RAG-базы."
            )
            patient_prompt = f"--- ИСТОРИЯ ИНТЕРВЬЮ ---\n{full_dialogue_history}\n\n--- МЕДИЦИНСКИЙ КОНТЕКСТ ---\n{retrieved_context}"
            patient_report = call_yandex_gpt([{"role": "user", "content": patient_prompt}], PATIENT_SYSTEM_INSTRUCTION,
                                             temperature=0.3)

        # --- ГЕНЕРАЦИЯ ОТЧЕТА 2: ДЛЯ ВРАЧА ---
        with st.spinner(f"🩺 Формирую глубокий клинический отчет для врача ({CURRENT_PROFILE['role_name']})..."):
            DOCTOR_SYSTEM_INSTRUCTION = (
                f"Ты — эксперт-{CURRENT_PROFILE['role_name']} и аналитик данных RAG. Твоя задача — составить строгий клинический отчет для ВРАЧА перед очным приемом. "
                "Перед тобой история диалога и извлеченные фрагменты из Клинических рекомендаций Минздрава РФ (RAG-контекст).\n\n"
                "СТРУКТУРА ОТВЕТА:\n"
                "1. **Предварительная клиническая картина**: Объективное резюме показателей. Исключи субъективные оценки (не пиши 'умеренно', указывай точную дозу). Не используй слово 'жалуется', если озвучены только факты/цифры.\n"
                f"2. **Клинический анализ показателей**: Оценка метрик пациента. КРИТИЧЕСКИ ВАЖНО: {CURRENT_PROFILE['critical_rules']}\n"
                "3. **Доказательная база (ОБОСНОВАНИЕ)**: СТРОГО ССЫЛАЙСЯ на предоставленные чанки RAG-контекста. "
                "Для каждого вывода или рекомендации ОБЯЗАТЕЛЬНО указывай конкретный номер страницы, "
                "который ты берешь из заголовка чанка `--- ИСТОЧНИК: КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ, СТРАНИЦА X ---`.\n"
                "Используй визуальное выделение цитат (начинай строку со знака '>', например: `> [Страница 220]: Цитата...`).\n\n"
                "4. **Фокус на очном приеме**: На что врачу обратить внимание (проверка совместимости препаратов, дообследования, стратегия терапии).\n\n"
                "КРИТИЧЕСКОЕ ПРАВИЛО: Отчет должен быть сухим, сугубо профессиональным (медицинским языком) и максимально опираться на факты из RAG-контекста."
            )
            doctor_prompt = f"--- ИСТОРИЯ ИНТЕРВЬЮ ---\n{full_dialogue_history}\n\n--- ИЗВЛЕЧЕННЫЕ RAG-ЧАНКИ ИЗ БАЗЫ ---\n{retrieved_context}"
            doctor_report = call_yandex_gpt([{"role": "user", "content": doctor_prompt}], DOCTOR_SYSTEM_INSTRUCTION,
                                            temperature=0.1)

        # Отрисовка вкладок и сохранение в историю
        st.markdown("✅ **Клинический анализ завершен. Подготовлены два отчета:**")
        tab1, tab2 = st.tabs(["👤 Отчет для пациента", f"🩺 Отчет для врача ({CURRENT_PROFILE['role_name'].capitalize()})"])

        with tab1:
            st.markdown(patient_report)
        with tab2:
            st.markdown(doctor_report)

        st.session_state.rag_results = {"patient": patient_report, "doctor": doctor_report}
        st.session_state.messages.append({
            "role": "assistant",
            "patient_content": patient_report,
            "doctor_content": doctor_report
        })