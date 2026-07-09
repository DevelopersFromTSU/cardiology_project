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

# Инициализация состояний сессии Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Здравствуйте! Я ваш интеллектуальный кардио-ассистент. Перед приёмом врача мне нужно провести небольшое интервью, чтобы собрать точные данные о вашем самочувствии. Расскажите, пожалуйста, что вас беспокоит, и какое у вас обычно артериальное давление?"}
    ]

if "stage" not in st.session_state:
    st.session_state.stage = "interview"  # Возможные стадии: 'interview' или 'analysis'

if "rag_results" not in st.session_state:
    st.session_state.rag_results = None

# Системный промпт для стадии сбора анамнеза (Интервьюер)
INTERVIEW_SYSTEM_INSTRUCTION = (
    "Ты — профессиональный, вежливый и чуткий медицинский робот-интервьюер (кардио-ассистент). "
    "Твоя ЕДИНСТВЕННАЯ задача на данном этапе — собрать полный анамнез пациента в формате естественного диалога. "
    "Не задавай все вопросы сразу, общайся как живой врач: комментируй ответы, задавай по 1-2 вопроса за раз.\n\n"
    "СПИСОК ПАРАМЕТРОВ, КОТОРЫЕ ТЕБЕ НУЖНО ВЫЯВИТЬ (Опирайся на официальный опросник):\n"
    "1. Артериальное давление (привычное и максимальное в мм.рт.ст.).\n"
    "2. Пульс (привычный и максимальный ударов в минуту).\n"
    "3. Курение (курит ли сейчас, стаж, сколько сигарет в день; если бросил — сколько лет не курит).\n"
    "4. Употребление алкоголя (частота в месяц, вид напитка, количество).\n"
    "5. Наличие аллергий (на что и какая реакция).\n"
    "6. Диета и привычки: употребление овощей/фруктов (более 400г в день?), контроль жира/холестерина по этикеткам, привычка подсаливать еду не пробуя, количество сладкого/сахара (более 6 ложек в день?).\n"
    "7. Физическая активность (сколько минут в день уходит на умеренную или быструю ходьбу?).\n"
    "8. Клиническая история (есть ли и с какого возраста: гипертензия, ИБС, аритмия, сахарный диабет; были ли инфаркт, инсульт или операции на сердце — указать годы).\n"
    "9. Семейный анамнез (были ли ранние инфаркты/инсульты/гипертония у родителей или родных братьев/сестер: матери/сестры до 65 лет, отца/братья до 55 лет).\n\n"
    "ПРАВИЛА ПОВЕДЕНИЯ:\n"
    "- Будь поддерживающим, не ставь диагнозы самостоятельно и не назначай лекарства! Твоя цель — собрать данные для базы.\n"
    "- Если пациент предоставил много информации в первом сообщении, зафиксируй её и мягко спроси то, что осталось невыясненным.\n"
    "- Когда ты чувствуешь, что собрал практически все ключевые метрики (или пациент заявляет, что это всё), вежливо скажи ему, что анкета готова и он может нажать кнопку 'Запустить клинический анализ' на панели.\n"
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
        # Игнорируем специальные составные сообщения с вкладками при отправке в API (там ключи patient_content/doctor_content)
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
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        response_data = response.json()
        return response_data['result']['alternatives'][0]['message']['text'].strip()
    except Exception as e:
        return f"⚠️ Ошибка при обращении к YandexGPT: {e}"


# Заголовок веб-страницы
st.set_page_config(page_title="Cardio Assistant", page_icon="🫀", layout="wide")
st.title("❤️ Интеллектуальный Кардио-Ассистент")
st.caption("Система предварительного сбора анамнеза и анализа жалоб (Powered by YandexGPT & Qdrant)")

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
                 "content": "Здравствуйте! Я ваш интеллектуальный кардио-ассистент. Расскажите, пожалуйста, что вас беспокоит, и какое у вас обычно артериальное давление?"}
            ]
            st.session_state.stage = "interview"
            st.session_state.rag_results = None
            st.rerun()

# Отображение истории сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Рисуем вкладки (Tabs) для финального отчета
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

        # Собираем всю историю в единый текст
        full_dialogue_history = "\n".join(
            [f"{m['role']}: {m.get('content', '')}" for m in st.session_state.messages if "content" in m])

        with st.spinner("🔍 Идет поиск по клиническим рекомендациям в базе данных Qdrant..."):
            medical_query = rewrite_patient_query(full_dialogue_history)
            st.caption(f"**Сформированные клинические теги:** {medical_query}")

            retrieved_chunks = hybrid_search(search_query=medical_query, top_k=10)
            if not retrieved_chunks:
                retrieved_context = "Клинические рекомендации по данному сочетанию симптомов в базе знаний не найдены."
            else:
                # [ИСПРАВЛЕНО]: Адаптация под структуру словаря из retriever.py
                context_blocks = []
                for chunk in retrieved_chunks:
                    context_blocks.append(
                        f"--- ИСТОЧНИК: КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ, СТРАНИЦА {chunk['page']} ---\n{chunk['text']}"
                    )
                retrieved_context = "\n\n".join(context_blocks)

        # --- ГЕНЕРАЦИЯ ОТЧЕТА 1: ДЛЯ ПАЦИЕНТА ---
        with st.spinner("📝 Формирую понятный отчет для пациента..."):
            PATIENT_SYSTEM_INSTRUCTION = (
                "Ты — эмпатичный и заботливый медицинский ИИ-ассистент. Твоя задача — составить краткий и понятный отчет для ПАЦИЕНТА "
                "по итогам его опроса. Опирайся на историю диалога и общие выводы из RAG-контекста, но НЕ используй сложные термины.\n\n"
                "СТРУКТУРА ОТВЕТА:\n"
                "1. **Ваша ситуация**: Кратко опиши, что беспокоит пациента (что мы поняли из диалога).\n"
                "2. **Почему мы так думаем**: Простыми словами свяжи его жалобы (например, высокое давление, пульс, наследственность) с тем, как это влияет на самочувствие.\n"
                "3. **Что сделать перед приемом**: Практические советы (например: вести дневник давления 2 раза в день, взять на прием старые ЭКГ, не отменять самому препараты, сдать кровь на холестерин/сахар, если это логично вытекает из анамнеза).\n"
                "4. **Дисклеймер**: Напоминание, что этот отчет — предварительный анализ, а не диагноз.\n\n"
                "КРИТИЧЕСКОЕ ПРАВИЛО: Пиши максимально просто и дружелюбно. КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ цитировать куски текста из RAG-базы."
            )
            patient_prompt = f"--- ИСТОРИЯ ИНТЕРВЬЮ ---\n{full_dialogue_history}\n\n--- МЕДИЦИНСКИЙ КОНТЕКСТ ---\n{retrieved_context}"
            patient_report = call_yandex_gpt([{"role": "user", "content": patient_prompt}], PATIENT_SYSTEM_INSTRUCTION,
                                             temperature=0.3)

        # --- ГЕНЕРАЦИЯ ОТЧЕТА 2: ДЛЯ ВРАЧА ---
        with st.spinner("🩺 Формирую глубокий клинический отчет для врача с цитированием базы..."):
            DOCTOR_SYSTEM_INSTRUCTION = (
                "Ты — эксперт-кардиолог и аналитик данных RAG. Твоя задача — составить строгий клинический отчет для ВРАЧА перед очным приемом. "
                "Перед тобой история диалога и извлеченные фрагменты из Клинических рекомендаций Минздрава РФ (RAG-контекст).\n\n"
                "СТРУКТУРА ОТВЕТА:\n"
                "1. **Предварительная синдромальная оценка**: Краткое резюме симптомов пациента на основе диалога.\n"
                "2. **Клинический анализ показателей**: Оценка метрик пациента (почему АД, ЧСС, факторы риска расцениваются как хорошие или плохие).\n"
                "3. **Доказательная база (ОБОСНОВАНИЕ)**: СТРОГО ССЫЛАЙСЯ на предоставленные чанки RAG-контекста. "
                "Для каждого вывода или рекомендации ОБЯЗАТЕЛЬНО указывай конкретный номер страницы, "
                "который ты берешь из заголовка чанка `--- ИСТОЧНИК: КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ, СТРАНИЦА X ---` "
                "(например: 'Согласно разделу на странице 220 клинических рекомендаций...'). "
                "Используй визуальное выделение цитат (начинай строку со знака '>', например: `> [Страница 220]: Цитата...`). "
                "Объясни, почему модель сделала такой вывод, опираясь именно на эти выдержки с указанием страниц.\n\n"
                "4. **Фокус на очном приеме**: На что врачу обратить внимание (проверка совместимости препаратов, какие дообследования назначить, стратегия терапии Шаг 1 / Шаг 2).\n\n"
                "КРИТИЧЕСКОЕ ПРАВИЛО: Отчет должен быть сухим, сугубо профессиональным (медицинским языком) и максимально опираться на факты из RAG-контекста."
            )
            doctor_prompt = f"--- ИСТОРИЯ ИНТЕРВЬЮ ---\n{full_dialogue_history}\n\n--- ИЗВЛЕЧЕННЫЕ RAG-ЧАНКИ ИЗ БАЗЫ ---\n{retrieved_context}"
            doctor_report = call_yandex_gpt([{"role": "user", "content": doctor_prompt}], DOCTOR_SYSTEM_INSTRUCTION,
                                            temperature=0.1)

        # Отрисовка вкладок и сохранение в историю
        st.markdown("✅ **Клинический анализ завершен. Подготовлены два отчета:**")
        tab1, tab2 = st.tabs(["👤 Отчет для пациента", "🩺 Отчет для врача (Клинический)"])

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