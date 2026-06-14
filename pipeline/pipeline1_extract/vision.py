import os
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

def get_gemini_client():
    os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
    os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def describe_image(pil_img): # <-- Теперь на вход идет объект картинки PIL
    if pil_img is None:
        return ""

    client = get_gemini_client()
    model_id = "gemini-2.5-flash"

    # Выносим личность и жесткие правила в системную роль
    sys_instr = (
        "Ты — профессиональный медицинский оцифровщик. Твоя задача: перенести текст с изображения "
        "в текстовый формат с ПРЕДЕЛЬНОЙ точностью. ЗАПРЕЩЕНО использовать внешние знания.\n\n"
        "Правила форматирования:\n"
        "1. АЛГОРИТМЫ: Используй вложенные списки и стрелки `->` для описания логических переходов.\n"
        "2. ТЕРМИНЫ: Тщательно распознавай символы. Не путай кириллицу с латиницей.\n"
        "3. ЗАГОЛОВКИ: Используй '###'.\n"
        "4. ТАБЛИЦЫ (БЕЗ ГРАФИКИ): Категорически запрещено использовать Markdown-сетку `|---|`. "
        "Каждую ячейку таблицы преобразуй в отдельный пункт списка, объединяя заголовок строки и заголовок столбца "
        "со значением ячейки через стрелки `->`. Каждая строка должна быть самодостаточной.\n"
        "5. БЕЗ ЛИШНЕГО: Выводи только оцифрованный текст без твоих комментариев."
        "6. АБЗАЦЫ И ПЕРЕНОСЫ: Используй двойной перенос строки (\\n\\n) СТРОГО для разделения независимых логических "
        "блоков или пунктов. КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ разрывать одно предложение или один пункт списка переносами "
        "строк (\\n). Одно предложение = одна сплошная строка."
    )

    try:
        # Передаем объект PIL напрямую в contents
        response = client.models.generate_content(
            model=model_id,
            contents=[pil_img],
            config={
                "system_instruction": sys_instr,
                "temperature": 0.1
            }
        )
        return response.text
    except Exception as e:
        print(f"⚠️ Ошибка Gemini Vision: {e}")
        return ""
