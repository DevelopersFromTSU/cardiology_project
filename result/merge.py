import os
import json
import re


def natural_sort_key(filename):
    """Сортирует файлы с учетом чисел (page_2 пойдет перед page_10)"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', filename)]


def merge_files(input_folder="../result", output_file="combined_book.md"):
    if not os.path.exists(input_folder):
        print(f"❌ Папка '{input_folder}' не найдена!")
        return

    # Получаем все файлы и сортируем их в естественном порядке
    files = sorted(os.listdir(input_folder), key=natural_sort_key)

    combined_content = []

    print(f"⏳ Начало сборки. Найдено файлов: {len(files)}")

    for filename in files:
        # Пропускаем файлы Python и сам итоговый файл, чтобы они не попали в книгу
        if filename.endswith('.py') or filename == output_file:
            continue

        filepath = os.path.join(input_folder, filename)
        if os.path.isdir(filepath):
            continue

        try:
            if filename.endswith('.json'):
                # Если это JSON от нашего пайплайна, достаем очищенный текст
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Проверяем оба возможных ключа из прошлых шагов
                    text = data.get("refined_text") or data.get("text") or ""
            else:
                # Если это обычный текстовый/md файл
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

            if text.strip():
                combined_content.append(text.strip())

        except Exception as e:
            print(f"⚠️ Ошибка при чтении файла {filename}: {e}")

    # Объединяем все куски через два переноса строки
    final_text = "\n\n" + "\n\n".join(combined_content) + "\n"

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(final_text)

    print(f"✅ Успешно собрано! Файл сохранен как: {output_file}")


if __name__ == "__main__":
    merge_files()