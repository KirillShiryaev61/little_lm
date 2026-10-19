import re

# Римская цифра (обязательно не пустая: хотя бы одна цифра I, V, X, L, C, D, M)
STRICT_ROMAN = r"(?=[MDCLXVI])M*(?:C[MD]|D?C*)(?:X[CL]|L?X*)(?:I[XV]|V?I*)"

# Строка, состоящая только из римской цифры, возможно в кавычках/тире/скобках/с точкой
standalone_roman_pattern = re.compile(
    rf"^\s*[\(\[\—\-\s]*{STRICT_ROMAN}[\)\]\.\—\-\s]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def remove_all_chapter_markers(text: str) -> str:

    # 1. Удаляем отдельно стоящие римские цифры на строке
    text = standalone_roman_pattern.sub("", text)

    # 2. Чистим избыточные переносы строк
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

if __name__ == "__main__":
    with open("onegin.txt", "r", encoding='utf-8') as f:
        raw_text = f.read()
    cleaned_text = remove_all_chapter_markers(raw_text)
    with open("cleaned_onegin.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("Готово! Текст очищен.")

