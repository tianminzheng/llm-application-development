import openai

messages = [ {"role": "system", "content": "你是一个翻译家"} ]

def detect_language(text):
    prompt = f"以下内容是哪种语言：{text}？"

    messages.append(
        {"role": "user", "content": prompt},
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    language = response.choices[0].message.content
    return language

def translate_text(text, target_language):
    prompt = f"将{text}翻译成{target_language}"

    messages.append(
        {"role": "user", "content": prompt},
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    translated_text = response.choices[0].message.content
    return translated_text



input_text = input()
detected_language = detect_language(input_text)
print("原始语言:", detected_language)

target_language = input()
translated_text = translate_text(input_text, target_language)
print(f"翻译成{target_language}: {translated_text}")