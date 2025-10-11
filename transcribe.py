# Нужен питон 3.11.9, можно установить локально, и дополнительные модули.
# Для запуска, корне проекта запускаем виртуальное окружение и запускаем скрипт:
#
# source .venv/bin/activate
# python transcribe.py

import os
import whisper
import subprocess

model = whisper.load_model("base")

audio_folder = "audio"
output_folder = "transcripts"
temp_audio_folder = "temp_audio"

os.makedirs(temp_audio_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

SUPPORTED_EXTENSIONS = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".mp4")

for filename in os.listdir(audio_folder):
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        continue

    input_path = os.path.join(audio_folder, filename)

    # Если это mp4 — извлекаем звук в mp3, иначе подаём файл напрямую
    if filename.lower().endswith(".mp4"):
        audio_only_path = os.path.join(
            temp_audio_folder, os.path.splitext(filename)[0] + ".mp3"
        )
        print(f"🎞️ Извлекаю звук из {filename}...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "libmp3lame",
                audio_only_path
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
    else:
        audio_only_path = input_path

    print(f"🔊 Расшифровываю: {filename}...")
    result = model.transcribe(audio_only_path, language="ru")
    text_filename = os.path.splitext(filename)[0] + ".txt"

    with open(os.path.join(output_folder, text_filename), "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"✅ Сохранено: {text_filename}\n")

print("🎉 Обработка завершена.")
