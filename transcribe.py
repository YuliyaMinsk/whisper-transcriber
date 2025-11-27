# Нужен питон 3.11.9, можно установить локально, и дополнительные модули.
# Для запуска, корне проекта запускаем виртуальное окружение и запускаем скрипт:
#
# source .venv/bin/activate
# python transcribe.py

import os
import subprocess
import torch
from transformers import pipeline
from typing import List, Tuple

# --- Константы ---
# Если нужно быстрее, можно использовать 'small' или 'base' модели
WHISPER_MODEL_NAME: str = "medium" 
WHISPER_LANGUAGE: str = "ru"

AUDIO_FOLDER: str = "audio"
OUTPUT_FOLDER: str = "transcripts"
TEMP_AUDIO_FOLDER: str = "temp_audio"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".mp4")

# --- Инициализация ASR Pipeline ---

# Определяем устройство для использования (MPS для Apple Silicon)
if torch.backends.mps.is_available():
    DEVICE: str = "mps"
    print("💡 Используется ускорение MPS (GPU) для ASR.")
elif torch.cuda.is_available():
    DEVICE: str = "cuda"
    print("💡 Используется ускорение CUDA (GPU) для ASR.")
else:
    DEVICE: str = "cpu"
    print("⚠️ Используется CPU для ASR.")

# Создаем ASR Pipeline, который автоматически загружает модель и токенизатор,
# а также управляет отправкой данных на выбранное устройство.
try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=f"openai/whisper-{WHISPER_MODEL_NAME}",
        device=DEVICE,
        tokenizer=f"openai/whisper-{WHISPER_MODEL_NAME}",
        chunk_length_s=30,  # Для лучшей обработки длинных файлов
    )
except Exception as e:
    print(f"❌ Ошибка при инициализации ASR Pipeline: {e}")
    print("Проверьте установку torch, transformers и accelerate.")
    exit(1)


def run_ffmpeg_audio_extract(input_path: str, output_path: str) -> None:
    """Извлекает аудио из видеофайла (например, MP4) в MP3."""
    command: List[str] = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "libmp3lame",
        output_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ FFmpeg не смог обработать файл: {input_path}")
        raise

def main() -> None:
    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    input_files: List[str] = [
        os.path.join(AUDIO_FOLDER, f) 
        for f in os.listdir(AUDIO_FOLDER) 
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not input_files:
        print(f"⚠️ В папке '{AUDIO_FOLDER}' не найдено подходящих файлов.")
        return

    for input_path in input_files:
        filename = os.path.basename(input_path)
        file_stem = os.path.splitext(filename)[0]

        audio_only_path: str
        
        # Обработка видеофайлов
        if filename.lower().endswith(".mp4"):
            audio_only_path = os.path.join(TEMP_AUDIO_FOLDER, file_stem + ".mp3")
            print(f"🎞️ Извлекаю звук из {filename}...")
            try:
                run_ffmpeg_audio_extract(input_path, audio_only_path)
            except Exception:
                continue # Пропускаем файл при ошибке FFmpeg
        else:
            audio_only_path = input_path

        # Расшифровка с помощью Hugging Face Pipeline
        print(f"🔊 Расшифровываю: {filename}...")
        try:
            # Pipeline возвращает список dict'ов, берем первый элемент
            result = asr_pipeline(
                audio_only_path, 
                generate_kwargs={"language": WHISPER_LANGUAGE}
            )
            transcribed_text: str = result[0]["text"]
        except Exception as e:
            print(f"❌ Ошибка при транскрибации {filename}: {e}")
            continue

        # Сохранение результата
        text_filename = file_stem + ".txt"
        output_path = os.path.join(OUTPUT_FOLDER, text_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)

        print(f"✅ Сохранено: {text_filename}\n")

    print("🎉 Обработка завершена.")


if __name__ == "__main__":
    main()