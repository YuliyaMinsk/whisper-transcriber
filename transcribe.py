# Нужен питон 3.11.9, можно установить локально, и дополнительные модули.
# Для запуска, корне проекта запускаем виртуальное окружение и запускаем скрипт:
#
# source .venv/bin/activate
# python transcribe.py

import os
import subprocess
import torch
import time
from datetime import datetime
from transformers import pipeline
from typing import List, Tuple

# --- Константы ---
# Если нужно быстрее, можно использовать 'small' или 'base' модели
WHISPER_MODEL_NAME: str = "medium" 
WHISPER_LANGUAGE: str = "ru"

AUDIO_FOLDER: str = "audio"
OUTPUT_FOLDER: str = "transcripts"
TEMP_AUDIO_FOLDER: str = "temp_audio"
LOG_FILE: str = "processing.log.csv"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".mp4")

# --- Вспомогательные функции для логирования ---

def format_duration(seconds: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS (как в другом скрипте)."""
    seconds_int: int = int(seconds)
    hours: int = seconds_int // 3600
    minutes: int = (seconds_int % 3600) // 60
    secs: int = seconds_int % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def write_log(
    filename: str,
    duration_seconds: float,
    num_speakers: int,
    processing_time_seconds: float,
) -> None:
    """
    Записывает информацию об обработке в лог-файл.
    """
    # Если файл не существует — создаём с заголовком
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as log_file:
            log_file.write("timestamp,filename,speakers,duration_sec,model,processing_time_sec\n")

    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_formatted: str = format_duration(duration_seconds)
    processing_time_formatted: str = format_duration(processing_time_seconds)

    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"{timestamp},{filename},{num_speakers},{duration_formatted},"
            f"{WHISPER_MODEL_NAME},{processing_time_formatted}\n"
        )


def get_audio_duration_seconds(file_path: str) -> float:
    """
    Получает длительность аудиофайла в секундах через ffprobe.
    Требует ffmpeg/ffprobe. Если что-то пошло не так — вернёт 0.0.
    """
    command: List[str] = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        output: str = result.stdout.strip()
        if not output:
            return 0.0
        return float(output)
    except Exception:
        print(f"⚠️ Не удалось получить длительность для файла: {file_path}")
        return 0.0


# --- Нормализация аудио через ffmpeg ---

def convert_to_wav_16k_mono(input_path: str, output_path: str) -> None:
    """
    Приводит любой вход (аудио/видео) к WAV 16 kHz mono.
    Используется для всех форматов: .m4a, .mp3, .ogg, .mp4 и т.д.
    """
    command: List[str] = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",          # убираем видео, если оно есть
        "-ac", "1",     # 1 канал (mono)
        "-ar", "16000", # 16 kHz
        "-f", "wav",
        output_path,
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as error:
        print(f"❌ FFmpeg не смог обработать файл: {input_path}, ошибка: {error}")
        raise


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
        os.path.join(AUDIO_FOLDER, filename)
        for filename in os.listdir(AUDIO_FOLDER)
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not input_files:
        print(f"⚠️ В папке '{AUDIO_FOLDER}' не найдено подходящих файлов.")
        return

    for input_path in input_files:
        filename: str = os.path.basename(input_path)
        file_stem: str = os.path.splitext(filename)[0]

        print(f"🎞️ Подготавливаю аудио: {filename}...")
        processing_start_time: float = time.time()

        # Конвертируем любой файл в temp WAV 16k mono
        prepared_wav_path: str = os.path.join(TEMP_AUDIO_FOLDER, f"{file_stem}.16k.wav")
        try:
            convert_to_wav_16k_mono(input_path, prepared_wav_path)
        except Exception:
            # Если ffmpeg не справился — пропускаем файл
            continue

        # Запускаем распознавание по подготовленному WAV
        print(f"🔊 Расшифровываю: {filename}...")
        try:
            result = asr_pipeline(
                prepared_wav_path,
                generate_kwargs={"language": WHISPER_LANGUAGE},
            )

            # У разных версий transformers результат может быть dict или list[dict]
            if isinstance(result, list):
                transcribed_text: str = str(result[0]["text"])
            else:
                transcribed_text = str(result["text"])
        except Exception as error:
            print(f"❌ Ошибка при транскрибации {filename}: {error}")
            continue

        # Сохраняем текст
        text_filename: str = file_stem + ".txt"
        output_path: str = os.path.join(OUTPUT_FOLDER, text_filename)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(transcribed_text)

        # Логируем длительность и время обработки
        duration_seconds: float = get_audio_duration_seconds(input_path)  # по оригиналу
        processing_time_seconds: float = time.time() - processing_start_time

        write_log(
            filename=filename,
            duration_seconds=duration_seconds,
            num_speakers=1,
            processing_time_seconds=processing_time_seconds,
        )

        print(
            f"✅ Сохранено: {text_filename}\n"
            f"⏱ Длительность файла: {format_duration(duration_seconds)}, "
            f"обработка заняла: {format_duration(processing_time_seconds)}\n"
        )

    print("🎉 Обработка завершена.")


if __name__ == "__main__":
    main()
