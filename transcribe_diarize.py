# Нужен Python 3.11.9, установленный локально, и дополнительные модули (см. requirements.txt).
# Перед запуском в корне проекта создай и активируй виртуальное окружение, затем запусти скрипт:
#
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# python transcribe_diarize.py [--speakers N]
#
# Использование:
#   python transcribe_diarize.py              # автоопределение количества спикеров
#   python transcribe_diarize.py -s 2         # явно указать 2 спикера
#   python transcribe_diarize.py --speakers 3 # явно указать 3 спикера
#   python transcribe_diarize.py --help       # показать справку
#
# Входные файлы берём из папки "audio".
# Промежуточные WAV-файлы кладём в "temp_audio".
# Все результаты (диалоги со спикерами) сохраняем в одну папку "transcripts":
#   - <basename>.diarized.txt
#   - <basename>.diarized.json
#
# Примечания:
# - Для извлечения/конвертации аудио требуется установленный ffmpeg в системе (brew/apt/choco).
# - Маппинг сегментов Whisper - спикеры делаем по "середине" сегмента.
#   При перекрытии назначаем спикера с максимальным покрытием по времени.

import os
import json
import time
import subprocess
import argparse
from typing import List, Tuple, Dict, Optional, Union

import whisper  # openai-whisper, можно еще использовать faster-whisper
from simple_diarizer.diarizer import Diarizer

try:
    import torchaudio
    if hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend("soundfile");
except Exception:
    pass


AUDIO_FOLDER: str = "audio"
TEMP_AUDIO_FOLDER: str = "temp_audio"
OUTPUT_FOLDER: str = "transcripts"
LOG_FILE: str = "processing.log.csv"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".mp4")

WHISPER_MODEL_NAME: str = "medium"
WHISPER_LANGUAGE: str = "ru"

SPEAKER_PREFIX: str = "SPEAKER_"

TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1


def ensure_directories() -> None:
    """Создаёт служебные папки при необходимости."""
    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def is_supported(filename: str) -> bool:
    """Проверяет, подходит ли расширение файла под список SUPPORTED_EXTENSIONS."""
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def run_ffmpeg_to_wav_16k_mono(input_path: str, output_wav_path: str) -> None:
    """
    Приводит любой вход (аудио/видео) к WAV 16кГц моно.
    Требуется установленный ffmpeg в системе.
    """
    # -y: перезаписать; -ac 1: моно; -ar 16000: частота; -vn: без видео
    command: List[str] = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "wav",
        output_wav_path,
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg не найден. Установи ffmpeg (например, brew install ffmpeg).")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"ffmpeg не смог обработать файл: {input_path}")


def seconds_to_timestamp(seconds: float) -> str:
    """
    Перевод секунды -> строковый таймкод HH:MM:SS.mmm
    Удобно для человекочитаемых .txt.
    """
    if seconds < 0:
        seconds = 0.0
    milliseconds_total: int = int(round(seconds * 1000))
    hours: int = milliseconds_total // (3600 * 1000)
    minutes: int = (milliseconds_total % (3600 * 1000)) // (60 * 1000)
    seconds_int: int = (milliseconds_total % (60 * 1000)) // 1000
    milliseconds: int = milliseconds_total % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"


def format_duration(seconds: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def write_log(
    filename: str,
    duration: float,
    num_speakers: int,
    processing_time: float
) -> None:
    """Записывает информацию об обработке в лог-файл."""
    from datetime import datetime
    
    # Если файл не существует — создаём с заголовком
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("timestamp,filename,speakers,duration_sec,model,processing_time_sec\n")
    
    # Дописываем строку с данными
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{filename},{num_speakers},{format_duration(duration)},{WHISPER_MODEL_NAME},{format_duration(processing_time)}\n")


# --------------------
# Диаризация + ASR
# --------------------

def run_diarization(wav_path: str, num_speakers: Optional[int]) -> List[Dict[str, Union[float, str]]]:
    """
    Запускает simple_diarizer над WAV-файлом.
    Возвращает список словарей с ключами: start, end, label (в секундах и текстовая метка спикера).
    """
    # Простая конфигурация: эмбеддер по умолчанию и spectral clustering.
    diarizer: Diarizer = Diarizer()  # можно задать embed_model='xvec'|'ecapa', cluster_method='sc'|'ahc'
    if num_speakers is not None and num_speakers > 0:
        raw_segments = diarizer.diarize(wav_path, num_speakers=num_speakers)
    else:
        raw_segments = diarizer.diarize(wav_path)

    # Нормализуем в список словарей одного формата
    normalized: List[Dict[str, Union[float, str]]] = []
    for segment in raw_segments:
        # segment может быть tuple(start, end, label) или dict с такими ключами — приведём аккуратно
        if isinstance(segment, (list, tuple)) and len(segment) >= 3:
            start_val, end_val, label_val = segment[0], segment[1], segment[2]
            normalized.append({"start": float(start_val), "end": float(end_val), "label": str(label_val)})
        elif isinstance(segment, dict):
            start_val = float(segment.get("start", 0.0))
            end_val = float(segment.get("end", 0.0))
            label_val = str(segment.get("label", "SPEAKER"))
            normalized.append({"start": start_val, "end": end_val, "label": label_val})
    # Сортировка по времени начала для детерминизма
    normalized.sort(key=lambda x: float(x["start"]))
    return normalized


def remap_speaker_labels(diar_segments: List[Dict[str, Union[float, str]]]) -> Dict[str, str]:
    """
    Приводит любые исходные лейблы к последовательным SPEAKER_1..N.
    Возвращает отображение {исходный_лейбл -> нормализованный_лейбл}.
    """
    mapping: Dict[str, str] = {}
    counter: int = 1
    for seg in diar_segments:
        original_label: str = str(seg["label"])
        if original_label not in mapping:
            mapping[original_label] = f"{SPEAKER_PREFIX}{counter}"
            counter += 1
    return mapping


def run_whisper_with_segments(
    wav_path: str,
    model,
    language: str,
    robust: bool = False,
) -> Tuple[List[Dict[str, Union[float, str]]], float]:
    """
    Запускает Whisper и возвращает (список сегментов, duration).

    robust=False (по умолчанию): обычный путь Whisper, качество как правило лучше.
    robust=True: condition_on_previous_text=False — включай, если запись «залипла»
    в "..." (тихий/невнятный старт вырождается в "...", и эта подсказка тащит
    "..." через весь файл). Отключает передачу предыдущего текста как подсказки.
    """
    result = model.transcribe(
        wav_path,
        language=language,
        verbose=False,
        condition_on_previous_text=not robust,
    )
    
    segments: List[Dict[str, Union[float, str]]] = []
    duration: float = 0.0
    
    for seg in result.get("segments", []):
        start_val: float = float(seg.get("start", 0.0))
        end_val: float = float(seg.get("end", 0.0))
        text_val: str = str(seg.get("text", "")).strip()
        if end_val > start_val and text_val:
            segments.append({"start": start_val, "end": end_val, "text": text_val})
        duration = max(duration, end_val)  # последний end = duration
    
    return segments, duration


def compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Возвращает длительность пересечения двух интервалов [a_start, a_end] и [b_start, b_end]."""
    left: float = max(a_start, b_start)
    right: float = min(a_end, b_end)
    return max(0.0, right - left)


def assign_speaker_to_asr_segment(
    asr_start: float,
    asr_end: float,
    diar_segments: List[Dict[str, Union[float, str]]],
) -> Optional[str]:
    """
    Назначает спикера ASR-сегменту по правилу:
    1) Берём середину сегмента и ищем интервал диаризации, куда она попадает.
    2) Если таких несколько или середина «на границе», выбираем по максимальному перекрытию.
    3) Если вообще перекрытия нет (крайний случай) — возвращаем None.
    """
    midpoint: float = (asr_start + asr_end) / 2.0

    # Кандидаты по попаданию середины
    candidates: List[Tuple[str, float]] = []
    for seg in diar_segments:
        d_start: float = float(seg["start"])
        d_end: float = float(seg["end"])
        d_label: str = str(seg["label"])

        # Середина попала в интервал?
        if d_start <= midpoint <= d_end:
            # Перекрытие всего сегмента для устойчивости на границах
            overlap: float = compute_overlap(asr_start, asr_end, d_start, d_end)
            if overlap > 0:
                candidates.append((d_label, overlap))

    if candidates:
        # Выбираем кандидата с максимальным перекрытием
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[0][0]

    # Если середина не попала никуда (редкий случай), выбираем по максимальному перекрытию вообще
    best_label: Optional[str] = None
    best_overlap: float = 0.0
    for seg in diar_segments:
        d_start = float(seg["start"])
        d_end = float(seg["end"])
        d_label = str(seg["label"])
        overlap = compute_overlap(asr_start, asr_end, d_start, d_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = d_label

    return best_label


def save_outputs(
    base_name: str,
    assigned_segments: List[Dict[str, Union[float, str]]],
    output_folder: str,
) -> None:
    """
    Сохраняет .diarized.txt и .diarized.json в output_folder.
    assigned_segments: список объектов {start, end, speaker, text}
    """
    txt_path: str = os.path.join(output_folder, f"{base_name}.diarized.txt")
    json_path: str = os.path.join(output_folder, f"{base_name}.diarized.json")

    # Человекочитаемый TXT
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        for seg in assigned_segments:
            start_ts: str = seconds_to_timestamp(float(seg["start"]))
            end_ts: str = seconds_to_timestamp(float(seg["end"]))
            speaker: str = str(seg["speaker"])
            text: str = str(seg["text"])
            f_txt.write(f"[{start_ts}–{end_ts}] {speaker}: {text}\n")

    # Структурированный JSON
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(assigned_segments, f_json, ensure_ascii=False, indent=2)

    print(f"✅ Сохранено: {os.path.basename(txt_path)} и {os.path.basename(json_path)}")


def process_one_file(input_path: str, num_speakers: Optional[int], model, robust: bool = False) -> None:
    """
    Обрабатывает один входной файл:
    - конвертирует в WAV 16k моно,
    - запускает диаризацию,
    - запускает Whisper,
    - сопоставляет сегменты и сохраняет результаты.

    Args:
        input_path: путь к входному аудиофайлу
        num_speakers: количество спикеров (None для автоопределения)
        model: загруженная модель Whisper
        robust: устойчивый режим Whisper против залипания в "..." (см. run_whisper_with_segments)
    """
    start_time = time.time()
    file_stem: str = os.path.splitext(os.path.basename(input_path))[0]
    wav_path: str = os.path.join(TEMP_AUDIO_FOLDER, f"{file_stem}.16k_mono.wav")

    # Конвертация к целевому формату
    if input_path.lower().endswith(".wav"):
        # На всякий случай тоже приводим к 16k/mono (чтобы не ловить несоответствие формата)
        run_ffmpeg_to_wav_16k_mono(input_path, wav_path)
    else:
        run_ffmpeg_to_wav_16k_mono(input_path, wav_path)

    # Диаризация
    print(f"🧩 Диаризация (определение спикеров): {os.path.basename(input_path)} ...")
    diar_segments = run_diarization(wav_path, num_speakers)
    if not diar_segments:
        print("⚠️ Диаризация не вернула сегменты. Продолжаю без спикеров (все как SPEAKER_1).")

    # Нормализуем метки спикеров в SPEAKER_1..N
    label_map = remap_speaker_labels(diar_segments)
    for seg in diar_segments:
        original = str(seg["label"])
        seg["label"] = label_map.get(original, original)

    # ASR (Whisper) — сегменты с таймкодами и текстом
    print(f"🔊 Расшифровка речи Whisper: {os.path.basename(input_path)} ...")
    asr_segments, duration = run_whisper_with_segments(wav_path, model, WHISPER_LANGUAGE, robust=robust)

    # Маппинг: каждому ASR-сегменту назначаем спикера
    assigned: List[Dict[str, Union[float, str]]] = []
    for seg in asr_segments:
        seg_start: float = float(seg["start"])
        seg_end: float = float(seg["end"])
        seg_text: str = str(seg["text"])
        if diar_segments:
            chosen_label: Optional[str] = assign_speaker_to_asr_segment(seg_start, seg_end, diar_segments)
            speaker_label: str = chosen_label if chosen_label is not None else f"{SPEAKER_PREFIX}1"
        else:
            speaker_label = f"{SPEAKER_PREFIX}1"

        assigned.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": speaker_label,
            "text": seg_text,
        })

    processing_time = time.time() - start_time
    num_speakers_logged: int = len(label_map) or 1

    write_log(
        filename=os.path.basename(input_path),
        duration=duration,
        num_speakers=num_speakers_logged,
        processing_time=processing_time
    )

    # Сохранение результатов в одну папку
    save_outputs(file_stem, assigned, OUTPUT_FOLDER)


def parse_arguments() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    
    Returns:
        Namespace с аргументами (speakers)
    """
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио с определением спикеров (speaker diarization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python transcribe_diarize.py              # автоопределение количества спикеров
  python transcribe_diarize.py -s 2         # 2 спикера
  python transcribe_diarize.py --speakers 3 # 3 спикера
  python transcribe_diarize.py -s 2 --robust  # устойчивый режим, если текст залип в "..."
        """
    )
    
    parser.add_argument(
        "-s", "--speakers",
        type=int,
        default=None,
        metavar="N",
        help="Количество спикеров (по умолчанию: автоопределение)"
    )

    parser.add_argument(
        "--robust",
        action="store_true",
        help="Устойчивый режим Whisper: включай, если запись 'залипла' в '...' "
             "(condition_on_previous_text=False). Обычно не нужен — качество по умолчанию выше."
    )

    return parser.parse_args()


def main() -> None:
    # Парсим аргументы командной строки
    args = parse_arguments()
    num_speakers = args.speakers
    robust = args.robust

    if num_speakers is not None:
        print(f"📊 Количество спикеров: {num_speakers}")
    else:
        print("📊 Количество спикеров: автоопределение")

    if robust:
        print("🛡️ Устойчивый режим Whisper (condition_on_previous_text=False)")
    
    ensure_directories()

    # Перебираем файлы во входной папке
    input_files: List[str] = []
    for filename in os.listdir(AUDIO_FOLDER):
        if not is_supported(filename):
            continue
        input_files.append(os.path.join(AUDIO_FOLDER, filename))

    if not input_files:
        print("⚠️ В папке 'audio' не найдено подходящих файлов.")
        return

    print(f"🎬 Найдено файлов для обработки: {len(input_files)}")
    print("⏳ Загружаю Whisper модель...")
    model = whisper.load_model(WHISPER_MODEL_NAME)
    print("✅ Модель загружена")

    for input_path in input_files:
        print(f"🎞️ Готовлю: {os.path.basename(input_path)}")
        try:
            process_one_file(input_path, num_speakers, model, robust=robust)
        except Exception as error:
            print(f"❌ Ошибка при обработке {os.path.basename(input_path)}: {error}")

    print("🎉 Обработка завершена.")


if __name__ == "__main__":
    main()
