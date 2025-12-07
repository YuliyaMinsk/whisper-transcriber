# Whisper Transcriber

Автоматическая транскрипция аудиофайлов с использованием OpenAI Whisper и определением спикеров (speaker diarization).

## Описание

Этот проект позволяет автоматически транскрибировать аудиофайлы в текст с помощью модели Whisper. Поддерживается два режима работы:

- **Простая транскрипция** (`transcribe.py`) — только распознавание речи с использованием Transformers Pipeline (Hugging Face)
- **Транскрипция с диаризацией** (`transcribe_diarize.py`) — распознавание речи (openai-whisper) + определение спикеров (simple-diarizer)

Скрипты обрабатывают файлы из папки `audio/` и сохраняют результаты в папку `transcripts/`.

**Модель:** Whisper Medium  
**Язык:** Русский  
**Оптимизация:** Поддержка GPU (CUDA, MPS для Apple Silicon)  
**Логирование:** Все обработанные файлы записываются в `processing.log.csv`

## Поддерживаемые форматы

- `.ogg` - Ogg Vorbis
- `.opus` - Opus
- `.mp3` - MP3
- `.wav` - WAV
- `.m4a` - M4A
- `.mp4` - MP4 (извлекается аудио)

## Требования

- Python 3.11.9
- ffmpeg (для конвертации аудио)
- macOS, Linux или Windows

Установка ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/YuliyaMinsk/whisper-transcriber.git
cd whisper-transcriber
```

2. Создайте виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Простая транскрипция (без определения спикеров)

1. Поместите аудиофайлы в папку `audio/`

2. Активируйте виртуальное окружение:
```bash
source .venv/bin/activate
```

3. Запустите транскрипцию:
```bash
python transcribe.py
```

4. Результаты будут сохранены в папку `transcripts/` в виде `.txt` файлов

**Примечание:** Скрипт автоматически определяет доступное устройство:
- MPS (GPU) для Apple Silicon
- CUDA для NVIDIA GPU
- CPU в остальных случаях

### Транскрипция с определением спикеров

Использует simple-diarizer для идентификации спикеров и openai-whisper для ASR.

1. Поместите аудиофайлы в папку `audio/`

2. Активируйте виртуальное окружение:
```bash
source .venv/bin/activate
```

3. Запустите транскрипцию с диаризацией:

```bash
# Автоопределение количества спикеров
python transcribe_diarize.py

# Явно указать количество спикеров
python transcribe_diarize.py --speakers 2
python transcribe_diarize.py -s 3

# Справка
python transcribe_diarize.py --help
```

4. Результаты будут сохранены в `transcripts/`:
   - `<имя_файла>.diarized.txt` — текст с временными метками и спикерами в формате `[HH:MM:SS.mmm–HH:MM:SS.mmm] SPEAKER_N: текст`
   - `<имя_файла>.diarized.json` — структурированные данные в JSON (start, end, speaker, text)

**Алгоритм назначения спикеров:**
- По середине сегмента ASR
- При неоднозначности — по максимальному перекрытию
- Fallback на SPEAKER_1 при отсутствии диаризации

## Структура проекта

```
whisper-transcriber/
├── README.md
├── requirements.txt
├── transcribe.py              # Простая транскрипция (Transformers Pipeline)
├── transcribe_diarize.py      # Транскрипция + диаризация (simple-diarizer + Whisper)
├── audio/
│   ├── README.md
│   ├── ARCHIVE/               # Архив обработанных файлов
│   └── (ваши аудиофайлы)
├── temp_audio/                # Временные WAV-файлы
├── transcripts/
│   ├── README.md
│   ├── ARCHIVE/               # Архив результатов
│   └── (результаты транскрипции)
├── pretrained_models/         # Предобученные модели для диаризации
└── .venv/
    └── (виртуальное окружение)
```

## Примечания

- При первом запуске Whisper загрузит модель Medium (~1.5GB)
- Модель оптимизирована для русского языка
- Для диаризации требуется ffmpeg и предобученные модели speechbrain (автоматически загружаются при первом запуске)
- Модель Whisper в `transcribe_diarize.py` загружается один раз и переиспользуется для всех файлов (оптимизация производительности)
- После обработки рекомендуется переместить файлы в `audio/ARCHIVE/` и `transcripts/ARCHIVE/`
- Все обработанные файлы логируются в `processing.log.csv` с временем обработки и параметрами
