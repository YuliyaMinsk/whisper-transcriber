# Whisper Transcriber

Автоматическая транскрипция аудиофайлов с использованием OpenAI Whisper и определением спикеров (speaker diarization).

## Описание

Этот проект позволяет автоматически транскрибировать аудиофайлы в текст с помощью модели Whisper от OpenAI. Поддерживается два режима работы:

- **Простая транскрипция** (`transcribe.py`) — только распознавание речи
- **Транскрипция с диаризацией** (`transcribe_diarize.py`) — распознавание речи + определение спикеров

Скрипты обрабатывают файлы из папки `audio/` и сохраняют результаты в папку `transcripts/`.

## Поддерживаемые форматы

- `.ogg` - Ogg Vorbis
- `.opus` - Opus
- `.mp3` - MP3
- `.wav` - WAV
- `.m4a` - M4A
- `.mp4` - MP4 (извлекается аудио)

### Требования

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

### Транскрипция с определением спикеров

1. Поместите аудиофайлы в папку `audio/`

2. Активируйте виртуальное окружение:
```bash
source .venv/bin/activate
```

3. (Опционально) Укажите количество спикеров в файле `transcribe_diarize.py`:
```python
NUM_SPEAKERS: Optional[int] = 3  # или 2, или None для автоопределения
```

4. Запустите транскрипцию с диаризацией:
```bash
python transcribe_diarize.py
```

5. Результаты будут сохранены в `transcripts/`:
   - `<имя_файла>.diarized.txt` — текст с временными метками и спикерами
   - `<имя_файла>.diarized.json` — структурированные данные в JSON

## Структура проекта

```
whisper-transcriber/
├── README.md
├── requirements.txt
├── transcribe.py              # Простая транскрипция
├── transcribe_diarize.py      # Транскрипция + диаризация
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
│   └── spkrec-xvect-voxceleb/
└── .venv/
    └── (виртуальное окружение)
```

## Примечания

- При первом запуске Whisper загрузит модель (~1GB)
- Модель оптимизирована для русского языка
- Для диаризации требуется ffmpeg и предобученные модели speechbrain
- После обработки рекомендуется переместить файлы в `audio/ARCHIVE/` и `transcripts/ARCHIVE/`
