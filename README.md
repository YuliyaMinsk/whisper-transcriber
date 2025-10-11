# Whisper Transcriber

Автоматическая транскрипция аудиофайлов с использованием OpenAI Whisper.

## Описание

Этот проект позволяет автоматически транскрибировать аудиофайлы в текст с помощью модели Whisper от OpenAI. Скрипт обрабатывает файлы из папки `audio/` и сохраняет результаты в папку `transcripts/`.

## Поддерживаемые форматы

- `.ogg` - Ogg Vorbis
- `.opus` - Opus
- `.mp3` - MP3
- `.wav` - WAV
- `.m4a` - M4A
- `.mp4` - MP4 (извлекается аудио)

### Требования

- Python 3.11.9
- macOS, Linux или Windows

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

## Структура проекта

```
whisper-transcriber/
├── README.md
├── requirements.txt
├── transcribe.py
├── audio/
│   ├── README.md
│   └── (ваши аудиофайлы)
├── transcripts/
│   ├── README.md
│   └── (результаты транскрипции)
└── .venv/
    └── (виртуальное окружение)
```

## Примечания

- При первом запуске Whisper загрузит модель (~1GB)
- Модель оптимизирована для русского языка
