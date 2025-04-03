# 🤖 FaceBot - Telegram бот для распознавания лиц

FaceBot — это умный Telegram-бот, который использует DeepFace и OpenCV для анализа лиц на фото.

## 🔥 Функционал
- 📷 Автоматическое распознавание лиц на загруженных фото
- 🧠 Использование нейросетевых моделей для сравнения лиц
- 📊 Сохранение данных пользователей в базе данных
- 🔗 Интеграция с Telegram через Telethon

## 🚀 Установка
1. **Клонируем репозиторий:**
   ```bash
   git clone https://github.com/Твой_Юзернейм/facebot.git
   cd facebot
   ```
2. **Создаём виртуальное окружение и устанавливаем зависимости:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux & macOS)
   venv\Scripts\activate  # (Windows)
   pip install -r requirements.txt
   ```
3. **Настроим переменные окружения:**
   ```bash
   export API_ID="your_api_id"
   export API_HASH="your_api_hash"
   export PHONE_NUMBER="your_phone"
   ```
   (Или создай `.env` с этими значениями)

4. **Запускаем бота:**
   ```bash
   python facebot.py
   ```

## 📜 Зависимости
- `Telethon` — для работы с Telegram API
- `DeepFace` — для распознавания лиц
- `OpenCV` — для обработки изображений
- `SQLite` — база данных пользователей

## 🛠 TODO
- [ ] Добавить поддержку видео
- [ ] Оптимизировать хранение эмбеддингов
- [ ] Улучшить логику базы данных

## 📄 Лицензия
Этот проект распространяется под MIT License.
