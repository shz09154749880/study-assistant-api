# Study Assistant API

یک پروژه‌ی ساده برای پاسخ به سوالات کاربران با استفاده از مدل‌های زبانی و روش‌های NLP.

##  ویژگی‌ها

- دریافت سوال کاربر از طریق API
- محاسبه‌ی embedding برای سوال و متون آموزشی با استفاده از OpenAI
- محاسبه شباهت برداری و یافتن نزدیک‌ترین متن
- تولید پاسخ از طریق مدل زبانی GPT
- لاگ‌گیری کامل برای عیب‌یابی بهتر

## اجرای محلی

### پیش‌نیازها:
- Python 3.9+
- کلید API از [OpenAI](https://platform.openai.com/account/api-keys)

### مراحل اجرا:

```bash
# نصب وابستگی‌ها
pip install -r requirements.txt

# ساخت فایل env و افزودن کلید OpenAI
cp .env.example .env

# اجرای اپلیکیشن
uvicorn main:app --reload
```

## اجرای با Docker

```bash
docker build -t study-assistant-api .
docker run -p 8000:8000 --env-file .env study-assistant-api
```

##  فایل Dockerfile:

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

##  توضیح کد

### `ask_question`
- ورودی: سوال کاربر
- پردازش: یافتن مشابه‌ترین متن با Embedding و محاسبه‌ی شباهت برداری
- خروجی: پاسخ تولید شده از GPT همراه با متن منبع

### `get_embedding`
- دریافت embedding برای متن با مدل `text-embedding-ada-002`

### `find_most_similar_passage`
- شباهت برداری با cosine similarity

### `generate_answer`
- تولید پاسخ با GPT با دادن context

##  ساختار خروجی API
```json
{
  "question": "سلول گیاهی چیست؟",
  "answer": "سلول‌های گیاهی دارای دیواره سلولی و کلروپلاست هستند و در فتوسنتز نقش دارند.",
  "source_passage": "..."
}
```

##  نویسنده
- [Zanganeh Shaghayegh]
