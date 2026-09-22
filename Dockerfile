FROM python:3.12-slim

ENV HF_HOME=/models \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Models live on a volume so image updates never re-download them.
VOLUME ["/models"]

EXPOSE 8190

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8190"]
