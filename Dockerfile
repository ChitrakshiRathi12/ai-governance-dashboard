# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ───────────────────────────────────────────────────────────────────
LABEL maintainer="Chitrakshi Rathi"
LABEL description="AI Governance Dashboard — LLM output monitoring"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY app/       ./app/
COPY dashboard/ ./dashboard/
COPY data/      ./data/

# ── Create output directories ──────────────────────────────────────────────────
RUN mkdir -p reports

# ── Expose ports ───────────────────────────────────────────────────────────────
# 8000 = FastAPI, 8501 = Streamlit
EXPOSE 8000 8501

# ── Default command: run FastAPI ───────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
