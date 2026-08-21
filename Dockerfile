FROM python:3.12-slim

WORKDIR /app

COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Not in requirements-core.txt: providers/anthropic_provider.py lazy-imports
# this SDK only when LLM_BACKEND=anthropic is selected at runtime. Installed
# unconditionally here so the Anthropic path works without a rebuild.
RUN pip install --no-cache-dir anthropic

COPY . .

ENV FLASK_APP=app.py
EXPOSE 5000

# app.py's own __main__ block calls app.run() with no host argument, which
# binds to 127.0.0.1 and would be unreachable through Docker's port mapping.
# Using the Flask CLI here instead of `python app.py` avoids touching that
# code: --host=0.0.0.0 is supplied at the CLI layer, not in the source.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
