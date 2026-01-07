FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/requirements.lock.txt

COPY . /app

# Ensure entrypoint script is executable
RUN chmod +x /app/docker/entrypoint.sh

# IMPORTANT: use ENTRYPOINT, not CMD
ENTRYPOINT ["/app/docker/entrypoint.sh"]
