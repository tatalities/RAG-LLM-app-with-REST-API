FROM python:3.10-slim

# Set the working directory
WORKDIR /RAG-LLM

# Copy the project files
COPY . .

# Install Poetry
RUN pip install poetry

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install SQLite if required
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3430200.tar.gz \
    && tar -xvzf sqlite-autoconf-3430200.tar.gz \
    && cd sqlite-autoconf-3430200 \
    && ./configure \
    && make \
    && make install 

# Set the PATH to include Poetry
ENV PATH="${PATH}:/root/.local/bin"

# Install Python dependencies using Poetry
RUN poetry install

# Expose required ports
EXPOSE 8000 7860

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start FastAPI and Gradio
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
