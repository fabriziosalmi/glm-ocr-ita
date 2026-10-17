# glmocr with the GGUF reader on Linux, amd64 and arm64: llama.cpp's own server image, glmocr installed from uv.lock.
# The models are not in the image: mount the folder with the .gguf and its mmproj-*.gguf (docs/GGUF.md).
#   docker build -t glmocr .
#   docker run --rm -v "$PWD/work/gguf:/models:ro" -v "$PWD/pages:/pages" glmocr infer /pages/page.png --model /models/glm-ocr-ita-Q8_0.gguf
# Both bases are pinned by digest. llama.cpp b11706: the build of the published images closest to the one the GGUF was
# measured with (b11707, on macOS with Metal); the text the image returns is compared with those measurements before release.
FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM ghcr.io/ggml-org/llama.cpp:server-b11706@sha256:f41d47eee678c02656eefc9ed8cce76cab584aa7032805eeef0e6639fcfda485
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/glmocr
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY glmocr ./glmocr
# --frozen: exactly uv.lock, hashes checked; the MLX packages are left out by their markers (Apple Silicon only)
RUN UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3 UV_PYTHON_DOWNLOADS=never \
    uv sync --frozen --no-dev --no-editable --compile-bytecode \
    && rm -rf /root/.cache/uv /opt/glmocr   # the code is now a copy in /opt/venv
WORKDIR /data
ENV PATH=/opt/venv/bin:$PATH \
    GLMOCR_LLAMA_SERVER=/app/llama-server \
    GLMOCR_WORK=/tmp/glmocr \
    PYTHONUNBUFFERED=1
HEALTHCHECK NONE
ENTRYPOINT ["glmocr"]
CMD ["--help"]
