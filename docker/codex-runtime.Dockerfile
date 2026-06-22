# syntax=docker/dockerfile:1
FROM node:22-slim

# Install system dependencies with cache mount and minimal packages
# Removed: build-essential (only needed for compiling, not runtime)
# Added: --no-install-recommends to skip extra packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    sudo \
    python3 \
    python3-dev \
    python3-minimal \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip3 /usr/local/bin/pip3
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV VIRTUAL_ENV=/opt/venv

# Install Astral uv and npm-backed agent CLIs in one layer
# Use npm cache mount for faster installs
RUN --mount=type=cache,target=/root/.npm \
    mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent playwright && \
    npx playwright install --with-deps chromium && \
    node --version && npm --version && npx --version && uv --version && uvx --version && python --version && python3 --version && pip --version && pip3 --version && codex --version && claude --version && pi --version

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 https://github.com/NousResearch/hermes-agent.git /usr/local/lib/hermes-agent && \
    cd /usr/local/lib/hermes-agent && \
    uv venv venv --python python3 && \
    uv pip install --python /usr/local/lib/hermes-agent/venv/bin/python -e . && \
    printf '#!/usr/bin/env bash\nunset PYTHONPATH\nunset PYTHONHOME\nexec /usr/local/lib/hermes-agent/venv/bin/hermes "$@"\n' > /usr/local/bin/hermes && \
    chmod +x /usr/local/bin/hermes && \
    (hermes --version || hermes --help >/dev/null)

# Bake the Langfuse Codex observability plugin into a staging CODEX_HOME so Teich
# can seed it into each session offline (no per-run network on the sandboxes).
# Only used when agent.codex.langfuse.enabled is set; side-channel/observability
# only. NOTE: pulls the plugin at its current HEAD -- the version is pinned at
# image-build time, not by a commit hash. chmod a+rX so the unprivileged in-
# container `codex` user (and the host `docker cp`) can read it.
RUN HOME=/opt/codex-langfuse CODEX_HOME=/opt/codex-langfuse/.codex \
    sh -c 'mkdir -p "$CODEX_HOME" \
      && codex plugin marketplace add langfuse/codex-observability-plugin \
      && codex plugin add tracing@codex-observability-plugin \
      && chmod -R a+rX /opt/codex-langfuse'

# Langfuse hook + SDK for Claude Code (cloned at HEAD). The SDK goes in the venv
# because Claude strips it from a hook's PATH, so the hook calls it by full path.
RUN /opt/venv/bin/pip install --no-cache-dir "langfuse>=4.0,<5" && \
    git clone --depth 1 https://github.com/langfuse/Claude-Observability-Plugin.git \
        /opt/claude-langfuse-plugin && \
    chmod -R a+rX /opt/claude-langfuse-plugin

# Create working directory and user in one layer
WORKDIR /workspace
RUN useradd -m -s /bin/bash codex && \
    mkdir -p /home/codex/.codex/sessions && \
    mkdir -p /home/codex/.claude && \
    mkdir -p /home/codex/.hermes && \
    printf 'codex ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/codex && \
    chmod 0440 /etc/sudoers.d/codex && \
    printf '#!/usr/bin/env bash\nexec sudo /usr/bin/apt-get "$@"\n' > /usr/local/bin/apt-get && \
    printf '#!/usr/bin/env bash\nexec sudo /usr/bin/apt "$@"\n' > /usr/local/bin/apt && \
    chmod +x /usr/local/bin/apt-get /usr/local/bin/apt && \
    chown -R codex:codex /home/codex /workspace ${PLAYWRIGHT_BROWSERS_PATH} ${VIRTUAL_ENV}

USER codex
ENV CODEX_HOME=/home/codex
ENV HOME=/home/codex
ENV NODE_PATH="/usr/local/lib/node_modules"
ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"

CMD ["bash"]
