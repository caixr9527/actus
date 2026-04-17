# 使用ubuntu基础镜像来源
FROM ubuntu:22.04

# 安装过程避免交互式提示并设置主机名
ENV DEBIAN_FRONTEND=noninteractive
ENV HOSTNAME=sandbox

# 将ubuntu默认的apt软件源地址统一换成阿里云镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list

# 更新和安装基础软件后并移除依赖
RUN apt-get update && apt-get install -y \
    sudo bc curl wget gnupg software-properties-common supervisor git \
    xterm socat xvfb x11vnc websockify\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建用户ubuntu并赋予sudo权限
RUN useradd -m -d /home/ubuntu -s /bin/bash ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

# 安装Python 3.10运行时，供 sandbox 主服务使用
RUN apt-get update && \
    apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 为pip3配置阿里云镜像源
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装node.js 24.12.0版本(LTS)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 将npm镜像源设置为阿里云镜像源
RUN npm config set registry https://registry.npmmirror.com

# 安装Google Chrome浏览器
RUN add-apt-repository ppa:xtradeb/apps -y && \
    apt-get update && \
    apt-get install -y chromium --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装中文字体和语言支持
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    language-pack-zh-hans \
    locales \
    && locale-gen zh_CN.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置默认编码
ENV LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8

# 安装SearXNG运行依赖，以及编译独立 Python 3.11 所需依赖
RUN apt-get update && apt-get install -y \
    python3-babel \
    build-essential \
    libbz2-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    tk-dev \
    uuid-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 为 SearXNG 单独编译安装 Python 3.11，避免依赖慢速 PPA
ARG PYTHON311_VERSION=3.11.14
RUN curl -fsSLo /tmp/Python-${PYTHON311_VERSION}.tar.xz https://www.python.org/ftp/python/${PYTHON311_VERSION}/Python-${PYTHON311_VERSION}.tar.xz && \
    mkdir -p /tmp/python-build && \
    tar -xJf /tmp/Python-${PYTHON311_VERSION}.tar.xz -C /tmp/python-build --strip-components=1 && \
    cd /tmp/python-build && \
    ./configure --prefix=/opt/python3.11 --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make install && \
    ln -sf /opt/python3.11/bin/python3.11 /usr/local/bin/python3.11 && \
    rm -rf /tmp/python-build /tmp/Python-${PYTHON311_VERSION}.tar.xz

# 安装SearXNG
ARG SEARXNG_REF=master
RUN useradd --system --home-dir /usr/local/searxng --shell /bin/bash searxng && \
    mkdir -p /usr/local/searxng /usr/local/searxng/run /etc/searxng && \
    for attempt in 1 2 3; do \
        curl -fsSLo /tmp/searxng.tar.gz https://codeload.github.com/searxng/searxng/tar.gz/refs/heads/${SEARXNG_REF} && \
        mkdir -p /usr/local/searxng/searxng-src && \
        tar -xzf /tmp/searxng.tar.gz -C /usr/local/searxng/searxng-src --strip-components=1 && \
        rm -f /tmp/searxng.tar.gz && \
        break; \
        echo "SearXNG源码下载失败，准备重试(${attempt}/3)"; \
        rm -rf /usr/local/searxng/searxng-src; \
        rm -f /tmp/searxng.tar.gz; \
        test "$attempt" -lt 3 || exit 1; \
        sleep 5; \
    done && \
    python3.11 -m venv /usr/local/searxng/searx-pyenv && \
    /usr/local/searxng/searx-pyenv/bin/pip install --no-cache-dir -U pip setuptools wheel granian pyyaml msgspec typing_extensions && \
    cd /usr/local/searxng/searxng-src && \
    /usr/local/searxng/searx-pyenv/bin/pip install --no-cache-dir --use-pep517 --no-build-isolation -e . && \
    chown -R searxng:searxng /usr/local/searxng /etc/searxng

# 设置工作空间
WORKDIR /sandbox

# 拷贝项目依赖文件到工作空间并安装依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 预装 Playwright Chromium 到 app 进程使用的 HOME，避免运行时找不到浏览器二进制
RUN mkdir -p /home/ubuntu/.cache/ms-playwright && \
    chown -R ubuntu:ubuntu /home/ubuntu/.cache && \
    su -s /bin/bash -c "HOME=/home/ubuntu PLAYWRIGHT_BROWSERS_PATH=/home/ubuntu/.cache/ms-playwright python3 -m playwright install chromium" ubuntu

# 拷贝当前整个项目文件
COPY . .

# 将supervisord.conf拷贝至supervisor服务配置目录中
COPY supervisord.conf /etc/supervisor/conf.d/sandbox.conf
COPY searxng/settings.yml /etc/searxng/settings.yml

# 暴露端口: 8081(FastAPI) 8082(SearXNG) 9222(CDP) 5900(vnc)
EXPOSE 8081 8082 9222 5900 5901

# 额外的环境变量配置
ENV CHROME_ARGS=""
ENV UVI_ARGS=""
ENV SEARXNG_SECRET=""
ENV PLAYWRIGHT_BROWSERS_PATH=/home/ubuntu/.cache/ms-playwright

# 使用supervisor管理所有进程
CMD ["supervisord", "-n", "-c", "/sandbox/supervisord.conf"]
