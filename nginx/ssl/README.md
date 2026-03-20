# TLS 证书说明（开发环境）

Nginx 现已默认启用 `443`，并从本目录加载证书文件：

- `fullchain.pem`
- `privkey.pem`

本地开发可使用 `mkcert` 生成：

```bash
mkcert -install
mkcert -cert-file nginx/ssl/fullchain.pem -key-file nginx/ssl/privkey.pem localhost 127.0.0.1 ::1
```

生成后重启网关容器：

```bash
docker compose up -d --force-recreate actus-nginx
```
