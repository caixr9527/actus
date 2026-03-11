# Backend 运行手册

## 1. 环境要求

- Python `>=3.12`
- PostgreSQL（默认连接：`127.0.0.1:5432/actus`）
- Redis（默认连接：`127.0.0.1:6379`）
- 腾讯云 COS 凭据（涉及文件上传/下载能力时必填）

## 2. 初始化

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. 关键环境变量

- `ENV`：运行环境（`development` / `production`）。
- `SQLALCHEMY_DATABASE_URI`：PostgreSQL 连接串。
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD`：Redis 连接信息。
- `AUTO_RUN_DB_MIGRATIONS`：启动时是否自动执行 Alembic 迁移（默认 `true`）。
- `CORS_ALLOWED_ORIGINS`：逗号分隔白名单。
- `CORS_ALLOW_CREDENTIALS`：为 `true` 时，`CORS_ALLOWED_ORIGINS` 不能包含 `*`。
- `APP_CONFIG_FILEPATH`：应用配置文件路径（默认 `config.yaml`）。

## 4. 启动方式

开发模式（热重载）：

```bash
cd backend
./dev.sh
```

生产模式（关闭热重载）：

```bash
cd backend
./run.sh
```

默认监听：`0.0.0.0:23140`。

## 5. 数据库迁移

自动迁移：启动时由 `AUTO_RUN_DB_MIGRATIONS=true` 触发。  
手动迁移：

```bash
cd backend
source .venv/bin/activate
alembic upgrade head
```

## 6. 健康检查与接口探活

- 健康检查：`GET /api/status`
- 示例：

```bash
curl -i http://127.0.0.1:23140/api/status
```

健康状态正常返回 `200`，依赖异常（如 Redis/Postgres）返回 `503`。

## 7. 日志位置

- 默认输出到标准错误（stderr）。
- 本地前台运行：直接查看终端日志。
- Docker 运行：通过容器日志查看（如 `docker logs <container>`）。

## 8. 常见故障排查

1. 启动时迁移失败：
   - 检查 `SQLALCHEMY_DATABASE_URI` 是否可达。
   - 确认数据库用户具备建表/扩展权限（`uuid-ossp`）。
2. Redis 健康检查失败：
   - 检查 `REDIS_HOST`、`REDIS_PORT`、`REDIS_PASSWORD`。
3. 启动时报 CORS 配置错误：
   - 当 `CORS_ALLOW_CREDENTIALS=true` 时，`CORS_ALLOWED_ORIGINS` 不能使用 `*`。
   - 生产环境禁止 `CORS_ALLOWED_ORIGINS=*`。
4. 文档页不可访问：
   - `production` 环境默认关闭 `/docs` 与 `/redoc`。

## 9. 测试

```bash
cd backend
source .venv/bin/activate
pytest -q
```
