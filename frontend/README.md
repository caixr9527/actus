# Actus Frontend

Actus 前端基于 Next.js 16 + React 19，负责会话交互、计划展示、工具预览与 VNC 远程界面承载。

## 1. 环境要求

- Node.js >= 20
- npm >= 10

## 2. 本地开发

```bash
cd frontend
npm install
npm run dev
```

默认访问：`http://localhost:3000`

## 3. 环境变量

前端通过 `NEXT_PUBLIC_API_BASE_URL` 指向后端 API 网关。

示例：

```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:23140/api
```

未配置时，默认值为 `http://localhost:23140/api`。

如果 `NEXT_PUBLIC_API_BASE_URL` 使用了跨域地址（例如前端 `http://localhost:3000` 调后端 `http://localhost:23140/api`），
需要保证后端 `.env` 满足以下条件，否则浏览器会直接拦截请求：

- `CORS_ALLOWED_ORIGINS` 包含前端页面来源（Origin）
- `CORS_ALLOW_CREDENTIALS=true`（认证请求会携带 HttpOnly Cookie）

示例：

```bash
# backend/.env
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CORS_ALLOW_CREDENTIALS=true
```

## 4. 常用命令

```bash
# 开发（默认使用 webpack，规避当前 Turbopack dev 解析问题）
npm run dev

# 可选：Turbopack 开发模式
npm run dev:turbo

# 生产构建
npm run build

# 启动生产服务
npm run start

# 静态检查
npm run lint

# Smoke baseline（最小可执行回归入口）
npm run test:smoke
# 等价于：npm run lint && npm run build
```

## 5. 快捷键

- `⌘/Ctrl + B`：折叠/展开侧边栏
- `⌘/Ctrl + K`：新建任务（跳转首页）
- `⌘/Ctrl + Enter`：发送消息

## 6. 常见问题

### 6.1 `npm run dev` 启动后出现样式依赖解析错误

当前默认脚本已切换到 `next dev --webpack`，如手动使用 `dev:turbo` 出现 `Can't resolve 'tailwindcss'`，请先回退到 `npm run dev`。

### 6.2 页面请求失败 / 数据为空

优先检查：

1. 后端是否已启动并可访问 `NEXT_PUBLIC_API_BASE_URL`
2. 浏览器 Network 中 `/api/...` 请求是否 2xx
3. 本地代理、跨域或网关配置是否正确
4. 若控制台提示 `CORS_ALLOWED_ORIGINS`，请将当前页面 Origin 加入后端白名单

## 7. CI 建议基线

最小质量闸门建议：

```bash
npm run test:smoke
```

后续可在此基础上增加单元测试与 E2E 测试。
