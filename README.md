# Actus

## 项目介绍
Actus 是一个支持私有化部署的通用 AI Agent 系统，围绕「会话 + 计划 + 工具执行」的工作流设计。用户可以通过 Web 界面发起任务，系统会自动规划步骤并调用多种工具完成执行。

## 核心能力
- 多轮会话与实时反馈：支持会话创建、历史追踪与流式事件展示。
- 计划驱动执行：采用 Planner + ReAct 的协同流程进行任务拆解和逐步推进。
- 工具生态扩展：内置文件、Shell、浏览器、搜索等能力，并支持 MCP / A2A 外部能力接入。
- 沙箱隔离运行：通过独立 Sandbox 容器执行高风险操作，支持远程 VNC 预览。

## 技术架构
- 前端：Next.js（`frontend/`），负责会话交互、任务展示和工具预览。
- 后端：FastAPI（`backend/`），负责 Agent 编排、会话管理、工具调用与接口服务。
- 基础设施：Redis + PostgreSQL + Nginx + Docker Compose（`docker-compose.yml`），用于状态管理、数据存储与统一网关。

## 本地开发
- 前端本地启动文档：请参考 [frontend/README.md](./frontend/README.md)。
