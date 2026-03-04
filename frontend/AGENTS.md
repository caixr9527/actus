# 项目开发规范

## 技术栈

- **框架**: Next.js 16.1.6 (App Router)
- **语言**: TypeScript 5 (strict mode)
- **UI**: React 19.2.3
- **样式**: Tailwind CSS 4
- **组件库**: shadcn/ui (new-york style) + Radix UI
- **图标**: Lucide React

## 代码规范

### TypeScript

- 使用严格模式 (`strict: true`)
- 优先使用 `type` 而非 `interface`（除非需要扩展）
- 组件 props 使用 `Readonly<{}>` 或 `React.ComponentProps<"element">`
- 避免使用 `any`，使用 `unknown` 或具体类型
- 使用路径别名 `@/*` 指向 `./src/*`

### React/Next.js

- 使用函数式组件和 React Hooks
- 服务端组件优先，仅在需要时使用 `"use client"`
- 组件文件使用 PascalCase，如 `chat-message.tsx`
- 导出组件使用命名导出：`export { Component }`
- 使用 Next.js App Router 约定（`page.tsx`, `layout.tsx`）

### 组件开发

- UI 组件放在 `@/components/ui/`
- 业务组件放在 `@/components/`
- 使用 `cn()` 工具函数合并 Tailwind 类名
- 使用 `class-variance-authority` (CVA) 定义组件变体
- Radix UI 组件使用 `asChild` 模式支持组合
- 只使用 `shadcn` 库的组件，组件被安装在 `@/components/ui/` 下，如果没有请安装

### 样式规范

- 使用 Tailwind CSS 工具类
- 使用 `cn()` 合并类名：`cn(baseClasses, conditionalClasses)`
- 支持暗色模式（使用 `dark:` 前缀）
- 使用 CSS 变量（通过 `globals.css` 定义）

### 文件组织

- 页面：`src/app/**/page.tsx`
- 布局：`src/app/**/layout.tsx`
- 组件：`src/components/**/*.tsx`
- 工具函数：`src/lib/**/*.ts`
- Hooks：`src/hooks/**/*.ts`
- UI 设计稿：`docs/design/`（仅存放导出的参考图，不提交大体积源文件）
- 项目idea文档：`docs/prd/idea.md`

### 代码质量

- 遵循 ESLint 规则（Next.js core-web-vitals + TypeScript）
- 只改动必要的部分，优先复用现有成熟代码，避免重复造轮子。
- 架构设计时让边界情况自然融入常规逻辑，而不是单独打补丁。
- 单个代码文件不超过1000行，否则应当进行功能拆分。
- 保持代码简单直观，不过度设计复杂架构方案。
- 代码应表达实际逻辑，结构清晰，不保留不再使用的代码，不留无用的混淆项，避免未来维护困惑。
- 在代码文件中使用与用户沟通一致的语言撰写简明专业的代码注释。

### 导入顺序

1. React/Next.js 核心
2. 第三方库
3. 内部组件 (`@/components`)
4. 工具函数 (`@/lib`, `@/hooks`)
5. 类型定义
6. 样式文件

### 性能优化

- 使用 Next.js 自动代码分割
- 大组件使用动态导入 `dynamic()`
- 图片使用 `next/image`
- 避免不必要的客户端组件

## 后端接口规范

- **Base URL**: `http://localhost:23140/api`
- **OpenAPI 文档**: `http://localhost:23140/openapi.json`
- **统一响应格式**: `{ code: number, msg: string, data: T | null }`
