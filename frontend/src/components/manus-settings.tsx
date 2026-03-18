"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import {
  LayoutGrid,
  LayoutList,
  Loader2,
  Settings,
  Trash,
  Wrench,
} from "lucide-react"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Item,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemTitle,
} from "@/components/ui/item"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { configApi, getApiErrorMessage } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useI18n } from "@/lib/i18n"
import type {
  AgentConfig,
  ListMCPServerItem,
  ListA2AServerItem,
} from "@/lib/api"

// ==================== 通用配置 ====================

type CommonSettingProps = {
  config: AgentConfig
  onChange: (config: AgentConfig) => void
}

function CommonSetting({ config, onChange }: CommonSettingProps) {
  const { t } = useI18n()
  const handleChange = (field: keyof AgentConfig, value: string) => {
    const numValue = value === "" ? undefined : Number(value)
    onChange({ ...config, [field]: numValue })
  }

  return (
    <form className="w-full px-1" onSubmit={(e) => e.preventDefault()}>
      <FieldGroup>
        <FieldSet>
          <FieldLegend className="text-lg font-bold text-gray-700">
            {t("manusSettings.common.title")}
          </FieldLegend>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="max_iterations">{t("manusSettings.common.maxIterations.label")}</FieldLabel>
              <Input
                id="max_iterations"
                type="number"
                placeholder={t("manusSettings.common.maxIterations.placeholder")}
                value={config.max_iterations ?? 100}
                onChange={(e) => handleChange("max_iterations", e.target.value)}
                min={0}
                max={200}
              />
              <FieldDescription className="text-xs">
                {t("manusSettings.common.maxIterations.description")}
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel htmlFor="max_retries">{t("manusSettings.common.maxRetries.label")}</FieldLabel>
              <Input
                id="max_retries"
                type="number"
                placeholder={t("manusSettings.common.maxRetries.placeholder")}
                value={config.max_retries ?? 3}
                onChange={(e) => handleChange("max_retries", e.target.value)}
                min={0}
                max={10}
              />
              <FieldDescription className="text-xs">
                {t("manusSettings.common.maxRetries.description")}
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel htmlFor="max_search_results">{t("manusSettings.common.maxSearchResults.label")}</FieldLabel>
              <Input
                id="max_search_results"
                type="number"
                placeholder={t("manusSettings.common.maxSearchResults.placeholder")}
                value={config.max_search_results ?? 10}
                onChange={(e) =>
                  handleChange("max_search_results", e.target.value)
                }
                min={0}
                max={30}
              />
              <FieldDescription className="text-xs">
                {t("manusSettings.common.maxSearchResults.description")}
              </FieldDescription>
            </Field>
          </FieldGroup>
        </FieldSet>
      </FieldGroup>
    </form>
  )
}

// ==================== A2A Agent 配置 ====================

type A2ASettingProps = {
  servers: ListA2AServerItem[]
  loading: boolean
  onToggleEnabled: (id: string, enabled: boolean) => void
  onDelete: (id: string) => void
  onAdd: (baseUrl: string) => Promise<boolean>
}

function A2ASetting({
  servers,
  loading,
  onToggleEnabled,
  onDelete,
  onAdd,
}: A2ASettingProps) {
  const { t } = useI18n()
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [addUrl, setAddUrl] = useState("")
  const [adding, setAdding] = useState(false)

  const handleAdd = async () => {
    if (!addUrl.trim()) {
      toast.error(t("manusSettings.a2a.addAddressRequired"))
      return
    }
    setAdding(true)
    try {
      const success = await onAdd(addUrl.trim())
      if (success) {
        setAddUrl("")
        setAddDialogOpen(false)
      }
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="w-full px-1">
      <FieldGroup>
        <FieldSet>
          <FieldLegend className="w-full flex justify-between items-center text-lg font-bold text-gray-700">
            {t("manusSettings.a2a.title")}
            <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
              <DialogTrigger asChild>
                <Button type="button" size="xs" className="cursor-pointer">
                  {t("manusSettings.a2a.addAction")}
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle className="text-gray-700">
                    {t("manusSettings.a2a.addDialogTitle")}
                  </DialogTitle>
                  <DialogDescription className="text-gray-500">
                    {t("manusSettings.a2a.addDialogDescription")}
                  </DialogDescription>
                </DialogHeader>
                <form
                  className="w-full"
                  onSubmit={(e) => {
                    e.preventDefault()
                    handleAdd()
                  }}
                >
                  <FieldGroup>
                    <FieldSet>
                      <Field>
                        <Input
                          id="a2a_base_url"
                          type="url"
                          placeholder={t("manusSettings.a2a.addDialogPlaceholder")}
                          value={addUrl}
                          onChange={(e) => setAddUrl(e.target.value)}
                          disabled={adding}
                        />
                      </Field>
                    </FieldSet>
                  </FieldGroup>
                </form>
                <DialogFooter>
                  <DialogClose asChild>
                    <Button
                      variant="outline"
                      className="cursor-pointer"
                      disabled={adding}
                    >
                      {t("common.cancel")}
                    </Button>
                  </DialogClose>
                  <Button
                    className="cursor-pointer"
                    onClick={handleAdd}
                    disabled={adding}
                  >
                    {adding && <Loader2 className="animate-spin" />}
                    {t("common.add")}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription className="text-sm">
            {t("manusSettings.a2a.description")}
          </FieldDescription>

          {/* 加载态 */}
          {loading && (
            <div className="flex justify-center py-8">
              <Loader2 className="size-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {/* 空态 */}
          {!loading && servers.length === 0 && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              {t("manusSettings.a2a.empty")}
            </div>
          )}

          {/* 列表 */}
          {!loading && servers.length > 0 && (
            <ItemGroup className="gap-3">
              {servers.map((server) => (
                <Item key={server.id} variant="outline">
                  <ItemContent>
                    <ItemTitle className="w-full flex justify-between items-center text-md font-bold text-gray-700">
                      <div className="flex gap-2 items-center">
                        {server.name}
                        {!server.enabled && <Badge>{t("common.disabled")}</Badge>}
                      </div>
                      <div className="flex items-center justify-center gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          className="cursor-pointer"
                          onClick={() => onDelete(server.id)}
                        >
                          <Trash />
                        </Button>
                        <Switch
                          checked={server.enabled}
                          onCheckedChange={(checked) =>
                            onToggleEnabled(server.id, checked)
                          }
                        />
                      </div>
                    </ItemTitle>
                    {server.description && (
                      <ItemDescription>{server.description}</ItemDescription>
                    )}
                    <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <LayoutList size={12} />
                      {server.input_modes?.map((mode) => (
                        <Badge
                          key={`in-${mode}`}
                          variant="secondary"
                          className="text-gray-500"
                        >
                          {t("manusSettings.a2a.inputMode", { mode })}
                        </Badge>
                      ))}
                      {server.output_modes?.map((mode) => (
                        <Badge
                          key={`out-${mode}`}
                          variant="secondary"
                          className="text-gray-500"
                        >
                          {t("manusSettings.a2a.outputMode", { mode })}
                        </Badge>
                      ))}
                      <Badge
                        variant={server.streaming ? "secondary" : "outline"}
                        className={cn(
                          server.streaming ? "text-gray-500" : "text-gray-400"
                        )}
                      >
                        {t("manusSettings.a2a.streaming", {
                          status: server.streaming ? t("common.enabled") : t("common.disabled"),
                        })}
                      </Badge>
                      <Badge
                        variant={
                          server.push_notifications ? "secondary" : "outline"
                        }
                        className={cn(
                          server.push_notifications
                            ? "text-gray-500"
                            : "text-gray-400"
                        )}
                      >
                        {t("manusSettings.a2a.pushNotifications", {
                          status: server.push_notifications ? t("common.enabled") : t("common.disabled"),
                        })}
                      </Badge>
                    </ItemDescription>
                  </ItemContent>
                </Item>
              ))}
            </ItemGroup>
          )}
        </FieldSet>
      </FieldGroup>
    </div>
  )
}

// ==================== MCP 服务器 ====================

type MCPSettingProps = {
  servers: ListMCPServerItem[]
  loading: boolean
  onToggleEnabled: (serverName: string, enabled: boolean) => void
  onDelete: (serverName: string) => void
  onAdd: (config: string) => Promise<boolean>
}

function MCPSetting({
  servers,
  loading,
  onToggleEnabled,
  onDelete,
  onAdd,
}: MCPSettingProps) {
  const { t } = useI18n()
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [addConfig, setAddConfig] = useState("")
  const [adding, setAdding] = useState(false)

  const mcpConfigPlaceholder = `{
  "mcpServers": {
    "qiniu": {
      "command": "uvx",
      "args": [
        "qiniu-mcp-server"
      ],
      "env": {
        "QINIU_ACCESS_KEY": "YOUR_ACCESS_KEY",
        "QINIU_SECRET_KEY": "YOUR_SECRET_KEY"
      }
    }
  }
}`

  const handleAdd = async () => {
    if (!addConfig.trim()) {
      toast.error(t("manusSettings.mcp.addConfigRequired"))
      return
    }
    setAdding(true)
    try {
      const success = await onAdd(addConfig.trim())
      if (success) {
        setAddConfig("")
        setAddDialogOpen(false)
      }
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="w-full px-1">
      <FieldGroup>
        <FieldSet>
          <FieldLegend className="w-full flex justify-between items-center text-lg font-bold text-gray-700">
            {t("manusSettings.mcp.title")}
            <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
              <DialogTrigger asChild>
                <Button type="button" size="xs" className="cursor-pointer">
                  {t("manusSettings.mcp.addAction")}
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle className="text-gray-700">
                    {t("manusSettings.mcp.addDialogTitle")}
                  </DialogTitle>
                  <DialogDescription className="text-gray-500">
                    {t("manusSettings.mcp.addDialogDescription")}
                  </DialogDescription>
                </DialogHeader>
                <form
                  className="w-full"
                  onSubmit={(e) => {
                    e.preventDefault()
                    handleAdd()
                  }}
                >
                  <FieldGroup>
                    <FieldSet>
                      <Field>
                        <Textarea
                          id="mcp_config"
                          placeholder={mcpConfigPlaceholder}
                          value={addConfig}
                          onChange={(e) => setAddConfig(e.target.value)}
                          className="min-h-[200px] font-mono text-xs"
                          disabled={adding}
                        />
                      </Field>
                    </FieldSet>
                  </FieldGroup>
                </form>
                <DialogFooter>
                  <DialogClose asChild>
                    <Button
                      variant="outline"
                      className="cursor-pointer"
                      disabled={adding}
                    >
                      {t("common.cancel")}
                    </Button>
                  </DialogClose>
                  <Button
                    className="cursor-pointer"
                    onClick={handleAdd}
                    disabled={adding}
                  >
                    {adding && <Loader2 className="animate-spin" />}
                    {t("common.add")}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription className="text-sm">
            {t("manusSettings.mcp.description")}
          </FieldDescription>

          {/* 加载态 */}
          {loading && (
            <div className="flex justify-center py-8">
              <Loader2 className="size-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {/* 空态 */}
          {!loading && servers.length === 0 && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              {t("manusSettings.mcp.empty")}
            </div>
          )}

          {/* 列表 */}
          {!loading && servers.length > 0 && (
            <ItemGroup className="gap-3">
              {servers.map((server) => (
                <Item key={server.server_name} variant="outline">
                  <ItemContent>
                    <ItemTitle className="w-full flex justify-between items-center text-md font-bold text-gray-700">
                      <div className="flex gap-2 items-center">
                        {server.server_name}
                        <Badge>{server.transport}</Badge>
                        {!server.enabled && <Badge>{t("common.disabled")}</Badge>}
                      </div>
                      <div className="flex items-center justify-center gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          className="cursor-pointer"
                          onClick={() => onDelete(server.server_name)}
                        >
                          <Trash />
                        </Button>
                        <Switch
                          checked={server.enabled}
                          onCheckedChange={(checked) =>
                            onToggleEnabled(server.server_name, checked)
                          }
                        />
                      </div>
                    </ItemTitle>
                    {server.tools.length > 0 && (
                      <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <Wrench size={12} />
                        {server.tools.map((tool) => (
                          <Badge
                            key={tool}
                            variant="secondary"
                            className="text-gray-500"
                          >
                            {tool}
                          </Badge>
                        ))}
                      </ItemDescription>
                    )}
                  </ItemContent>
                </Item>
              ))}
            </ItemGroup>
          )}
        </FieldSet>
      </FieldGroup>
    </div>
  )
}

// ==================== 设置弹窗主组件 ====================

type SettingTab =
  | "common-setting"
  | "a2a-setting"
  | "mcp-setting"

const SETTING_MENUS: Array<{
  key: SettingTab
  icon: typeof Settings
  titleKey: string
}> = [
  { key: "common-setting", icon: Settings, titleKey: "manusSettings.menu.common" },
  { key: "a2a-setting", icon: LayoutGrid, titleKey: "manusSettings.menu.a2a" },
  { key: "mcp-setting", icon: Wrench, titleKey: "manusSettings.menu.mcp" },
]

type ManusSettingsProps = {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  showTrigger?: boolean
}

export function ManusSettings({
  open: openProp,
  onOpenChange,
  showTrigger = true,
}: ManusSettingsProps = {}) {
  const { t } = useI18n()
  // ---- 防止 SSR hydration 不匹配（Radix Dialog 在服务端/客户端生成不同的 aria-controls ID）----
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])

  // ---- 弹窗 & 导航 ----
  const [internalOpen, setInternalOpen] = useState(false)
  const open = openProp ?? internalOpen
  const setOpen = useCallback(
    (nextOpen: boolean) => {
      if (openProp === undefined) {
        setInternalOpen(nextOpen)
      }
      onOpenChange?.(nextOpen)
    },
    [openProp, onOpenChange],
  )
  const [activeSetting, setActiveSetting] =
    useState<SettingTab>("common-setting")

  // ---- 数据 ----
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({})
  const [mcpServers, setMcpServers] = useState<ListMCPServerItem[]>([])
  const [a2aServers, setA2aServers] = useState<ListA2AServerItem[]>([])

  // ---- 状态 ----
  const [loadingConfig, setLoadingConfig] = useState(false)
  const [loadingMCP, setLoadingMCP] = useState(false)
  const [loadingA2A, setLoadingA2A] = useState(false)
  const [saving, setSaving] = useState(false)

  // 防止 Strict Mode 重复获取
  const fetchingRef = useRef(false)

  // ---- 数据拉取（各接口独立请求、独立更新，互不阻塞） ----
  const fetchAllConfigs = useCallback(() => {
    if (fetchingRef.current) return
    fetchingRef.current = true

    // 1. Agent 配置（通常很快）
    setLoadingConfig(true)
    configApi
      .getAgentConfig()
      .then((agent) => {
        setAgentConfig(agent)
      })
      .catch((err) => {
        console.error("[Settings] 获取基础配置失败:", err)
      })
      .finally(() => {
        setLoadingConfig(false)
      })

    // 2. MCP 服务器列表（可能较慢）
    setLoadingMCP(true)
    configApi
      .getMCPServers()
      .then((data) => {
        setMcpServers(data?.mcp_servers ?? [])
      })
      .catch((err) => {
        console.error("[Settings] 获取 MCP 服务器列表失败:", err)
      })
      .finally(() => {
        setLoadingMCP(false)
      })

    // 3. A2A 服务器列表
    setLoadingA2A(true)
    configApi
      .getA2AServers()
      .then((data) => {
        setA2aServers(data?.a2a_servers ?? [])
      })
      .catch((err) => {
        console.error("[Settings] 获取 A2A 服务器列表失败:", err)
      })
      .finally(() => {
        setLoadingA2A(false)
      })
  }, [])

  // 弹窗打开时拉取数据
  useEffect(() => {
    if (open) {
      fetchAllConfigs()
    } else {
      // 弹窗关闭时重置 ref，下次打开可以重新获取
      fetchingRef.current = false
    }
  }, [open, fetchAllConfigs])

  // ---- 保存（通用配置） ----
  const handleSave = async () => {
    setSaving(true)
    try {
      if (activeSetting === "common-setting") {
        await configApi.updateAgentConfig(agentConfig)
        toast.success(t("manusSettings.toast.commonSaved"))
      }
    } catch (err) {
      toast.error(getApiErrorMessage(err, "manusSettings.toast.saveFailed", t))
    } finally {
      setSaving(false)
    }
  }

  // ---- MCP 操作 ----
  const handleMCPToggle = useCallback(
    async (serverName: string, enabled: boolean) => {
      // 乐观更新
      setMcpServers((prev) =>
        prev.map((s) => (s.server_name === serverName ? { ...s, enabled } : s)),
      )
      try {
        await configApi.updateMCPServerEnabled(serverName, enabled)
        toast.success(
          t("manusSettings.toast.toggleServerSuccess", {
            name: serverName,
            status: enabled ? t("common.enabled") : t("common.disabled"),
          }),
        )
      } catch {
        // 回滚
        setMcpServers((prev) =>
          prev.map((s) =>
            s.server_name === serverName ? { ...s, enabled: !enabled } : s,
          ),
        )
        toast.error(t("manusSettings.toast.operationFailed"))
      }
    },
    [t],
  )

  const handleMCPDelete = useCallback(
    async (serverName: string) => {
      const prev = mcpServers
      // 乐观更新
      setMcpServers((list) => list.filter((s) => s.server_name !== serverName))
      try {
        await configApi.deleteMCPServer(serverName)
        toast.success(t("manusSettings.toast.mcpDeleted", { name: serverName }))
      } catch {
        setMcpServers(prev)
        toast.error(t("manusSettings.toast.deleteFailed"))
      }
    },
    [mcpServers, t],
  )

  const handleMCPAdd = useCallback(
    async (configText: string): Promise<boolean> => {
      try {
        const parsed = JSON.parse(configText)
        await configApi.addMCPServer(parsed)
        toast.success(t("manusSettings.toast.mcpAdded"))
        // 重新拉取列表
        try {
          const data = await configApi.getMCPServers()
          setMcpServers(data?.mcp_servers ?? [])
        } catch {
          /* 忽略刷新失败 */
        }
        return true
      } catch (err) {
        if (err instanceof SyntaxError) {
          toast.error(t("manusSettings.toast.invalidJson"))
        } else {
          toast.error(getApiErrorMessage(err, "manusSettings.toast.addFailed", t))
        }
        return false
      }
    },
    [t],
  )

  // ---- A2A 操作 ----
  const handleA2AToggle = useCallback(
    async (id: string, enabled: boolean) => {
      setA2aServers((prev) =>
        prev.map((s) => (s.id === id ? { ...s, enabled } : s)),
      )
      try {
        await configApi.updateA2AServerEnabled(id, enabled)
        const server = a2aServers.find((s) => s.id === id)
        toast.success(
          t("manusSettings.toast.toggleServerSuccess", {
            name: server?.name ?? t("manusSettings.a2a.defaultAgentName"),
            status: enabled ? t("common.enabled") : t("common.disabled"),
          }),
        )
      } catch {
        setA2aServers((prev) =>
          prev.map((s) => (s.id === id ? { ...s, enabled: !enabled } : s)),
        )
        toast.error(t("manusSettings.toast.operationFailed"))
      }
    },
    [a2aServers, t],
  )

  const handleA2ADelete = useCallback(
    async (id: string) => {
      const prev = a2aServers
      const target = a2aServers.find((s) => s.id === id)
      setA2aServers((list) => list.filter((s) => s.id !== id))
      try {
        await configApi.deleteA2AServer(id)
        toast.success(
          t("manusSettings.toast.a2aDeleted", {
            name: target?.name ?? id,
          }),
        )
      } catch {
        setA2aServers(prev)
        toast.error(t("manusSettings.toast.deleteFailed"))
      }
    },
    [a2aServers, t],
  )

  const handleA2AAdd = useCallback(
    async (baseUrl: string): Promise<boolean> => {
      try {
        await configApi.addA2AServer({ base_url: baseUrl })
        toast.success(t("manusSettings.toast.a2aAdded"))
        // 重新拉取列表
        try {
          const data = await configApi.getA2AServers()
          setA2aServers(data?.a2a_servers ?? [])
        } catch {
          /* 忽略刷新失败 */
        }
        return true
      } catch (err) {
        toast.error(getApiErrorMessage(err, "manusSettings.toast.addFailed", t))
        return false
      }
    },
    [t],
  )

  // 客户端挂载前，仅渲染普通按钮占位，避免 Radix Dialog SSR hydration 不匹配
  if (!mounted) {
    if (!showTrigger) {
      return null
    }
    return (
      <Button variant="outline" size="icon-sm" className="cursor-pointer">
        <Settings />
      </Button>
    )
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {showTrigger ? (
        <DialogTrigger asChild>
          <Button variant="outline" size="icon-sm" className="cursor-pointer">
            <Settings />
          </Button>
        </DialogTrigger>
      ) : null}

      {/* 弹窗内容 */}
      <DialogContent className="!max-w-[850px]">
        {/* 头部 */}
        <DialogHeader className="border-b pb-4">
          <DialogTitle className="text-gray-700">{t("manusSettings.dialogTitle")}</DialogTitle>
          <DialogDescription className="text-gray-500">
            {t("manusSettings.dialogDescription")}
          </DialogDescription>
        </DialogHeader>

        {/* 中间主体 */}
        <div className="flex flex-row gap-4">
          {/* 左侧导航菜单 */}
          <div className="max-w-[180px]">
            <div className="flex flex-col gap-0">
              {SETTING_MENUS.map((menu) => (
                <Button
                  key={menu.key}
                  variant={activeSetting === menu.key ? "default" : "ghost"}
                  className="cursor-pointer justify-start"
                  onClick={() => setActiveSetting(menu.key)}
                >
                  <menu.icon />
                  {t(menu.titleKey)}
                </Button>
              ))}
            </div>
          </div>

          {/* 分隔符 */}
          <Separator orientation="vertical" />

          {/* 右侧内容 */}
          <div className="flex-1 h-[500px] scrollbar-hide overflow-y-auto">
            {loadingConfig && activeSetting === "common-setting" ? (
              <div className="flex justify-center items-center h-full">
                <Loader2 className="size-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                {activeSetting === "common-setting" && (
                  <CommonSetting
                    config={agentConfig}
                    onChange={setAgentConfig}
                  />
                )}
              </>
            )}
            {activeSetting === "a2a-setting" && (
              <A2ASetting
                servers={a2aServers}
                loading={loadingA2A}
                onToggleEnabled={handleA2AToggle}
                onDelete={handleA2ADelete}
                onAdd={handleA2AAdd}
              />
            )}
            {activeSetting === "mcp-setting" && (
              <MCPSetting
                servers={mcpServers}
                loading={loadingMCP}
                onToggleEnabled={handleMCPToggle}
                onDelete={handleMCPDelete}
                onAdd={handleMCPAdd}
              />
            )}
          </div>
        </div>

        {/* 底部按钮 */}
        <DialogFooter className="border-t pt-4">
          <DialogClose asChild>
            <Button variant="outline" className="cursor-pointer">
              {t("common.cancel")}
            </Button>
          </DialogClose>
          {activeSetting === "common-setting" ? (
            <Button
              className="cursor-pointer"
              disabled={saving}
              onClick={handleSave}
            >
              {saving && <Loader2 className="animate-spin" />}
              {t("common.save")}
            </Button>
          ) : null}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
