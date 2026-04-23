#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点裁剪与执行策略常量。"""

import re

# 技能子图标识
PLANNER_EXECUTE_STEP_SKILL_ID = "planner_react.execute_step"

# 消息窗口裁剪策略
MESSAGE_WINDOW_MAX_ITEMS = 100
MESSAGE_WINDOW_MAX_MESSAGE_CHARS = 500
MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS = 8

# 记忆候选筛选策略
MEMORY_CANDIDATE_MIN_CONFIDENCE = 0.3

# 会话摘要压缩策略
CONVERSATION_SUMMARY_MAX_PARTS = 4

# Prompt 上下文裁剪策略
PROMPT_CONTEXT_BRIEF_LIMIT = 3
PROMPT_CONTEXT_ARTIFACT_LIMIT = 6
PROMPT_CONTEXT_MEMORY_LIMIT = 6
PROMPT_CONTEXT_MESSAGE_LIMIT = 6
PROMPT_CONTEXT_COMPLETED_STEP_LIMIT = 5
PROMPT_CONTEXT_SUMMARY_MAX_CHARS = 400
PROMPT_CONTEXT_MESSAGE_MAX_CHARS = 200
PROMPT_CONTEXT_MEMORY_SUMMARY_MAX_CHARS = 160
PROMPT_CONTEXT_MEMORY_CONTENT_MAX_CHARS = 240
PROMPT_CONTEXT_OPEN_ITEM_LIMIT = 8

# 步骤执行治理策略
STEP_EXECUTION_TIMEOUT_SECONDS = 180 * 3

# 工具调用与工具循环治理策略
NOTIFY_USER_FUNCTION_NAME = "message_notify_user"
ASK_USER_FUNCTION_NAME = "message_ask_user"
TOOL_RESULT_MAX_TEXT_CHARS = 2400
TOOL_RESULT_MAX_LIST_ITEMS = 12
TOOL_RESULT_MAX_DICT_ITEMS = 12
BROWSER_HIGH_LEVEL_FUNCTION_NAMES: tuple[str, ...] = (
    "browser_read_current_page_structured",
    "browser_extract_main_content",
    "browser_extract_cards",
    "browser_find_link_by_text",
    "browser_find_actionable_elements",
)
BROWSER_ATOMIC_FUNCTION_NAMES: tuple[str, ...] = (
    "browser_view",
    "browser_navigate",
    "browser_restart",
    "browser_click",
    "browser_input",
    "browser_move_mouse",
    "browser_press_key",
    "browser_select_option",
    "browser_scroll_up",
    "browser_scroll_down",
    "browser_console_exec",
    "browser_console_view",
)
BROWSER_PROGRESS_FUNCTIONS: tuple[str, ...] = (
    "browser_read_current_page_structured",
    "browser_extract_main_content",
    "browser_extract_cards",
    "browser_find_actionable_elements",
    "browser_view",
    "browser_scroll_down",
    "browser_scroll_up",
)
REPEAT_TOOL_LIMIT = 2
SEARCH_REPEAT_LIMIT = 2
# P3-2A 收敛修复：对同一页面正文抓取指纹也做重复早停，避免 fetch_page 长循环空转。
FETCH_REPEAT_LIMIT = 2
BROWSER_NO_PROGRESS_LIMIT = 2
TOOL_FAILURE_LIMIT = 3
# 检索质量治理阈值
RESEARCH_MIN_FETCH_COUNT = 2
RESEARCH_MIN_DOMAIN_COUNT = 2
RESEARCH_PROGRESS_MAX_TRACK_ITEMS = 12
# P3-一次性收口：跨域抓取偏离先做可恢复拦截，超过阈值再收敛终止。
RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT = 2
# P3-2A 收敛修复：按任务模式限制单步骤工具轮次上限，防止默认上限过大导致耗时失控。
TASK_MODE_MAX_TOOL_ITERATIONS: dict[str, int] = {
    "human_wait": 2,
    "general": 6,
    "research": 8,
    "web_reading": 10,
    "browser_interaction": 10,
    "coding": 8,
    "file_processing": 6,
}

# 工具能力与任务模式白名单
SEARCH_FUNCTION_NAMES: tuple[str, ...] = ("search_web", "fetch_page")
READ_ONLY_FILE_FUNCTION_NAMES: tuple[str, ...] = (
    "read_file",
    "list_files",
    "find_files",
    "search_in_file",
)
FILE_FUNCTION_NAMES: tuple[str, ...] = (
    *READ_ONLY_FILE_FUNCTION_NAMES,
    "write_file",
    "replace_in_file",
)
TASK_MODE_ALLOWED_PREFIXES: dict[str, tuple[str, ...]] = {
    "research": (),
    "web_reading": (),
    "browser_interaction": (),
    "coding": ("shell_",),
    "file_processing": (),
    "human_wait": (),
    "general": ("shell_",),
}
TASK_MODE_ALLOWED_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "research": (
        *SEARCH_FUNCTION_NAMES,
        *READ_ONLY_FILE_FUNCTION_NAMES,
        ASK_USER_FUNCTION_NAME,
    ),
    "web_reading": (
        *SEARCH_FUNCTION_NAMES,
        *BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
        ASK_USER_FUNCTION_NAME,
    ),
    "browser_interaction": (
        *SEARCH_FUNCTION_NAMES,
        *BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
        *FILE_FUNCTION_NAMES,
        ASK_USER_FUNCTION_NAME,
    ),
    "coding": (
        *SEARCH_FUNCTION_NAMES,
        *FILE_FUNCTION_NAMES,
        ASK_USER_FUNCTION_NAME,
    ),
    "file_processing": (
        *FILE_FUNCTION_NAMES,
        "shell_execute",
        ASK_USER_FUNCTION_NAME,
    ),
    "human_wait": (
        ASK_USER_FUNCTION_NAME,
    ),
    "general": (
        *SEARCH_FUNCTION_NAMES,
        *FILE_FUNCTION_NAMES,
        ASK_USER_FUNCTION_NAME,
    ),
}

# 入口路由与任务模式判断规则
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<!\w)(/[A-Za-z0-9._/\-]+)")
SHELL_COMMAND_PATTERN = re.compile(
    r"(?:^|[\s`])(?:python3?|bash|sh|zsh|node|npm|pnpm|yarn|uv|pytest|git|make|docker|kubectl|cargo)\b",
    re.IGNORECASE,
)
CODE_BLOCK_PATTERN = re.compile(r"```|`[^`]+`", re.MULTILINE)
NUMBERED_LIST_PATTERN = re.compile(r"(^|\n)\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE)
SEQUENCE_PATTERN = re.compile(
    r"(然后|之后|接着|再|最后|随后|分步骤|step\s*\d|next|then|after that|finally)",
    re.IGNORECASE,
)
# 直接寒暄/应答型输入，优先走 direct_answer。
PHATIC_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|thanks|thank you|谢谢|再见|bye|早上好|晚上好|在吗|收到|ok|okay|好的)[!,.，。 ]*$",
    re.IGNORECASE,
)
CONTEXTUAL_FOLLOWUP_PATTERN = re.compile(
    r"(^(继续|展开|详细点|再详细一点|再补充|补充一下|更详细|具体点|再说说|换个说法|精简一下|改写一下)[!,.，。 ]*$)"
    r"|((不够|再|更|需要).{0,8}(详细|具体|完整|展开|补充))"
    r"|((继续|展开|补充|改写|精简).{0,8}(上面|上一轮|刚才|这个|该内容|原主题)?)"
    r"|(\b(continue|elaborate|more detail|expand|rewrite|summarize|shorten|add more)\b)",
    re.IGNORECASE,
)
# 等待语义拆成两层：
# 1. WAIT_PATTERN 识别用户原始消息里的“先确认/等待回复”这类直接等待语气；
# 2. WAIT_REQUEST_PATTERN 识别 Planner 产出的“向用户请求确认/请求用户选择/向用户询问”这类请求式文案。
# 两者最终都会被 _has_explicit_wait_semantics() 合并判断，分开定义只是为了区分来源语气，便于后续调试误判。
WAIT_PATTERN = re.compile(
    r"((先|需要|等待|等我|请先).{0,12}(确认|审批|同意|允许|选择|回复|补充|输入|澄清))"
    r"|((before|confirm|approval|approve|select|reply|input)\b)",
    re.IGNORECASE,
)
WAIT_REQUEST_PATTERN = re.compile(
    r"((请求|征求).{0,10}(确认|审批|同意|允许|选择|回复|补充|输入|澄清|意见|反馈))"
    r"|((向|请|让).{0,4}用户.{0,12}(确认|选择|回复|补充|输入|澄清|询问|提问|审批|同意|允许))"
    r"|((等待).{0,4}用户.{0,12}(确认|选择|回复|补充|输入|澄清|反馈))",
    re.IGNORECASE,
)
BROWSER_INTERACTION_PATTERN = re.compile(
    r"(登录|点击|滚动|输入|下拉|按钮|表单|提交|接管|验证码|上传|切换标签|勾选|同意)"
    r"|(\bbrowser_(click|input|move_mouse|press_key|select_option|scroll_up|scroll_down|console_exec)\b)",
    re.IGNORECASE,
)
WEB_READING_PATTERN = re.compile(
    r"(网页|页面|站点|官网|文章|正文|文档页|详情页|链接信息|页面内容|read page|page content)"
    r"|(\bbrowser_(view|navigate|restart|console_view|read_current_page_structured|extract_main_content|extract_cards|find_link_by_text|find_actionable_elements)\b)",
    re.IGNORECASE,
)
READ_ACTION_PATTERN = re.compile(
    r"(阅读|读取|查看.{0,8}(页面|正文|内容|文档)?|看一下.{0,8}(页面|正文|内容|文档)?|提取.{0,8}(正文|内容))"
    r"|(\b(read|fetch|inspect)\b)",
    re.IGNORECASE,
)
SEARCH_PATTERN = re.compile(
    r"(搜索|检索|查找|调研|搜集|资料|文章|网页内容|链接信息)"
    r"|(\b(search_web|fetch_page|search|research|lookup)\b)",
    re.IGNORECASE,
)
PLANNING_PATTERN = re.compile(
    r"(规划|计划|安排|方案|步骤|拆解|roadmap|plan\b|step\b)",
    re.IGNORECASE,
)
SYNTHESIS_PATTERN = re.compile(
    r"(整理|总结|归纳|提炼|梳理|关键点|要点|结论|摘要)",
    re.IGNORECASE,
)
COMPARISON_PATTERN = re.compile(
    r"(比较|对比|推荐|优缺点|区别|怎么选|选哪个)",
    re.IGNORECASE,
)
PLAN_ONLY_PATTERN = re.compile(
    r"((只|先).{0,6}(给|出|列).{0,6}(步骤|计划|方案|大纲))"
    r"|((先|暂时|现在).{0,8}(不要|别).{0,4}(执行|开始执行|跑起来))"
    r"|(\b(plan only|steps only|do not execute|don't execute)\b)",
    re.IGNORECASE,
)
FILE_PATTERN = re.compile(
    r"(文件|目录|日志|附件|read_file|write_file|list_files|find_files|replace_in_file|search_in_file)"
    r"|(\.(md|txt|json|yaml|yml|csv|log|py|ts|tsx|js|jsx)\b)",
    re.IGNORECASE,
)
CODING_PATTERN = re.compile(
    r"(代码|脚本|命令行|终端|测试|编译|运行命令|修复bug|修复测试|shell)"
    r"|(\bshell_[a-z_]+\b)",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(
    r"(搜索|读取|打开|访问|执行|运行|查看|创建|修改|写入|分析|调研|检索|浏览|打开页面)"
    r"|(\b(search|read|open|visit|run|execute|write|analyze|inspect|browse|fetch)\b)",
    re.IGNORECASE,
)
WRITE_ACTION_PATTERN = re.compile(
    r"(写入|改写|修改|替换|删除|创建文件|新建文件|覆盖|保存|落盘|执行命令|运行命令|shell_execute|write_file|replace_in_file|shell_)"
    r"|(\b(write|rewrite|modify|replace|delete|create|save|persist|execute command|run command|shell_execute)\b)",
    re.IGNORECASE,
)
WRITE_ACTION_DENY_PATTERN = re.compile(
    r"((不要|不需要|无需|别|禁止).{0,8}(写入|改写|修改|替换|删除|创建|执行命令|运行命令|落盘))"
    r"|(\b(do not|don't|no need to)\s+(write|modify|replace|delete|create|save|execute)\b)",
    re.IGNORECASE,
)
TOOL_REFERENCE_PATTERN = re.compile(
    r"\b(search_web|fetch_page|browser_[a-z_]+|read_file|write_file|list_files|find_files|replace_in_file|search_in_file|shell_[a-z_]+)\b",
    re.IGNORECASE,
)
EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN = re.compile(
    r"((保存|写入|导出|输出|生成|创建|落盘|整理).{0,10}(文件|文档|txt|md|markdown|json|csv))"
    r"|((文件|文档|txt|md|markdown|json|csv).{0,10}(保存|写入|导出|输出|生成|创建|落盘))"
    r"|(\b(save|export|write|generate|create)\b.{0,12}\b(file|document|markdown|json|csv|txt)\b)",
    re.IGNORECASE,
)
# P3-最终收口：replan 禁止生成“工具可用性测试/探活”类漂移步骤，避免偏离用户业务目标。
REPLAN_META_VALIDATION_STEP_PATTERN = re.compile(
    r"(工具可用性|工具是否可用|工具探活|测试工具|验证工具|检查工具|tool availability|tool usable|test tools?|validate tools?|smoke test)",
    re.IGNORECASE,
)
REPLAN_META_VALIDATION_ALLOW_PATTERN = re.compile(
    r"(验证工具|测试工具|检查工具可用性|tool availability|test tools?|validate tools?)",
    re.IGNORECASE,
)
REPLAN_META_VALIDATION_DENY_PATTERN = re.compile(
    r"((不要|不需要|无需|别|禁止).{0,6}(验证工具|测试工具|检查工具))"
    r"|(\b(do not|don't|no need to)\s+(test|validate|check)\s+tools?\b)",
    re.IGNORECASE,
)

# 附件交付偏好判定规则
ATTACHMENT_DELIVERY_DENY_PATTERN = re.compile(
    r"((不要|不需要|无需|不用|别).{0,8}(作为)?(.{0,4})?(最终)?附件)"
    r"|((不要|不需要|无需|不用|别).{0,8}(返回|交付).{0,4}附件)"
    r"|(\b(do not|don't|without|no)\s+(return\s+)?attachments?\b)",
    re.IGNORECASE,
)
ATTACHMENT_DELIVERY_ALLOW_PATTERN = re.compile(
    r"((作为|以).{0,4}(最终)?附件(.{0,4})?(返回|交付|提供)?)"
    r"|((附上|带上|返回).{0,4}附件)"
    r"|(\b(with|attach)\s+(it\s+as\s+)?attachments?\b)",
    re.IGNORECASE,
)
