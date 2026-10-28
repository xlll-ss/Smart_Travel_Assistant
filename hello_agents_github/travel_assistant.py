"""基于本地 Ollama、实时工具和 RAG 的智能旅行助手 Web 应用。

这个文件是项目的应用入口，同时包含六层职责：
1. 数据模型：用 dataclass 描述偏好、地点、工具结果和结构化行程。
2. 数据存储：用 SQLite 保存用户偏好和可维护的地点目录。
3. Agent 工具：按需调用地理编码、天气、路线和旅行知识库。
4. Agent 编排：筛选地点、构造 messages、调用 LLM，并校验模型 JSON。
5. Web 界面：用 Gradio 把业务方法绑定到页面事件。
6. 启动与运维：读取 .env、配置日志、认证和服务参数。

一次提问的核心数据流是：
页面输入 -> TravelPreferences -> TravelToolRouter 决策 -> 按需调用工具
-> build_messages 组装上下文 -> Ollama 生成 JSON -> _parse_plan 程序校验
-> Markdown 展示 -> 将最近几轮写回当前浏览器会话历史。
"""

# ------------------------------ 标准库 ---------------------------------
# argparse：读取 --host、--port 等命令行启动参数。
import argparse
# hashlib：把用户名/会话 ID 哈希化，避免日志和文件名暴露原始身份。
import hashlib
# json：解析 LLM、HTTP 工具及 SQLite 标签字段中的 JSON 数据。
import json
# logging：记录请求、工具降级和异常，便于定位线上问题。
import logging
# os：读取 .env 已加载到进程中的环境变量。
import os
# re：识别工具意图、切分标签及清理文本。
import re
# sqlite3：持久化用户偏好和地点目录，无需额外数据库服务。
import sqlite3
# time：用 perf_counter 统计一次 Web 请求的耗时。
import time
# uuid：为每次页面操作生成可追踪、不会重复的错误编号。
import uuid
# Callable/Iterator：为回调工厂和数据库连接上下文提供类型标注。
from collections.abc import Callable, Iterator
# contextmanager：把 SQLite 连接的提交、回滚、关闭封装成 with 语句。
from contextlib import contextmanager
# dataclass：减少纯数据对象的样板代码；frozen=True 可防止中途误改事实数据。
from dataclasses import dataclass, field
# datetime/timezone：生成数据库更新时间和统一时间戳。
from datetime import datetime, timezone
# RotatingFileHandler：限制单个日志文件大小，并保留有限数量的历史日志。
from logging.handlers import RotatingFileHandler
# Path：跨平台地拼接项目、数据库、日志和上传文件路径。
from pathlib import Path
# Lock：保护共享 RAG/缓存实例，避免并发请求同时修改同一资源。
from threading import Lock
# urlencode：安全构造 Photon、Open-Meteo、OSRM 的查询参数。
from urllib.parse import urlencode
# Request/urlopen：使用标准库发送轻量 HTTP GET 请求，不额外引入 requests。
from urllib.request import Request, urlopen
# ZoneInfo：把实时工具报告固定显示为 Asia/Shanghai 时区。
from zoneinfo import ZoneInfo

# python-dotenv：启动时把项目根目录 .env 中的配置加载为环境变量。
from dotenv import load_dotenv

# HelloAgentsLLM：项目自己的统一 LLM 适配器，本应用通过它调用本地 Ollama。
from hello_agents.core.llm import HelloAgentsLLM
# BGEEmbedding：本地加载中文语义向量模型；EmbeddingBackend 便于测试时注入替身；
# RAGTool：统一封装结构化 PDF 解析、切块、向量检索和 SQLite；
# format_location：把同一模块返回的页码、章节和表格信息转换为引用位置。
from wise_douctment import BGEEmbedding, EmbeddingBackend, RAGTool, format_location


# ------------------------------ 全局配置 ---------------------------------
# 所有相对路径都锚定脚本目录，避免从不同终端目录启动时读错文件。
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
DATABASE_PATH = PROJECT_DIR / "travel_assistant.db"
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
USER_KNOWLEDGE_BASE_DIR = KNOWLEDGE_BASE_DIR / "users"
LOG_DIR = PROJECT_DIR / "logs"
TRAVEL_BUDGET_RANK = {"免费": 0, "经济": 1, "适中": 2, "宽松": 3}
TRAVEL_ACTIVITY_PREFERENCES = {"室内为主", "室内外皆可", "户外为主"}
LOGGER = logging.getLogger("travel_assistant")

##在服务器启动时，配置日志系统。它把日志同时输出到控制台和本地文件
def configure_logging(log_dir: str | Path = LOG_DIR) -> Path:
    """配置控制台和轮转文件日志，避免长期运行时日志无限增长。"""
    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "travel_assistant.log"
    level_name = os.getenv("TRAVEL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    LOGGER.setLevel(level)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
##如果代码跑了一年，日志文件会变成几十个 G，直接把硬盘撑爆。
# 这个配置保证了单文件达到 5MB 就自动切分，最多保留 5 个历史文件（总共占 25MB），旧的会自动被删除
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.getenv("TRAVEL_LOG_MAX_BYTES", "5242880")),
        backupCount=int(os.getenv("TRAVEL_LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    return log_path

##把用户的真实 ID（比如 auth:zhangsan）变成一串毫无意义的哈希值（比如 8f3a...）。
def user_log_id(user_id: str) -> str:
    """生成不可逆短标识，让日志可关联同一用户但不暴露用户名。"""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]

##从 .env 环境变量（TRAVEL_AUTH）里读取账号密码配置，并转化成 Gradio 登录组件能识别的元组列表
def parse_auth_config(value: str | None) -> list[tuple[str, str]] | None:
    """解析 Gradio 认证配置，把环境变量文本变成账号密码元组。"""
    if not value or not value.strip():
        return None
    credentials: list[tuple[str, str]] = []
    for item in value.split(","):
        username, separator, password = item.strip().partition(":")
        if not separator or not username.strip() or not password:
            raise ValueError("TRAVEL_AUTH 必须使用 user:password 格式，多个账号用逗号分隔")
        credentials.append((username.strip(), password))
    return credentials

##在多用户环境里，给当前访问者发一个唯一的“身份证号”（user_id）
##场景 A（强认证）：如果老板配置了 TRAVEL_AUTH，那进来的都是已登录用户，系统用 auth:xxx 作为 ID，换台电脑登录也能看到自己的知识库。
##场景 B（游客模式）：如果没配账号密码，系统用浏览器的 session_hash 生成 session:xxx。一旦你关掉浏览器，这个身份和里面的私有知识库就再也找不回来了。
def resolve_user_identity(
    username: str | None,
    session_hash: str | None,
) -> str:
    """生成隔离数据所需的用户 ID，优先使用登录用户，否则使用会话。"""
    if username and username.strip():
        digest = hashlib.sha256(username.strip().encode("utf-8")).hexdigest()[:40]
        return f"auth:{digest}"
    if session_hash and session_hash.strip():
        digest = hashlib.sha256(session_hash.strip().encode("utf-8")).hexdigest()[:40]
        return f"session:{digest}"
    raise RuntimeError("无法识别当前用户会话")

##当后台代码崩溃（抛出异常）时，决定给网页前端弹什么提示
def user_facing_error(exc: Exception, request_id: str) -> str:
    """隐藏内部异常细节，只向用户返回安全提示和可追踪编号。"""
    if isinstance(exc, (ValueError, LookupError)):
        return f"{exc}（错误编号：{request_id}）"
    return f"操作失败，请稍后重试。（错误编号：{request_id}）"


# ------------------------------ Agent 规则 --------------------------------
# SYSTEM_PROMPT 约束最终回答；程序稍后还会再次校验，不能只相信提示词。
SYSTEM_PROMPT = """你是一名谨慎、务实的中文旅行规划助手。

请根据用户的旅行偏好提供建议，并遵守以下规则：
1. 用户的活动偏好是硬约束。用户不喜欢户外或要求室内时，不得推荐主题公园、
   公园、徒步、游船、露天街区、户外观景等以户外活动为主的地点。
2. 系统提供候选地点时，只能推荐候选列表中的地点，不得自行补充其他地点。
3. 推荐不超过 3 个候选，并分别说明理由和建议停留时间。
4. 先给出明确结论，再说明推荐理由。
5. 只有“实时工具结果”中的天气、坐标、距离和时间可以当作实时事实；
   不要虚构票价、营业状态或工具没有提供的数据。
6. 信息无法确认或候选为空时，明确说明数据不足，不要编造地点。
7. 只输出一个合法 JSON 对象，不要输出 Markdown 代码块或额外说明。JSON 格式必须是：
   {"summary":"结论","itinerary":[{"period":"上午/下午/晚上",
   "place_name":"候选地点名","activity":"活动安排","duration_hours":2,
   "transport":"交通建议","estimated_cost":"费用说明"}],
   "budget_summary":"总预算说明","reminders":["提醒"]}
8. 可以引用实时工具明确提供的 OSRM 驾车距离和时间，但不得把它说成公交耗时；
   不得编造公交地铁线路、站名或工具未提供的距离。费用只能基于候选的消费等级描述，
   不得编造具体金额。
9. “旅行知识库检索结果”来自用户导入的攻略，只能作为参考资料，可能已经过时；
   使用其中信息时要说明来自攻略，涉及营业时间、票价和预约规则时提醒用户再次核实。
10. 只能使用本轮 Agent 工具决策实际提供的数据。未调用的工具视为没有相关数据，
    不得自行补充天气、路线、地点或知识库事实。
11. 如果程序候选地点为空，itinerary 必须严格返回空列表 []；问候、天气回答、
    攻略说明等非行程内容写入 summary，不得为了填充 itinerary 编造地点。
"""

# TOOL_ROUTER_PROMPT 让一次较小的 LLM 调用只负责选择工具，不直接回答问题。
TOOL_ROUTER_PROMPT = """你是旅行 Agent 的工具路由器，只负责判断工具需求。
请只返回一个合法 JSON 对象，格式为：
{"place_search":true,"weather":false,"routes":false,"knowledge":false,"reason":"简短原因"}

判断规则：
1. 用户要求推荐地点、选择去处或制定行程时，place_search=true。
2. 用户询问当前天气、气温、降雨，或明确要求结合天气时，weather=true。
3. 用户询问距离、驾车时间、路线或交通比较时，routes=true；routes=true 时 place_search 也必须为 true。
4. 用户明确询问攻略、导入文档、知识库、开放时间、预约、特色或注意事项时，knowledge=true。
5. 不要因为这是旅行应用就把所有值都设为 true，只选择回答本轮问题真正需要的工具。
6. 纯天气问题不需要地点检索；只有同时要求推荐地点或制定行程时才把 place_search 设为 true。
7. 问候、致谢和闲聊不需要任何工具，四个值都必须为 false。

示例：
- “杭州现在天气怎么样？” -> false, true, false, false
- “推荐一个室内地点” -> true, false, false, false
- “比较几个地点的驾车距离” -> true, false, true, false
- “根据导入攻略说明预约事项” -> false, false, false, true
- “你好” -> false, false, false, false
"""


def create_llm() -> HelloAgentsLLM:
    """集中创建 LLM 客户端，避免各业务类重复读取模型配置。"""
    model = os.getenv("LLM_MODEL_ID")
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    if not model:
        raise RuntimeError(".env 中缺少 LLM_MODEL_ID")

    return HelloAgentsLLM(
        provider="ollama",
        model=model,
        base_url=base_url,
        timeout=int(os.getenv("LLM_TIMEOUT", "180")),
    )


# ------------------------------ 核心数据模型 ------------------------------
# frozen=True 让请求期间的偏好保持只读，防止某个工具意外篡改用户约束。
#TravelPreferences Place都是用户与业务基础（输入与输入约束）
@dataclass(frozen=True)
class TravelPreferences:
    """统一承载页面输入，供筛选器、工具和 LLM 共享同一份用户约束。"""

    city: str
    mood: str
    activity_preference: str
    budget: str
    duration_hours: int
    companions: str
    departure: str = ""

    def validate(self) -> None:
        """在调用模型和外部工具前拦截非法输入，减少无效调用。"""
        if not self.city.strip():
            raise ValueError("城市不能为空")
        if self.activity_preference not in TRAVEL_ACTIVITY_PREFERENCES:
            raise ValueError("活动偏好必须是：室内为主、室内外皆可或户外为主")
        if self.budget not in TRAVEL_BUDGET_RANK:
            raise ValueError("预算必须是：免费、经济、适中或宽松")
        if not 1 <= self.duration_hours <= 24:
            raise ValueError("游玩时间必须在 1 到 24 小时之间")
        if len(self.departure.strip()) > 120:
            raise ValueError("出发地不能超过 120 个字符")
#to_context()：把字段拼成一段格式化文本，方便塞进大模型的 Prompt 里
    def to_context(self) -> str:
        """把结构化字段转换成提示词文本，但保留清晰的字段边界。"""
        return (
            f"城市：{self.city.strip()}\n"
            f"出发地：{self.departure.strip() or '未提供，使用城市中心'}\n"
            f"心情：{self.mood}\n"
            f"活动偏好：{self.activity_preference}\n"
            f"预算：{self.budget}\n"
            f"可用时间：{self.duration_hours} 小时\n"
            f"同行方式：{self.companions}"
        )

@dataclass(frozen=True)
#代表 SQLite 里的一条真实地点数据
class Place:
    """一条可信候选地点；LLM 只能从这些记录中选择，避免凭空编造地点。"""

    name: str
    city: str
    environment: str
    budget_level: str
    recommended_hours: float
    tags: tuple[str, ...]
    description: str
    place_id: int | None = None ##带有 place_id，用于在 CRUD 时精准定位（支持重命名）
#转换成一长串文本（如“名称：xxx；环境：室内...”），给大模型作为“候选白名单”。
    def to_context(self) -> str:
        """转换成提供给模型的候选地点文本。"""
        return (
            f"名称：{self.name}；环境：{self.environment}；"
            f"消费等级：{self.budget_level}；"
            f"建议时长：{self.recommended_hours:g} 小时；"
            f"特点：{'、'.join(self.tags)}；说明：{self.description}"
        )


##TravelToolDecision ItineraryStop TravelPlan TravelToolRouter 都是Agent 智能体决策与计划（大脑产物）
@dataclass(frozen=True)
class TravelToolDecision:
    """记录本轮需要哪些工具，使工具选择可检查、可展示、可测试。"""

    use_place_search: bool
    use_weather: bool
    use_routes: bool
    use_knowledge: bool
    reason: str
    source: str = "llm"
#把布尔值转成文字（如“天气：跳过”），告诉大模型这轮没查天气，别瞎编。
    def to_context(self) -> str:
        """转换为模型可读文本，让最终回答模型知道哪些事实允许使用。"""
        return (
            f"地点检索：{'调用' if self.use_place_search else '跳过'}；"
            f"天气：{'调用' if self.use_weather else '跳过'}；"
            f"路线：{'调用' if self.use_routes else '跳过'}；"
            f"知识库：{'调用' if self.use_knowledge else '跳过'}；"
            f"原因：{self.reason}"
        )
#给用户看的展示面板，显示“Agent 本轮调用了什么工具”。
    def to_markdown(self) -> str:
        """转换为用户可见说明，让工具调用过程保持透明。"""
        called = [
            label
            for enabled, label in (
                (self.use_place_search, "地点检索"),
                (self.use_weather, "天气"),
                (self.use_routes, "路线距离"),
                (self.use_knowledge, "旅行知识库"),
            )
            if enabled
        ]
        return (
            "### Agent 工具调用\n"
            f"- 本轮调用：{'、'.join(called) if called else '无需外部工具'}\n"
            f"- 决策依据：{self.reason}"
        )

##（单站行程） 作用：行程里的某一个时间段
@dataclass(frozen=True)
class ItineraryStop:
    """表示结构化行程的一个时段，便于程序校验时长、预算和地点。"""

    period: str
    place_name: str
    activity: str
    duration_hours: float
    transport: str
    estimated_cost: str

#（最终计划） 作用：大模型生成的、且通过了所有程序校验的最终旅行方案
@dataclass(frozen=True)
class TravelPlan:
    """保存通过程序约束校验的最终计划，并负责稳定地渲染到页面。"""

    summary: str
    itinerary: tuple[ItineraryStop, ...]
    budget_summary: str
    reminders: tuple[str, ...]

    @staticmethod
    #防止大模型生成的文本里包含 | 符号，导致 Markdown 表格错位。
    def _escape_table(value: str) -> str:
        """转义 Markdown 表格分隔符，防止模型文本破坏表格布局。"""
        return value.replace("|", "\\|").replace("\n", " ")

    def to_markdown(self) -> str:
        """把结构化计划渲染成适合聊天框显示的 Markdown。"""
        sections = ["### 行程结论", self.summary]
        if self.itinerary:
            table_lines = [
                "| 时段 | 地点 | 活动 | 时长 | 交通 | 预计费用 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            table_lines.extend(
                (
                    "| "
                    + " | ".join(
                        self._escape_table(value)
                        for value in (
                            stop.period,
                            stop.place_name,
                            stop.activity,
                            f"{stop.duration_hours:g} 小时",
                            stop.transport,
                            stop.estimated_cost,
                        )
                    )
                    + " |"
                )
                for stop in self.itinerary
            )
            sections.extend(["### 行程安排", "\n".join(table_lines)])
        sections.extend(["### 预算说明", self.budget_summary])
        if self.reminders:
            sections.append("### 出行提醒")
            sections.extend(f"- {reminder}" for reminder in self.reminders)
        return "\n\n".join(sections)

##在每次问答开始前，决定到底要调用哪些工具，不调用哪些工具。
class TravelToolRouter:
    """只负责工具选择，把“是否调用工具”从最终回答逻辑中分离。

    LLM 能理解自然语言中的复杂意图；关键词回退保证模型不可用时应用仍能
    完成基础路由。两者结合比直接在主流程里堆叠 if/else 更容易测试和扩展。
    """

    def __init__(self, llm: HelloAgentsLLM):
        """注入共享 LLM 客户端，避免路由器自己创建重复连接配置。"""
        self.llm = llm

    @staticmethod
    ##土办法（兜底规则）如果大模型挂了、网络断了、或者返回了一堆废话，就触发这个方法
    def _fallback(question: str) -> TravelToolDecision:
        """用确定性关键词完成最低可用路由，不依赖 LLM。"""
        text = question.casefold()
        use_weather = bool(
            re.search(r"天气|气温|温度|下雨|降雨|晴天|阴天|冷不冷|热不热", text)
        )
        use_routes = bool(
            re.search(r"路线|距离|多远|多久|驾车|开车|交通|怎么去|如何到达", text)
        )
        use_knowledge = bool(
            re.search(r"攻略|知识库|文档|资料|开放时间|营业时间|预约|注意事项|特色", text)
        )
        use_place_search = use_routes or bool(
            re.search(r"推荐|去哪|哪里|哪个地方|地点|行程|安排|规划|怎么玩", text)
        )
        return TravelToolDecision(
            use_place_search=use_place_search,
            use_weather=use_weather,
            use_routes=use_routes,
            use_knowledge=use_knowledge,
            reason="路由模型不可用，已使用关键词规则判断",
            source="fallback",
        )
##聪明办法（大模型决策）
    def decide(
        self,
        preferences: TravelPreferences,
        question: str,
        history: list[dict[str, str]],
    ) -> TravelToolDecision:
        """结合偏好、最近历史和本轮问题生成经过程序修正的工具计划。"""
        recent_history = "\n".join(
            f"{message['role']}：{message['content'][:500]}"
            for message in history[-4:]
        ) or "无"
        messages = [
            {"role": "system", "content": TOOL_ROUTER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"旅行偏好：\n{preferences.to_context()}\n\n"
                    f"最近对话：\n{recent_history}\n\n"
                    f"当前问题：\n{question.strip()}"
                ),
            },
        ]
        try:
            # temperature=0 降低随机性；工具路由需要稳定，不需要创意表达。
            raw = self.llm.invoke(
                messages,
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            payload = json.loads(raw)
            required = {"place_search", "weather", "routes", "knowledge"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError("工具决策缺少必要字段")
            if any(not isinstance(payload[key], bool) for key in required):
                raise ValueError("工具决策字段必须是布尔值")
            use_routes = payload["routes"]
            use_place_search = payload["place_search"] or use_routes

            # 程序规则是第二道护栏：即使模型误判，问候也不会触发昂贵工具。
            clean_question = question.strip().casefold()
            if re.fullmatch(
                r"(?:你好|您好|嗨|哈喽|hello|hi|谢谢|感谢|再见)[！!。.？?\s]*",
                clean_question,
            ):
                return TravelToolDecision(
                    use_place_search=False,
                    use_weather=False,
                    use_routes=False,
                    use_knowledge=False,
                    reason="当前是问候或闲聊，不需要调用旅行工具",
                )
            has_place_intent = bool(
                re.search(r"推荐|去哪|哪里|哪个地方|地点|行程|安排|规划|怎么玩", clean_question)
            )
            has_knowledge_intent = bool(
                re.search(
                    r"攻略|知识库|文档|资料|开放时间|营业时间|预约|注意事项|特色",
                    clean_question,
                )
            )
            if payload["weather"] and not use_routes and not has_place_intent:
                use_place_search = False

            return TravelToolDecision(
                use_place_search=use_place_search,
                use_weather=payload["weather"],
                use_routes=use_routes,
                use_knowledge=payload["knowledge"] and has_knowledge_intent,
                reason=str(payload.get("reason", "本地模型完成工具路由")).strip()
                or "本地模型完成工具路由",
            )
        except Exception as exc:
            LOGGER.warning(
                "event=tool_router_fallback error_type=%s",
                type(exc).__name__,
            )
            return self._fallback(question)

## GeoLocation、WeatherSnapshot、RouteEstimate（外部 API 结果）
#隔离外部 API 的原始 JSON 结构。不管 Photon 和 Open-Meteo 返回什么乱七八糟的格式，程序第一步就是把它们转成这三个极度干净的数据类

@dataclass(frozen=True)
class GeoLocation:
    """统一表示地理编码结果，隔离外部 API 的原始 JSON 结构。"""

    requested_name: str
    resolved_name: str
    latitude: float
    longitude: float

@dataclass(frozen=True)
class WeatherSnapshot:
    """保存天气快照，明确区分可验证工具事实和模型生成文字。"""

    observed_at: str
    condition: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_mm: float
    wind_speed_kmh: float

@dataclass(frozen=True)
class RouteEstimate:
    """保存两点间驾车估算，防止模型把它误写成公交或步行时间。"""

    origin: str
    destination: str
    distance_km: float
    duration_minutes: int

##TravelToolReport（实时工具报告） 把上面三个 API 结果集合起来，并带上 errors（降级错误记录）
@dataclass(frozen=True)
class TravelToolReport:
    """汇总本轮实时事实与降级错误，同时供 LLM 上下文和页面展示使用。"""

    generated_at: str
    weather_requested: bool
    routes_requested: bool
    locations: tuple[GeoLocation, ...]
    weather: WeatherSnapshot | None
    routes: tuple[RouteEstimate, ...]
    errors: tuple[str, ...]
##转成文字（“当前天气：晴...”），塞进 Prompt 给大模型
    def to_context(self) -> str:
        """转换成提供给 LLM 的事实文本。"""
        lines = [f"工具数据生成时间：{self.generated_at}"]
        if not self.weather_requested:
            lines.append("天气工具：本轮未调用。")
        elif self.weather is not None:
            weather = self.weather
            lines.append(
                "当前天气："
                f"{weather.condition}，{weather.temperature_c:g} 摄氏度，"
                f"体感 {weather.apparent_temperature_c:g} 摄氏度，"
                f"降水 {weather.precipitation_mm:g} 毫米，"
                f"风速 {weather.wind_speed_kmh:g} 千米/小时；"
                f"观测时间 {weather.observed_at}。"
            )
        else:
            lines.append("当前天气：工具未取得可用数据。")
        if self.locations:
            lines.append("地理位置解析：")
            lines.extend(
                f"- {location.requested_name} -> {location.resolved_name} "
                f"({location.latitude:.5f}, {location.longitude:.5f})"
                for location in self.locations
            )
        if not self.routes_requested:
            lines.append("路线工具：本轮未调用。")
        elif self.routes:
            lines.append("OSRM 驾车路线参考：")
            lines.extend(
                f"- {route.origin} -> {route.destination}："
                f"{route.distance_km:g} 千米，约 {route.duration_minutes} 分钟"
                for route in self.routes
            )
        else:
            lines.append("驾车路线：工具未取得可用数据。")
        if self.errors:
            lines.append("工具降级信息：")
            lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)
##转成表格，向用户展示真实的天气、路线距离和坐标。
    def to_markdown(self) -> str:
        """生成直接向用户展示的实时数据报告。"""
        sections = [
            "### 实时工具数据",
            f"更新时间：{self.generated_at}",
        ]
        if self.weather_requested and self.weather is not None:
            weather = self.weather
            sections.append(
                f"天气：{weather.condition}，{weather.temperature_c:g} 摄氏度，"
                f"体感 {weather.apparent_temperature_c:g} 摄氏度，"
                f"降水 {weather.precipitation_mm:g} 毫米，"
                f"风速 {weather.wind_speed_kmh:g} 千米/小时。"
            )
        if self.locations:
            location_lines = ["| 查询地点 | 解析结果 | 坐标 |", "| --- | --- | --- |"]
            location_lines.extend(
                f"| {location.requested_name} | {location.resolved_name} | "
                f"{location.latitude:.5f}, {location.longitude:.5f} |"
                for location in self.locations
            )
            sections.extend(["#### 地理位置", "\n".join(location_lines)])
        if self.routes_requested and self.routes:
            route_lines = [
                "| 起点 | 终点 | 驾车距离 | 驾车时间 |",
                "| --- | --- | --- | --- |",
            ]
            route_lines.extend(
                f"| {route.origin} | {route.destination} | "
                f"{route.distance_km:g} 千米 | 约 {route.duration_minutes} 分钟 |"
                for route in self.routes
            )
            sections.extend(["#### 路线距离", "\n".join(route_lines)])
        if self.errors:
            sections.extend(
                [
                    "#### 工具提示",
                    "\n".join(f"- {error}" for error in self.errors),
                ]
            )
        sources = ["Photon"]
        if self.weather_requested:
            sources.append("Open-Meteo")
        if self.routes_requested:
            sources.append("OSRM")
        source_note = f"数据源：{'、'.join(sources)}。"
        if self.routes_requested:
            source_note += "路线时间仅为驾车参考。"
        sections.append(source_note)
        return "\n\n".join(sections)



##KnowledgeSnippet 与 TravelKnowledgeReport（RAG 知识库结果）
@dataclass(frozen=True)
class KnowledgeSnippet:
    """统一表示一个 RAG 命中片段，并保留来源、位置和相关度。"""

    document_id: str
    source: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def location(self) -> str:
        """旧文本没有页码时返回空字符串，PDF 则展示物理页码与章节。"""
        return format_location(self.metadata)

@dataclass(frozen=True)
class TravelKnowledgeReport:
    """汇总本轮 RAG 证据，使回答能够展示引用来源而不是只给结论。"""

    snippets: tuple[KnowledgeSnippet, ...]
##带上来源标记（如 [攻略 1] 来源：xxx），喂给大模型。
    def to_context(self) -> str:
        """把检索结果转换成带来源的 LLM 上下文。"""
        if not self.snippets:
            return "本轮没有检索到相关旅行攻略。"
        return "\n\n".join(
            f"[攻略 {index}] 来源：{snippet.source}；片段："
            f"{snippet.chunk_index + 1}；"
            f"{snippet.location + '；' if snippet.location else ''}"
            f"相关度：{snippet.score:.3f}\n"
            f"{snippet.content}"
            for index, snippet in enumerate(self.snippets, start=1)
        )
##展示“旅行知识库参考”，告诉用户这段话来自哪份攻略，并提醒用户“攻略可能过时，请核实”。
    def to_markdown(self) -> str:
        """向用户展示本轮使用过的攻略来源和短摘要。"""
        if not self.snippets:
            return ""
        lines = ["### 旅行知识库参考"]
        for snippet in self.snippets:
            excerpt = re.sub(r"\s+", " ", snippet.content).strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:220].rstrip() + "..."
            lines.append(
                f"- **{snippet.source}**，片段 {snippet.chunk_index + 1}，"
                f"{snippet.location + '，' if snippet.location else ''}"
                f"相关度 {snippet.score:.3f}：{excerpt}"
            )
        lines.append("攻略内容可能过时，营业时间、票价和预约规则请在出发前核实。")
        return "\n".join(lines)

#tineraryStop GeoLocation TravelToolReport KnowledgeSnippet这些类都没有返回，
# 就单纯一个类通过@dataclass变成一个可以直接调用的函数用来统一格式
##@dataclass 把“相关的数据和操作数据的方法”打包在了一起，就像整理过的抽屉，找什么一目了然。
# 这就是为什么在大型系统开发中，哪怕最简单的一个响应包，都要专门定义一个类的根本原因


# ------------------------------ RAG 知识库 --------------------------------
#专门负责管理用户的私有知识库（RAG）
#TravelKnowledgeBase：单个用户的知识库引擎
##这个类本质上是对底层底层 RAGTool（来自 wise_douctment.py）的二次封装。它适配了旅行场景，并加了线程锁。
class TravelKnowledgeBase:
    """把通用 RAGTool 包装成旅行领域接口，避免页面依赖底层实现细节。

    这个适配层统一限制文件类型、检查返回状态并把字典转换成类型明确的
    TravelKnowledgeReport。以后更换向量库时，业务层只需修改这里。
    """

    ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

    def __init__(
        self,
        root: Path,
        namespace: str = "travel_guides",
        embedding: EmbeddingBackend | None = None,
    ):
        """在指定目录和命名空间中创建底层 RAG 实例及并发锁。"""
        self.rag_tool = RAGTool(
            knowledge_base_path=root,
            rag_namespace=namespace,
            embedding=embedding,
#这决定了底层 SQLite 数据库的文件名。如果 namespace 是 user_A，它就会创建并读取 user_A.db；
# 如果是 user_B，就是 user_B.db。这就实现了物理级别的数据隔离
        )
        #因为 Web 服务是多线程的，加入线程锁防止多个人同时往里写数据导致 SQLite 锁死
        self._lock = Lock()

    def import_file(self, file_path: str | Path) -> dict[str, object]:
        """导入 PDF、Markdown 或 TXT，按同名文档覆盖旧版本。"""
        path = Path(file_path)
        if path.suffix.lower() not in self.ALLOWED_SUFFIXES:
            raise ValueError("只支持 PDF、Markdown 和 TXT 文件")
        with self._lock:
            result = self.rag_tool.execute(
                "add_document",
                file_path=str(path),
                document_id=path.name,
                source_label=path.name,
            )
        if not result.get("success"):
            raise ValueError(str(result.get("error", "文档导入失败")))
        return result

    def add_guide(self, title: str, content: str) -> dict[str, object]:
        """把用户粘贴的旅行攻略直接写入知识库。"""
        clean_title = title.strip()
        clean_content = content.strip()
        if not clean_title:
            raise ValueError("攻略标题不能为空")
        if not clean_content:
            raise ValueError("攻略内容不能为空")
        if len(clean_title) > 120:
            raise ValueError("攻略标题不能超过 120 个字符")
        document_id = f"攻略：{clean_title}"
        with self._lock:
            result = self.rag_tool.execute(
                "add_text",
                text=clean_content,
                document_id=document_id,
            )
        if not result.get("success"):
            raise ValueError(str(result.get("error", "攻略保存失败")))
        return result

    def search(
        self,
        query: str,
        limit: int = 3,
        minimum_score: float = 0.08,##阈值防呆
    ) -> TravelKnowledgeReport:
        """检索相关攻略，并丢弃相关度过低的片段。"""
        with self._lock:
            results = self.rag_tool.search(query, limit=limit)
        snippets = tuple(
            KnowledgeSnippet(
                document_id=str(item["document_id"]),
                source=str(item["source"]),
                chunk_index=int(item["chunk_index"]),
                content=str(item["content"]),
                score=float(item["score"]),
                metadata=item.get("metadata", {}),
            )
            for item in results
            if float(item["score"]) >= minimum_score
        )
        return TravelKnowledgeReport(snippets=snippets)

    def list_documents(self) -> list[dict[str, object]]:
        """返回知识库中的文档及文本块数量。"""
        with self._lock:
            return self.rag_tool.execute("list_documents")

    def clear(self) -> str:
        """清空旅行知识库。"""
        with self._lock:
            return str(self.rag_tool.execute("clear"))

    def close(self) -> None:
        """关闭底层 SQLite 连接。"""
        with self._lock:
            self.rag_tool.connection.close()
##UserKnowledgeBaseManager：多用户隔离管家
class UserKnowledgeBaseManager:
    """为每个用户/会话维护独立 RAG 实例，防止不同用户的攻略互相泄露。"""

    def __init__(
        self,
        root: str | Path = USER_KNOWLEDGE_BASE_DIR,
        embedding: EmbeddingBackend | None = None,
    ):
        """准备用户知识库根目录、共享向量模型和进程内实例缓存。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 所有用户共享同一个只读模型，避免每个 SQLite 命名空间重复占用内存。
        self.embedding = embedding or BGEEmbedding()
##维护了一个字典 _instances，用来缓存每个用户的 TravelKnowledgeBase 实例，避免每次请求都重新打开一次数据库连接
        self._instances: dict[str, TravelKnowledgeBase] = {}
        self._lock = Lock()

    @staticmethod
##它不会把真实用户 ID 直接当文件名。而是用 SHA-256 哈希加密，截取前 24 位。这样即使黑客摸到了服务器，看到了 travel_guides_ab12...db 文件，也不知道这是谁。
    def _namespace(user_id: str) -> str:
        """把用户身份转换成安全、稳定且不含原始用户名的数据库命名空间。"""
        clean_user_id = TravelPreferenceStore._validate_user_id(user_id)
        digest = hashlib.sha256(clean_user_id.encode("utf-8")).hexdigest()[:24]
        return f"travel_guides_{digest}"
##这里用之前的_instances缓存字典来调用知识库，判断是否存在
    def for_user(self, user_id: str) -> TravelKnowledgeBase:
        """复用同一用户的实例，避免每次请求重复打开 SQLite 连接。"""
        namespace = self._namespace(user_id)
        with self._lock:
            knowledge_base = self._instances.get(namespace)
            if knowledge_base is None:
                knowledge_base = TravelKnowledgeBase(
                    self.root,
                    namespace=namespace,
                    embedding=self.embedding,
                )
                self._instances[namespace] = knowledge_base
            return knowledge_base

    def close_all(self) -> None:
        """关闭所有缓存实例，供服务优雅退出或测试清理使用。"""
        with self._lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for knowledge_base in instances:
            knowledge_base.close()


# ------------------------------ 实时旅行工具 ------------------------------
##这是一个纯工具函数，不包含任何业务逻辑，只负责用 Python 标准库 urllib 发送 HTTP GET 请求，并把返回的字节流解析成 Python 字典
def _fetch_json(url: str, timeout: float) -> object:
    """统一外部 HTTP 请求的超时、User-Agent 和 JSON 解析行为。"""
    request = Request(
        url,
        headers={"User-Agent": "hello-agents-travel-assistant/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)
##定位仪（查经纬度） 把人类能看懂的地名（如“杭州图书馆”），变成机器能看懂的经纬度，在Phonto查
class GeocodingTool:
    """调用 Photon 把地点名转换为坐标，供天气和路线工具继续使用。

    天气和路线 API 不理解“杭州图书馆”这样的地址，所以必须先做地理编码。
    内存缓存还能减少同一地点在连续问答中的网络请求。
    """

    API_URL = "https://photon.komoot.io/api/"

    def __init__(self, timeout: float = 8.0):
        """保存网络超时，并初始化线程安全的地点坐标缓存。"""
        self.timeout = timeout
##缓存机制：它内部有一个 _cache 字典。如果 A 用户刚查过杭州，B 用户接着查杭州，直接从内存返回，不再发网络请求。
#缓存的是完整GeoLocation对象，包括了“地名+坐标”映射关系。
        self._cache: dict[str, GeoLocation] = {}
#加上锁，保证多线程安全
        self._cache_lock = Lock()

    def geocode(self, requested_name: str, query: str) -> GeoLocation:
        """解析一个查询并返回统一坐标对象，优先复用本地缓存。"""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("地理编码查询不能为空")
        cache_key = clean_query.casefold()##将字符串转换为用于“无大小写比较”的格式
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return GeoLocation(
                requested_name=requested_name,
                resolved_name=cached.resolved_name,
                latitude=cached.latitude,
                longitude=cached.longitude,
            )

        url = self.API_URL + "?" + urlencode(
            {"q": clean_query, "limit": 1, "lang": "en"}
        )
        payload = _fetch_json(url, self.timeout)
        if not isinstance(payload, dict) or not payload.get("features"):
            raise LookupError(f"没有找到地理位置：{requested_name}")
        feature = payload["features"][0]
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        properties = feature.get("properties", {})
        if len(coordinates) < 2:
            raise ValueError(f"地理位置缺少坐标：{requested_name}")
        resolved_parts = [
            properties.get("name"),
            properties.get("city"),
            properties.get("state"),
            properties.get("country"),
        ]
        resolved_name = " / ".join(
            str(part) for part in dict.fromkeys(resolved_parts) if part
        )
        cached_location = GeoLocation(
            requested_name=clean_query,
            resolved_name=resolved_name or clean_query,
            latitude=float(coordinates[1]),
            longitude=float(coordinates[0]),
        )
        with self._cache_lock:
            self._cache[cache_key] = cached_location
        return GeoLocation(
            requested_name=requested_name,
            resolved_name=cached_location.resolved_name,
            latitude=cached_location.latitude,
            longitude=cached_location.longitude,
        )
##接收一个 GeoLocation 坐标，去 Open-Meteo 查当前天气
class WeatherTool:
    """调用 Open-Meteo 获取当前天气，让模型使用实时事实而不是猜测。"""

    API_URL = "https://api.open-meteo.com/v1/forecast"
    WEATHER_CODES = {
        0: "晴",
        1: "大致晴朗",
        2: "局部多云",
        3: "阴",
        45: "雾",
        48: "雾凇",
        51: "小毛毛雨",
        53: "毛毛雨",
        55: "强毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "小阵雨",
        81: "阵雨",
        82: "强阵雨",
        95: "雷暴",
        96: "雷暴伴小冰雹",
        99: "雷暴伴强冰雹",
    }

    def __init__(self, timeout: float = 8.0):
        """保存天气 API 的单次请求超时。"""
        self.timeout = timeout

    def get_current(self, location: GeoLocation) -> WeatherSnapshot:
        """读取指定坐标的当前天气并转换成稳定的数据模型。"""
        url = self.API_URL + "?" + urlencode(
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": (
                    "temperature_2m,apparent_temperature,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "Asia/Shanghai",
            }
        )
        payload = _fetch_json(url, self.timeout)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("current"), dict
        ):
            raise ValueError("天气服务没有返回 current 数据")
        current = payload["current"]
        weather_code = int(current.get("weather_code", -1))
##天气 API 返回的天气代码是数字（比如 0 代表晴，61 代表小雨）。内置了 WEATHER_CODES 字典，把数字翻译成人类能看懂的中文
        return WeatherSnapshot(
            observed_at=str(current.get("time", "未知")),
            condition=self.WEATHER_CODES.get(
                weather_code,
                f"未知天气代码 {weather_code}",
            ),
            temperature_c=float(current["temperature_2m"]),
            apparent_temperature_c=float(current["apparent_temperature"]),
            precipitation_mm=float(current["precipitation"]),
            wind_speed_kmh=float(current["wind_speed_10m"]),
        )
##导航仪（算距离和时间）调用OSRM 这个 API 极其强大，它不需要你发 3 次请求，而是直接返回一个 矩阵（Matrix）
##接收一组坐标（比如 1 个起点 + 3 个候选地点），计算它们两两之间的驾车距离和驾车时间
class RouteTool:
    """调用 OSRM Table API 批量计算驾车距离和时间，减少网络往返。"""

    API_URL = "https://router.project-osrm.org/table/v1/driving/"

    def __init__(self, timeout: float = 8.0):
        """保存路线 API 的单次请求超时。"""
        self.timeout = timeout

    def get_estimates(
        self,
        locations: list[GeoLocation],
    ) -> list[RouteEstimate]:
        """一次请求多个坐标的矩阵，并展开为两两之间的路线估算。"""
        if len(locations) < 2:
            return []
        coordinates = ";".join(
            f"{location.longitude:.6f},{location.latitude:.6f}"
            for location in locations
        )
        url = (
            self.API_URL
            + coordinates
            + "?"
            + urlencode({"annotations": "distance,duration"})
        )
        payload = _fetch_json(url, self.timeout)
        if not isinstance(payload, dict) or payload.get("code") != "Ok":
            raise ValueError("路线服务没有返回有效矩阵")
        distances = payload.get("distances")
        durations = payload.get("durations")
        if not isinstance(distances, list) or not isinstance(durations, list):
            raise ValueError("路线服务缺少距离或时间矩阵")
##通过双层 for 循环，从矩阵里把数据抠出来，转成一个 RouteEstimate 对象的列表（比如“起点 -> 博物馆，5公里，15分钟”）。
        estimates: list[RouteEstimate] = []
        for origin_index in range(len(locations)):
            for destination_index in range(origin_index + 1, len(locations)):
                distance = distances[origin_index][destination_index]
                duration = durations[origin_index][destination_index]
                if distance is None or duration is None:
                    continue
                estimates.append(
                    RouteEstimate(
                        origin=locations[origin_index].requested_name,
                        destination=locations[destination_index].requested_name,
                        distance_km=round(float(distance) / 1000, 1),
                        duration_minutes=max(1, round(float(duration) / 60)),
                    )
                )
        return estimates
##它自己不发任何网络请求，它的作用就是按需调度上面三个工具，并处理报错。
class RealTimeTravelTools:
    """编排三个实时工具，并把局部失败转换为可展示的降级信息。

    单个 API 暂时不可用时，不应该让整个旅行问答失败，因此这里捕获每项工具
    的异常并继续执行其他工具，最后统一返回 TravelToolReport。
    """

    def __init__(
        self,
        geocoding: GeocodingTool,
        weather: WeatherTool,
        routes: RouteTool,
    ):
        """注入三个单一职责工具，方便测试时替换为离线假实现。"""
        self.geocoding = geocoding
        self.weather = weather
        self.routes = routes

    @classmethod
    def create_default(cls) -> "RealTimeTravelTools":
##create_default 是一个类方法（Classmethod），它的核心作用是：快速创建一个带有默认配置的 RealTimeTravelTools 实例
#要实例化直接realtime_tools = RealTimeTravelTools.create_default()
        """使用同一个环境变量超时创建默认工具组合。"""
        timeout = float(os.getenv("TRAVEL_TOOL_TIMEOUT", "8"))
        return cls(
            GeocodingTool(timeout),
            WeatherTool(timeout),
            RouteTool(timeout),
        )
##按需执行：看路由器的决定。如果路由器说“这轮只查天气”，它就只调 WeatherTool，连 RouteTool 的代码都不碰。
    def collect(
        self,
        preferences: TravelPreferences,
        candidates: list[Place],
        include_weather: bool = True,
        include_routes: bool = True,
    ) -> TravelToolReport:
        """按路由决策收集实时快照，返回包含事实和错误的统一报告。"""
        generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        errors: list[str] = []
        locations: list[GeoLocation] = []

        city_location: GeoLocation | None = None
        #只要这轮要查天气或路线，第一件事绝对是拿城市的经纬度
        if include_weather or include_routes:
            try:
                #拿城市坐标
                city_location = self.geocoding.geocode(
                    f"{preferences.city}市",
                    preferences.city,
                )
                locations.append(city_location)
            except Exception as exc:
                errors.append(f"城市地理解析失败：{exc}")
                LOGGER.warning("event=geocoding_failed target=city error_type=%s", type(exc).__name__)

        origin_location: GeoLocation | None = None
        candidate_locations: list[GeoLocation] = []
        if include_routes:
            # 路线计算需要明确起点；“市中心”可直接复用城市坐标。
            departure_name = preferences.departure.strip()
            generic_departures = {
                preferences.city,
                f"{preferences.city}市",
                f"{preferences.city}中心",
                f"{preferences.city}市中心",
                "市中心",
            }
            if departure_name in generic_departures and city_location is not None:
                origin_location = GeoLocation(
                    requested_name=f"出发地：{departure_name}",
                    resolved_name=city_location.resolved_name,
                    latitude=city_location.latitude,
                    longitude=city_location.longitude,
                )
                locations.append(origin_location)
            elif departure_name:
                try:
                    origin_location = self.geocoding.geocode(
                        f"出发地：{departure_name}",
                        f"{departure_name} {preferences.city}",
                    )
                    locations.append(origin_location)
                except Exception as exc:
                    errors.append(f"出发地地理解析失败：{exc}")
                    LOGGER.warning(
                        "event=geocoding_failed target=departure error_type=%s",
                        type(exc).__name__,
                    )
            elif city_location is not None:
                origin_location = GeoLocation(
                    requested_name=f"出发地：{preferences.city}市中心",
                    resolved_name=city_location.resolved_name,
                    latitude=city_location.latitude,
                    longitude=city_location.longitude,
                )
                locations.append(origin_location)

            # OSRM 只接受坐标，因此候选地点也需要逐个完成地理编码。
            for place in candidates:
                try:
                    location = self.geocoding.geocode(
                        place.name,
                        f"{place.name} {place.city}",
                    )
                    locations.append(location)
                    candidate_locations.append(location)
                except Exception as exc:
                    errors.append(f"地点“{place.name}”地理解析失败：{exc}")
                    LOGGER.warning(
                        "event=geocoding_failed target=candidate error_type=%s",
                        type(exc).__name__,
                    )

        # 天气只需要一个城市或出发地坐标，不依赖候选地点。
        weather_snapshot: WeatherSnapshot | None = None
        weather_location = city_location or origin_location
        if include_weather and weather_location is not None:
            try:
                weather_snapshot = self.weather.get_current(weather_location)
            except Exception as exc:
                errors.append(f"天气工具失败：{exc}")
                LOGGER.warning("event=weather_failed error_type=%s", type(exc).__name__)
        elif include_weather:
            errors.append("天气工具缺少可用城市坐标")

        # 只有起点和至少一个候选地点都解析成功，路线矩阵才有意义。
        route_estimates: list[RouteEstimate] = []
        if include_routes and origin_location is not None and candidate_locations:
            try:
                route_estimates = self.routes.get_estimates(
                    [origin_location, *candidate_locations]
                )
            except Exception as exc:
                errors.append(f"路线工具失败：{exc}")
                LOGGER.warning("event=route_failed error_type=%s", type(exc).__name__)
        elif include_routes:
            errors.append("路线工具缺少出发地或候选地点坐标")

        return TravelToolReport(
            generated_at=generated_at,
            weather_requested=include_weather,
            routes_requested=include_routes,
            locations=tuple(locations),
            weather=weather_snapshot,
            routes=tuple(route_estimates),
            errors=tuple(errors),
        )


# ------------------------------ 地点筛选 ---------------------------------
class PlaceCatalog:
    """实现与存储无关的地点筛选规则，让内存和 SQLite 共用同一套约束。

    这里使用确定性代码执行预算、室内外和时长等硬约束，而不是让 LLM 自己
    判断。这样相同输入会得到稳定候选，也能用单元测试直接验证。
    """

    BUDGET_RANK = TRAVEL_BUDGET_RANK

    def search(
        self,
        preferences: TravelPreferences,
        limit: int = 3,
    ) -> list[Place]:
        """由具体目录实现提供地点数据。"""
#抛错“我不管你的数据是从 SQLite 来的，还是从内存里来的。谁继承我，谁就必须自己写 search 去把数据找出来
        raise NotImplementedError
##这是父类里真正干活的静态逻辑。不管子类怎么把数据找出来，最后都要扔进这个函数里洗一遍
    def _filter_places(
        self,
        places: tuple[Place, ...],
        preferences: TravelPreferences,
        limit: int,
    ) -> list[Place]:
        """对指定地点集合应用统一的业务筛选规则。"""
        preferences.validate()
        city = preferences.city.strip().casefold()
        budget_limit = self.BUDGET_RANK.get(preferences.budget, 2)

        candidates = [
            place
            for place in places
            if place.city.casefold() == city
            and self.BUDGET_RANK.get(place.budget_level, len(self.BUDGET_RANK))
            <= budget_limit
            and place.recommended_hours <= preferences.duration_hours
        ]

        if preferences.activity_preference == "室内为主":
            candidates = [
                place for place in candidates if place.environment == "室内"
            ]
        elif preferences.activity_preference == "户外为主":
            candidates = [
                place for place in candidates if place.environment == "户外"
            ]

        preferred_tags = {preferences.mood, preferences.companions}
        # 先按匹配标签数降序，再按所需时长和名称升序，保证结果稳定可复现。
        candidates.sort(
            key=lambda place: (
                -len(preferred_tags.intersection(place.tags)),#交集越多越靠前
                place.recommended_hours,                      #时间越短越靠前
                place.name,                                   #名字字典序
            )
        )
        return candidates[:limit]

    def to_context(self, preferences: TravelPreferences) -> str:
        """生成只能由程序筛选结果构成的候选上下文。"""
        candidates = self.search(preferences)
        if not candidates:
            return "当前本地点目录中没有符合条件的候选。"
        return "\n".join(
            f"{index}. {place.to_context()}"
            for index, place in enumerate(candidates, start=1)
        )


def create_default_places() -> tuple[Place, ...]:
    """返回教学用初始地点；内容不是实时营业数据。"""
    return (
            Place(
                "浙江省博物馆之江馆区",
                "杭州",
                "室内",
                "免费",
                3,
                ("文化", "想安静", "独自出行", "亲子出行"),
                "以浙江历史文化展览为主，具体开放安排需出发前核实。",
            ),
            Place(
                "中国丝绸博物馆",
                "杭州",
                "室内",
                "免费",
                2,
                ("文化", "想安静", "独自出行", "家庭出行"),
                "以丝绸历史、工艺和服饰展览为主。",
            ),
            Place(
                "杭州工艺美术博物馆",
                "杭州",
                "室内",
                "免费",
                2,
                ("文化", "想尝鲜", "朋友同行", "独自出行"),
                "适合了解杭州传统工艺，部分体验项目需单独确认。",
            ),
            Place(
                "杭州图书馆",
                "杭州",
                "室内",
                "免费",
                2,
                ("阅读", "想安静", "独自出行"),
                "适合安静阅读和短时间休息，开放时间需提前核实。",
            ),
            Place(
                "西湖景区",
                "杭州",
                "户外",
                "免费",
                3,
                ("散步", "想放松", "朋友同行", "情侣出行"),
                "以湖滨步行和户外观景为主。",
            ),
            Place(
                "宋城景区",
                "杭州",
                "混合",
                "宽松",
                4,
                ("演出", "想尝鲜", "朋友同行", "家庭出行"),
                "包含演出和大量景区游览区域，不属于纯室内地点。",
            ),
            Place(
                "上海博物馆东馆",
                "上海",
                "室内",
                "免费",
                3,
                ("文化", "想安静", "独自出行", "家庭出行"),
                "以中国古代艺术展览为主，参观预约规则需提前核实。",
            ),
            Place(
                "上海自然博物馆",
                "上海",
                "室内",
                "适中",
                3,
                ("科普", "亲子出行", "朋友同行"),
                "以自然科学和生命演化展览为主。",
            ),
            Place(
                "外滩",
                "上海",
                "户外",
                "免费",
                2,
                ("建筑", "想放松", "朋友同行", "情侣出行"),
                "以沿江步行和户外城市景观为主。",
            ),
        )


# ------------------------------ SQLite 数据层 -----------------------------
#只负责把用户的“偏好”存进 SQLite，或者读出来。不参与任何筛选，也不调大模型。
class TravelPreferenceStore:
    """按 user_id 持久化最近一次偏好，让用户刷新页面后仍可主动读取。

    偏好不放进全局变量，因为多用户并发时会互相覆盖；SQLite 对这个小型项目
    足够轻量，也让数据在程序重启后继续存在。
    """

    def __init__(self, database_path: str | Path):
        """保存数据库路径，并确保偏好表在首次使用前已经存在。"""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """统一管理 SQLite 事务：成功提交，异常回滚，最后一定关闭连接。"""
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_table(self) -> None:
        """创建旅行偏好表，并为旧表补充出发地字段。"""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS travel_preferences (
                    user_id TEXT PRIMARY KEY,
                    city TEXT NOT NULL,
                    departure TEXT NOT NULL DEFAULT '',
                    mood TEXT NOT NULL,
                    activity_preference TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    duration_hours INTEGER NOT NULL,
                    companions TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(travel_preferences)"
                )
            }
##如果发现老版本的数据库没有 departure（出发地）字段，还会自动执行 ALTER TABLE 补上这个字段
            if "departure" not in columns:
                connection.execute(
                    """
                    ALTER TABLE travel_preferences
                    ADD COLUMN departure TEXT NOT NULL DEFAULT ''
                    """
                )

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        """清理并检查演示用的用户标识。"""
        clean_user_id = user_id.strip()
        if not clean_user_id:
            raise ValueError("用户标识不能为空")
        if len(clean_user_id) > 64:
            raise ValueError("用户标识不能超过 64 个字符")
        return clean_user_id
##接收用户 ID 和偏好对象。用 INSERT ... ON CONFLICT DO UPDATE（UPSERT 语法）。
# 如果你今天选了“室内”，明天选了“户外”，它会把旧记录直接覆盖，永远只保留你最近一次的偏好
    def save(self, user_id: str, preferences: TravelPreferences) -> None:
        """插入偏好；同一用户再次保存时更新原记录。"""
        clean_user_id = self._validate_user_id(user_id)
        preferences.validate()
        updated_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO travel_preferences (
                    user_id, city, departure, mood, activity_preference,
                    budget, duration_hours, companions, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    city = excluded.city,
                    departure = excluded.departure,
                    mood = excluded.mood,
                    activity_preference = excluded.activity_preference,
                    budget = excluded.budget,
                    duration_hours = excluded.duration_hours,
                    companions = excluded.companions,
                    updated_at = excluded.updated_at
                """,
                (
                    clean_user_id,
                    preferences.city.strip(),
                    preferences.departure.strip(),
                    preferences.mood,
                    preferences.activity_preference,
                    preferences.budget,
                    preferences.duration_hours,
                    preferences.companions,
                    updated_at,
                ),
            )

    def load(self, user_id: str) -> TravelPreferences | None:
        """读取指定用户的偏好；不存在时返回 None。"""
        clean_user_id = self._validate_user_id(user_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT city, departure, mood, activity_preference,
                       budget, duration_hours, companions
                FROM travel_preferences
                WHERE user_id = ?
                """,
                (clean_user_id,),
            ).fetchone()

        if row is None:
            return None
        return TravelPreferences(
            city=row["city"],
            departure=row["departure"],
            mood=row["mood"],
            activity_preference=row["activity_preference"],
            budget=row["budget"],
            duration_hours=int(row["duration_hours"]),
            companions=row["companions"],
        )

##负责地点目录的增删改查（CRUD）。它是整个系统里 SQL 语句写得最复杂的地方。
class PlaceStore:
    """封装地点目录的 SQLite CRUD，不参与推荐规则和 LLM 调用。

    将 SQL 集中在数据层，可以让 Agent 只面对 Place 对象；稳定 place_id 则保证
    用户重命名城市或地点后仍然更新原记录，而不是误建一条新记录。
    """

    VALID_ENVIRONMENTS = {"室内", "户外", "混合"}
    VALID_BUDGET_LEVELS = set(PlaceCatalog.BUDGET_RANK)

    def __init__(self, database_path: str | Path):
        """保存数据库路径，并创建或迁移地点表结构。"""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """创建一次性连接，并保证事务结束后关闭。"""
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
##yield 在这里的魔力：它把函数劈成了两半。yield 之前是“进门发钥匙”，yield 之后是“出门收钥匙”。
# 业务代码在 with 块里面执行时，函数是暂停的
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_table(self) -> None:
        """创建新版地点表，并自动迁移旧版联合主键表。"""
        with self._connect() as connection:
            table_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'places'
                """
            ).fetchone()
            if table_exists is None:
                self._create_places_table(connection)
                return

            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(places)")
            }
            if "place_id" not in columns:
                self._migrate_legacy_table(connection)

    @staticmethod##（静态方法）既不接收实例 self，也不接收类 cls。它就是个普通函数，只是被“强行”归拢到了类里面，方便管理
    def _create_places_table(connection: sqlite3.Connection) -> None:
        """创建包含稳定主键和地点名称唯一约束的表。"""
        connection.execute(
            """
            CREATE TABLE places (
                place_id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                name TEXT NOT NULL,
                environment TEXT NOT NULL,
                budget_level TEXT NOT NULL,
                recommended_hours REAL NOT NULL,
                tags_json TEXT NOT NULL,
                description TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (city, name)
            )
            """
        )

    def _migrate_legacy_table(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """把旧版“城市+名称”主键迁移为稳定的自增 place_id。"""
        """把旧版 (city, name) 主键表迁移到 place_id 主键表。"""
        connection.execute("ALTER TABLE places RENAME TO places_legacy")
        self._create_places_table(connection)
        connection.execute(
            """
            INSERT INTO places (
                city, name, environment, budget_level,
                recommended_hours, tags_json, description, updated_at
            )
            SELECT city, name, environment, budget_level,
                   recommended_hours, tags_json, description, updated_at
            FROM places_legacy
            ORDER BY city, name
            """
        )
        connection.execute("DROP TABLE places_legacy")

    @classmethod#类方法）：属于“整个家族”的方法第一个参数不是 self，而是 cls（代表这个类本身）。它通常用于创建工厂实例或者读取类级别的配置。
    def _normalize_place(cls, place: Place) -> Place:
        """校验字段并返回去除首尾空白的新对象。"""
        name = place.name.strip()
        city = place.city.strip()
        description = place.description.strip()
        tags = tuple(tag.strip() for tag in place.tags if tag.strip())

        if not name or not city:
            raise ValueError("地点名称和城市不能为空")
        if place.environment not in cls.VALID_ENVIRONMENTS:
            raise ValueError("环境类型必须是：室内、户外或混合")
        if place.budget_level not in cls.VALID_BUDGET_LEVELS:
            raise ValueError("消费等级必须是：免费、经济、适中或宽松")
        if not 0 < place.recommended_hours <= 24:
            raise ValueError("建议时长必须大于 0 且不超过 24 小时")
        if not tags:
            raise ValueError("地点至少需要一个标签")
        if not description:
            raise ValueError("地点说明不能为空")
        if place.place_id is not None and place.place_id <= 0:
            raise ValueError("place_id 必须是正整数")

        return Place(
            name=name,
            city=city,
            environment=place.environment,
            budget_level=place.budget_level,
            recommended_hours=float(place.recommended_hours),
            tags=tags,
            description=description,
            place_id=place.place_id,
        )

    @staticmethod
    def _to_parameters(place: Place, updated_at: str) -> tuple[object, ...]:
        """把 Place 转换成参数化 SQL 所需的元组。"""
        return (
            place.city,
            place.name,
            place.environment,
            place.budget_level,
            place.recommended_hours,
            # SQLite 没有元组类型，因此把标签序列化为保留中文的 JSON 文本。
            json.dumps(place.tags, ensure_ascii=False),
            place.description,
            updated_at,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Place:
        """把 SQLite 行恢复成 Place。"""
        return Place(
            place_id=int(row["place_id"]),
            name=row["name"],
            city=row["city"],
            environment=row["environment"],
            budget_level=row["budget_level"],
            recommended_hours=float(row["recommended_hours"]),
            tags=tuple(json.loads(row["tags_json"])),
            description=row["description"],
        )

    def seed(self, places: tuple[Place, ...]) -> None:
        """仅在地点表为空时导入一次默认地点。"""
        normalized_places = [self._normalize_place(place) for place in places]
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            place_count = connection.execute(
                "SELECT COUNT(*) FROM places"
            ).fetchone()[0]
            if place_count > 0:
                return
            connection.executemany(
                """
                INSERT INTO places (
                    city, name, environment, budget_level,
                    recommended_hours, tags_json, description, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    self._to_parameters(place, updated_at)
                    for place in normalized_places
                ],
            )

    def upsert(self, place: Place) -> Place:
        """无 ID 时新增，有 ID 时更新同一条记录并支持重命名。"""
        normalized_place = self._normalize_place(place)
        updated_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                if normalized_place.place_id is None:
                    connection.execute(
                        """
                        INSERT INTO places (
                            city, name, environment, budget_level,
                            recommended_hours, tags_json, description, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(city, name) DO UPDATE SET
                            environment = excluded.environment,
                            budget_level = excluded.budget_level,
                            recommended_hours = excluded.recommended_hours,
                            tags_json = excluded.tags_json,
                            description = excluded.description,
                            updated_at = excluded.updated_at
                        """,
                        self._to_parameters(normalized_place, updated_at),
                    )
                    row = connection.execute(
                        """
                        SELECT place_id, city, name, environment, budget_level,
                               recommended_hours, tags_json, description
                        FROM places WHERE city = ? AND name = ?
                        """,
                        (normalized_place.city, normalized_place.name),
                    ).fetchone()
                else:
                    cursor = connection.execute(
                        """
                        UPDATE places SET
                            city = ?, name = ?, environment = ?,
                            budget_level = ?, recommended_hours = ?,
                            tags_json = ?, description = ?, updated_at = ?
                        WHERE place_id = ?
                        """,
                        (
                            *self._to_parameters(normalized_place, updated_at),
                            normalized_place.place_id,
                        ),
                    )
                    if cursor.rowcount == 0:
                        raise ValueError("要更新的地点已经不存在")
                    row = connection.execute(
                        """
                        SELECT place_id, city, name, environment, budget_level,
                               recommended_hours, tags_json, description
                        FROM places WHERE place_id = ?
                        """,
                        (normalized_place.place_id,),
                    ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ValueError("同一城市中已经存在这个地点名称") from exc

        if row is None:
            raise RuntimeError("保存地点后无法重新读取记录")
        return self._from_row(row)
##通过“业务唯一键”查 是“按名字找”，适合外部用户或脚本按业务字段查询
    def get(self, city: str, name: str) -> Place | None:
        """按城市和名称读取单个地点。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT place_id, city, name, environment, budget_level,
                       recommended_hours, tags_json, description
                FROM places
                WHERE city = ? AND name = ?
                """,
                (city.strip(), name.strip()),
            ).fetchone()
        return None if row is None else self._from_row(row)
##通过“物理主键”查 查找依据：唯一的自增 ID（place_id）是“按身份证号找”，适合系统内部精确更新、删除操作
    def get_by_id(self, place_id: int) -> Place | None:
        """按稳定 place_id 读取地点。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT place_id, city, name, environment, budget_level,
                       recommended_hours, tags_json, description
                FROM places WHERE place_id = ?
                """,
                (int(place_id),),
            ).fetchone()
        return None if row is None else self._from_row(row)
##是精准制导，给核心推荐业务用
    def list_by_city(self, city: str) -> list[Place]:
        """列出一个城市的全部地点。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT place_id, city, name, environment, budget_level,
                       recommended_hours, tags_json, description
                FROM places
                WHERE city = ?
                ORDER BY name
                """,
                (city.strip(),),
            ).fetchall()
        return [self._from_row(row) for row in rows]
##是大扫除，给后台管理或初始化用
    def list_all(self) -> list[Place]:
        """列出数据库中的全部地点。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT place_id, city, name, environment, budget_level,
                       recommended_hours, tags_json, description
                FROM places
                ORDER BY city, name
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, city: str, name: str) -> bool:
        """删除一个地点，并返回是否确实删除了记录。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM places WHERE city = ? AND name = ?",
                (city.strip(), name.strip()),
            )
        return cursor.rowcount > 0

    def delete_by_id(self, place_id: int) -> bool:
        """按稳定 place_id 删除地点。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM places WHERE place_id = ?",
                (int(place_id),),
            )
        return cursor.rowcount > 0
##SQLitePlaceCatalog 。绝不越权去写业务逻辑，只负责把 SQLite 里的 Raw Data（原始数据）捞出来，喂给父类那台精密的“过滤机器”。
class SQLitePlaceCatalog(PlaceCatalog):
    """连接数据层和筛选层：从 SQLite 取 Place，再应用统一推荐规则。"""

    def __init__(self, place_store: PlaceStore):
        """注入只负责 CRUD 的 PlaceStore。"""
        self.place_store = place_store

    def search(
        self,
        preferences: TravelPreferences,
        limit: int = 3,
    ) -> list[Place]:
        """查询目标城市记录，再复用父类的确定性筛选与排序。"""
        city_places = tuple(
            self.place_store.list_by_city(preferences.city)
        )
        return self._filter_places(city_places, preferences, limit)


# ------------------------------ Agent 编排层 ------------------------------
##它自己不去写 SQL，不去解析 HTTP 请求，也不去算向量。
# 它的唯一职责就是：把各个专业模块（路由、地点库、实时工具、知识库、大模型）按照正确的顺序串联起来，完成一次完整的问答。
class TravelAssistant:
    """协调路由、地点筛选、实时工具、RAG、LLM 与会话历史。

    这是应用的核心编排类。它不直接实现数据库或 HTTP 细节，而是把各模块按
    固定顺序组合起来；因此每个模块可以独立测试，也可以在构造函数中替换。
    """

    def __init__(
        self,
        llm: HelloAgentsLLM,
        max_history_turns: int = 3,
        place_catalog: PlaceCatalog | None = None,
        realtime_tools: RealTimeTravelTools | None = None,
        knowledge_base: TravelKnowledgeBase | None = None,
        tool_router: TravelToolRouter | None = None,
    ):
        """注入各能力模块，并设置最多保留的历史对话轮数。"""
        if max_history_turns <= 0:
            raise ValueError("max_history_turns 必须大于 0")
        self.llm = llm                                         #大模型
        self.max_history_turns = max_history_turns
        self.place_catalog = place_catalog                     #地点检索
        self.realtime_tools = realtime_tools                   #查天气、算路线
        self.knowledge_base = knowledge_base                   #公共攻略库
        self.tool_router = tool_router or TravelToolRouter(llm)#判断这轮要干嘛
##防止上下文无限增长，把历史截断到最多 3 轮
    def _limit_history(
        self,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """过滤非法消息并限制上下文长度，防止历史无限增长拖慢本地模型。"""
        valid_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in history
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ]
        message_limit = self.max_history_turns * 2
        return valid_messages[-message_limit:]

    def build_messages(
        self,
        preferences: TravelPreferences,
        question: str,
        history: list[dict[str, str]] | None = None,
        candidates: list[Place] | None = None,
        tool_report: TravelToolReport | None = None,
        knowledge_report: TravelKnowledgeReport | None = None,
        tool_decision: TravelToolDecision | None = None,
    ) -> list[dict[str, str]]:
        """按 system -> 最近历史 -> 当前 user 的顺序构造 LLM messages。"""
        preferences.validate()
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("用户问题不能为空")

        if candidates is None and self.place_catalog is not None:
            candidates = self.place_catalog.search(preferences)
        if candidates:
            place_context = "\n".join(
                f"{index}. {place.to_context()}"
                for index, place in enumerate(candidates, start=1)
            )
        elif tool_decision is not None and not tool_decision.use_place_search:
            place_context = "本轮 Agent 决策跳过地点检索。"
        else:
            place_context = "当前本地点目录中没有符合条件的候选。"
        realtime_context = (
            tool_report.to_context()
            if tool_report is not None
            else "本轮 Agent 未调用天气或路线工具。"
        )
        knowledge_context = (
            knowledge_report.to_context()
            if knowledge_report is not None
            else "本轮 Agent 未调用旅行知识库。"
        )
        decision_context = (
            tool_decision.to_context()
            if tool_decision is not None
            else "未提供工具决策。"
        )

        # system 消息承载可信规则与本轮工具事实，优先级高于普通对话历史。
        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n本轮用户偏好：\n"
                    + preferences.to_context()
                    + "\n\n本轮 Agent 工具决策：\n"
                    + decision_context
                    + "\n\n程序筛选后的候选地点：\n"
                    + place_context
                    + "\n\n实时工具结果：\n"
                    + realtime_context
                    + "\n\n旅行知识库检索结果：\n"
                    + knowledge_context
                ),
            }
        ]
        # 历史只包含 user/assistant 消息；当前问题必须最后追加，表示本轮输入。
        recent_history = self._limit_history(history or [])
        messages.extend(recent_history)
        messages.append({"role": "user", "content": clean_question})
        return messages

    def _request_answer(self, messages: list[dict[str, str]]) -> str:
        """调用本地 LLM 并要求 JSON 模式，集中处理空响应。"""
        answer = self.llm.invoke(
            messages,
            temperature=0.3,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        if not answer or not answer.strip():
            raise RuntimeError("模型没有返回有效内容")
        return answer.strip()

    @staticmethod
#大模型生成 JSON 后，并不是直接信任，而是要进行极其严格的“质检”：
    def _parse_plan(
        raw_answer: str,
        preferences: TravelPreferences,
        candidates: list[Place],
    ) -> TravelPlan:
        """把不可信模型 JSON 转成 TravelPlan，并执行所有硬约束。

        提示词只能引导模型，不能提供绝对保证。因此候选地点、预算、室内外偏好
        和总时长必须由 Python 再校验一次，任何违规结果都不能直接展示给用户。
        """
        json_start = raw_answer.find("{")
        json_end = raw_answer.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise ValueError("模型没有返回 JSON 对象")
        try:
            payload = json.loads(raw_answer[json_start : json_end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("模型返回的 JSON 无法解析") from exc
        if not isinstance(payload, dict):
            raise ValueError("模型返回的顶层数据必须是对象")

        summary = str(payload.get("summary", "")).strip()
        budget_summary = str(payload.get("budget_summary", "")).strip()
        raw_itinerary = payload.get("itinerary", [])
        raw_reminders = payload.get("reminders", [])
        if not summary or not budget_summary:
            raise ValueError("结构化行程缺少结论或预算说明")
        if not isinstance(raw_itinerary, list) or len(raw_itinerary) > 3:
            raise ValueError("行程必须是最多包含 3 项的列表")
        if not isinstance(raw_reminders, list):
            raise ValueError("出行提醒必须是列表")

        # 用程序候选建立白名单，模型输出名单外地点时立即拒绝。
        candidate_by_name = {place.name: place for place in candidates}
        itinerary: list[ItineraryStop] = []
        total_hours = 0.0
        for item in raw_itinerary:
            if not isinstance(item, dict):
                raise ValueError("每个行程项目都必须是对象")
            place_name = str(item.get("place_name", "")).strip()
            if place_name not in candidate_by_name:
                raise ValueError(f"模型使用了候选列表之外的地点：{place_name}")
            selected_place = candidate_by_name[place_name]
            place_budget_rank = TRAVEL_BUDGET_RANK.get(selected_place.budget_level)
            budget_limit = TRAVEL_BUDGET_RANK[preferences.budget]
            if place_budget_rank is None or place_budget_rank > budget_limit:
                raise ValueError(f"地点“{place_name}”超过用户预算约束")
            if (
                preferences.activity_preference == "室内为主"
                and selected_place.environment != "室内"
            ):
                raise ValueError(f"地点“{place_name}”违反室内偏好")
            if (
                preferences.activity_preference == "户外为主"
                and selected_place.environment != "户外"
            ):
                raise ValueError(f"地点“{place_name}”违反户外偏好")
            try:
                duration = float(item.get("duration_hours", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("行程时长必须是数字") from exc
            if duration <= 0:
                raise ValueError("行程时长必须大于 0")
            total_hours += duration
            text_fields = {
                key: (
                    str(item.get(key)).strip()
                    if item.get(key) is not None
                    else ""
                )
                for key in ("period", "activity", "transport", "estimated_cost")
            }
            missing_required = [
                key
                for key in ("period", "activity")
                if not text_fields[key]
            ]
            if missing_required:
                raise ValueError(
                    "行程项目缺少必要字段：" + "、".join(missing_required)
                )
            transport = text_fields["transport"] or (
                "参考实时路线报告，并使用地图核实具体出行方式"
            )
            estimated_cost = text_fields["estimated_cost"] or (
                "具体费用需出发前核实"
            )
            itinerary.append(
                ItineraryStop(
                    period=text_fields["period"],
                    place_name=place_name,
                    activity=text_fields["activity"],
                    duration_hours=duration,
                    transport=transport,
                    estimated_cost=estimated_cost,
                )
            )
        if total_hours > preferences.duration_hours:
            raise ValueError("模型生成的总行程时长超过用户可用时间")

        reminders = tuple(
            str(reminder).strip()
            for reminder in raw_reminders
            if str(reminder).strip()
        )
        return TravelPlan(
            summary=summary,
            itinerary=tuple(itinerary),
            budget_summary=budget_summary,
            reminders=reminders,
        )

    @staticmethod
    def _fallback_plan(
        preferences: TravelPreferences,
        candidates: list[Place],
    ) -> TravelPlan:
        """模型连续违规时，用可信候选生成不会突破硬约束的保守计划。

        这不是第三次猜测，而是完全由程序从已经通过城市、预算、活动偏好和
        单地点时长筛选的候选中组装结果，因此不会引入候选列表外的地点。
        """
        if not candidates:
            return TravelPlan(
                summary="本轮没有可用于结构化行程的可信候选地点。",
                itinerary=(),
                budget_summary="没有生成地点行程，因此暂无可估算的消费等级。",
                reminders=(
                    "可先在“地点目录管理”中添加并核实地点，再重新提问。",
                ),
            )

        periods = ("上午", "下午", "晚上")
        remaining_hours = float(preferences.duration_hours)
        stops: list[ItineraryStop] = []
        for place in candidates:
            if len(stops) >= 3 or remaining_hours <= 0:
                break
            duration = min(float(place.recommended_hours), remaining_hours)
            if duration <= 0:
                continue
            highlights = "、".join(place.tags[:3]) or place.description or "地点体验"
            stops.append(
                ItineraryStop(
                    period=periods[len(stops)],
                    place_name=place.name,
                    activity=f"体验{place.name}，重点关注：{highlights}",
                    duration_hours=duration,
                    transport="请使用地图工具核实具体路线和出行方式",
                    estimated_cost=(
                        f"{place.budget_level}消费等级，具体费用请在出发前核实"
                    ),
                )
            )
            remaining_hours -= duration

        selected_names = "、".join(stop.place_name for stop in stops)
        budget_levels = "、".join(
            dict.fromkeys(
                place.budget_level
                for place in candidates
                if place.name in {stop.place_name for stop in stops}
            )
        )
        return TravelPlan(
            summary=f"已根据你的硬性偏好生成保守方案：{selected_names}。",
            itinerary=tuple(stops),
            budget_summary=(
                f"所选地点属于{budget_levels}消费等级，具体金额请在出发前核实。"
            ),
            reminders=(
                "本地模型的原始结果未通过地点约束校验，本方案由程序使用可信候选生成。",
                "攻略中的新地点需先录入地点目录并核实信息，才能加入结构化行程。",
            ),
        )

    def _request_plan(
        self,
        messages: list[dict[str, str]],
        preferences: TravelPreferences,
        candidates: list[Place],
    ) -> TravelPlan:
        """请求并校验计划；首次不合格时把具体错误反馈给模型修正一次。"""
        raw_answer = self._request_answer(messages)
        try:
            return self._parse_plan(raw_answer, preferences, candidates)
        except ValueError as first_error:
            allowed_names = [place.name for place in candidates]
            candidate_constraint = (
                "当前没有候选地点，因此 itinerary 必须严格为 []，不得输出任何地点。"
                if not candidates
                else (
                    "itinerary 中每个 place_name 必须逐字等于以下字符串之一："
                    f"{json.dumps(allowed_names, ensure_ascii=False)}。"
                    "不得改写、概括或创建其他地点名称。"
                )
            )
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw_answer},
                {
                    "role": "user",
                    "content": (
                        "上一份 JSON 未通过程序校验："
                        f"{first_error}。请严格按照系统指定格式重新输出，"
                        f"{candidate_constraint}只能返回 JSON 对象。"
                    ),
                },
            ]
            repaired_answer = self._request_answer(repair_messages)
            try:
                return self._parse_plan(
                    repaired_answer,
                    preferences,
                    candidates,
                )
            except ValueError as second_error:
                LOGGER.warning(
                    "event=plan_fallback first_error=%s second_error=%s candidates=%d",
                    first_error,
                    second_error,
                    len(candidates),
                )
                return self._fallback_plan(preferences, candidates)

    def reply_with_history(
        self,
        preferences: TravelPreferences,
        question: str,
        history: list[dict[str, str]] | None,
        user_knowledge_base: TravelKnowledgeBase | None = None,
        knowledge_base_factory: Callable[[], TravelKnowledgeBase] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """执行完整 Agent 流程，并返回回答及更新后的会话历史。

        历史由调用方传入和保存，而不是存在共享 TravelAssistant 对象中，这样多个
        浏览器会话同时访问时不会串话。
        """
        preferences.validate()###检查偏好是否数据合理
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("用户问题不能为空")
        #把前端传回来的历史记录过滤一遍，只保留合法的 user 和 assistant 消息，并且只截取最近的 3 轮（防止本地小模型上下文超长）
        session_history = self._limit_history(history or [])###把问题放到会话历史里面去，如果存在历史，否则为空

        # 1. 先由路由器判断本轮真正需要的工具，避免每次都调用全部服务。
        ##这里调用 TravelToolRouter，把“用户偏好 + 问题 + 历史”发给大模型，让大模型返回一个 JSON，
        # 告诉程序：这一轮到底要不要查地点？要不要查天气？要不要查知识库？
        tool_decision = self.tool_router.decide(
            preferences,
            clean_question,
            session_history,
        )
        # 2. 只有推荐/路线问题才从可信地点库执行硬约束筛选。
        candidates = (
            #继承 PlaceCatalog，重写search函数，里面调用self._filter_places(...)，这里是在数据库里面拿数据
            ##要是后面需要实现从其他地方拿数据，这里就直接新写一个类函数继承PlaceCatalog，重写search函数就行，继续调用self._filter_places
            self.place_catalog.search(preferences)
            if tool_decision.use_place_search and self.place_catalog is not None
            else []
        )
        # 3. 按决策获取天气或路线；工具内部允许单项失败并降级。
        tool_report = (
            #它是一个包工头。根据 decide 的决定，如果开了天气，它就派（GeocodingTool 和 WeatherTool）去查经纬度和温度；
            # 如果开了路线，它就派（RouteTool）去算距离和时间
            self.realtime_tools.collect(
                preferences,
                candidates,
                include_weather=tool_decision.use_weather,
                include_routes=tool_decision.use_routes,
            )
            if self.realtime_tools is not None
            and (tool_decision.use_weather or tool_decision.use_routes)
            else None
        )
        # 4. 仅在需要攻略证据时检索共享库和当前用户的私人知识库。
        knowledge_report = None
        if tool_decision.use_knowledge:
            knowledge_bases = []
            if self.knowledge_base is not None:
                knowledge_bases.append(self.knowledge_base)
            if user_knowledge_base is not None:
                knowledge_bases.append(user_knowledge_base)
            elif knowledge_base_factory is not None:
                knowledge_bases.append(knowledge_base_factory())

            snippets = []
            seen_bases: set[int] = set()
            for knowledge_base in knowledge_bases:
                if id(knowledge_base) in seen_bases:
                    continue
                seen_bases.add(id(knowledge_base))
                snippets.extend(
                    knowledge_base.search(
                        f"{preferences.city}\n{clean_question}",
                        limit=3,
                    ).snippets
                )
            knowledge_report = TravelKnowledgeReport(
                snippets=tuple(
                    sorted(snippets, key=lambda item: item.score, reverse=True)[:3]
                )
            )
        # 5. 把规则、历史、候选和工具事实统一组装成 messages。
        messages = self.build_messages(
            preferences,
            question,
            session_history,
            candidates,
            tool_report,
            knowledge_report,
            tool_decision,
        )
        # 6. 调用 LLM，并将生成结果重新交给 Python 做结构与约束校验。
        plan = self._request_plan(messages, preferences, candidates)
        answer_parts = [tool_decision.to_markdown()]
        if tool_report is not None:
            answer_parts.append(tool_report.to_markdown())
        if knowledge_report is not None and knowledge_report.snippets:
            answer_parts.append(knowledge_report.to_markdown())
        answer_parts.append(plan.to_markdown())
        answer = "\n\n".join(answer_parts)
        # 7. 只把最终展示内容写入本会话历史，并再次限制保留轮数。
        updated_history = self._limit_history(
            [
                *session_history,
                {"role": "user", "content": clean_question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer, updated_history

# ------------------------------ Gradio 界面层 -----------------------------
def create_web_app(
    assistant: TravelAssistant,
    preference_store: TravelPreferenceStore,
    place_store: PlaceStore,
    knowledge_base_manager: UserKnowledgeBaseManager,
):
    """创建页面并把 UI 事件绑定到业务对象，返回尚未启动的 Gradio 应用。

    把建页和 launch 分开，测试可以只构造应用而不占用端口；依赖通过参数传入，
    也避免回调函数自行创建数据库和 Agent 导致状态不一致。
    """
    # Gradio 是 Web 层的可选依赖，局部导入可让纯业务测试不必加载整个 UI 框架。
    import gradio as gr
##生成一个 12 位的短 UUID，用来追踪每一次报错
    def request_id() -> str:
        """为一次页面操作生成短追踪号，连接用户提示与服务器日志。"""
        return uuid.uuid4().hex[:12]
##统一捕获异常。如果后台报错了，它不会把 Python 的红色堆栈暴露给用户，而是记录日志，然后给用户弹出一个带 request_id 的友好提示（gr.Error）。
    def fail_web_request(
        action: str,
        user_id: str,
        operation_id: str,
        exc: Exception,
    ) -> None:
        """统一记录回调异常，并转换成不泄露内部细节的 gr.Error。"""
        log_user = user_log_id(user_id) if user_id else "unknown"
        if isinstance(exc, (ValueError, LookupError)):
            LOGGER.warning(
                "event=%s_failed request_id=%s user=%s error_type=%s",
                action,
                operation_id,
                log_user,
                type(exc).__name__,
            )
        else:
            LOGGER.exception(
                "event=%s_failed request_id=%s user=%s error_type=%s",
                action,
                operation_id,
                log_user,
                type(exc).__name__,
            )
        raise gr.Error(user_facing_error(exc, operation_id)) from exc
#当用户刚打开网页，或者刷新页面时触发。它会去读用户的登录名或浏览器 Session ID，算出这个用户的 user_id，并去数据库里查他专属的知识库状态。
    def initialize_session(
        request: gr.Request,
    ) -> tuple[str, str, str, list[list[object]]]:
        """页面加载时建立用户隔离身份，并读取该用户的知识库状态。"""
        user_id = resolve_user_identity(request.username, request.session_hash)
        display_name = (
            f"登录用户：{request.username}"
            if request.username
            else f"临时会话：{request.session_hash[:8]}"
        )
        LOGGER.info("event=session_started user=%s", user_log_id(user_id))
        try:
            status = knowledge_summary(user_id)
            rows = knowledge_rows(user_id)
        except Exception as exc:
            operation_id = request_id()
            LOGGER.exception(
                "event=session_initialization_failed request_id=%s user=%s error_type=%s",
                operation_id,
                user_log_id(user_id),
                type(exc).__name__,
            )
            status = f"私人知识库暂时无法加载。（错误编号：{operation_id}）"
            rows = []
        return (
            user_id,
            display_name,
            status,
            rows,
        )

    place_table_headers = [
        "ID",
        "城市",
        "地点名称",
        "环境类型",
        "消费等级",
        "建议时长",
        "标签",
        "地点说明",
    ]
#parse_tags()：把用户输入的 "安静、文化" 变成元组 ("安静", "文化")
    def parse_tags(tags_text: str) -> tuple[str, ...]:
        """把“安静、文化”这类表单文本转换成 Place 使用的不可变元组。"""
        return tuple(
            tag.strip()
            for tag in re.split(r"[,，、]+", tags_text)
            if tag.strip()
        )
#place_to_row() / place_to_form()：把后台的 Place 对象，变成前端表格能显示的一行，或者变成输入框能回填的值。
    def place_to_row(place: Place) -> list[object]:
        """按表头固定顺序把 Place 转成 Gradio Dataframe 的一行。"""
        return [
            place.place_id,
            place.city,
            place.name,
            place.environment,
            place.budget_level,
            place.recommended_hours,
            "、".join(place.tags),
            place.description,
        ]
    def place_to_form(place: Place) -> tuple[str, str, str, str, float, str, str]:
        """把 Place 转成七个控件值，用于点击表格后回填编辑表单。"""
        return (
            place.city,
            place.name,
            place.environment,
            place.budget_level,
            place.recommended_hours,
            "、".join(place.tags),
            place.description,
        )
##有了 make_preferences，所有的 int 转换都在这里统一处理了，防止报错
    def make_preferences(
        city: str,
        departure: str,
        mood: str,
        activity_preference: str,
        budget: str,
        duration_hours: int,
        companions: str,
    ) -> TravelPreferences:
        """把页面控件值转换为统一的旅行偏好对象。"""
        return TravelPreferences(
            city=city,
            departure=departure,
            mood=mood,
            activity_preference=activity_preference,
            budget=budget,
            duration_hours=int(duration_hours),
            companions=companions,
        )
##把左侧表单打包成对象，存进 SQLite。
    def save_user_preferences(
        user_id: str,
        city: str,
        departure: str,
        mood: str,
        activity_preference: str,
        budget: str,
        duration_hours: int,
        companions: str,
    ) -> None:
        """把当前页面偏好保存到 SQLite。"""
        operation_id = request_id()
        started_at = time.perf_counter()
        try:
            preferences = make_preferences(
                city,
                departure,
                mood,
                activity_preference,
                budget,
                duration_hours,
                companions,
            )
            preference_store.save(user_id, preferences)
        except Exception as exc:
            fail_web_request("preferences_save", user_id, operation_id, exc)
        LOGGER.info(
            "event=preferences_saved request_id=%s user=%s duration_ms=%d",
            operation_id,
            user_log_id(user_id),
            round((time.perf_counter() - started_at) * 1000),
        )
        gr.Info("旅行偏好已保存")
##从 SQLite 查出来，返回给前端，自动填满左侧表单。
    def load_user_preferences(
        user_id: str,
    ) -> tuple[str, str, str, str, str, int, str]:
        """从 SQLite 读取偏好并回填页面控件。"""
        operation_id = request_id()
        try:
            preferences = preference_store.load(user_id)
            if preferences is None:
                raise LookupError("当前用户还没有保存旅行偏好")
        except Exception as exc:
            fail_web_request("preferences_load", user_id, operation_id, exc)
        LOGGER.info(
            "event=preferences_loaded request_id=%s user=%s",
            operation_id,
            user_log_id(user_id),
        )
        gr.Info("旅行偏好已读取")
        return (
            preferences.city,
            preferences.departure,
            preferences.mood,
            preferences.activity_preference,
            preferences.budget,
            preferences.duration_hours,
            preferences.companions,
        )
##接收所有的表单值、问题和历史对话 把数据打包丢给 assistant.reply_with_history()
    def submit_question(
        user_id: str,
        city: str,
        departure: str,
        mood: str,
        activity_preference: str,
        budget: str,
        duration_hours: int,
        companions: str,
        question: str,
        session_history: list[dict[str, str]] | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
        """把网页输入转换为偏好对象，调用助手并更新聊天框。"""
        operation_id = request_id()
        started_at = time.perf_counter()
        LOGGER.info(
            "event=chat_started request_id=%s user=%s history_messages=%d",
            operation_id,
            user_log_id(user_id),
            len(session_history or []),
        )
        try:
            preferences = make_preferences(
                city,
                departure,
                mood,
                activity_preference,
                budget,
                duration_hours,
                companions,
            )
            _, updated_history = assistant.reply_with_history(
                preferences,
                question,
                session_history,
                knowledge_base_factory=lambda: knowledge_base_manager.for_user(user_id),
            )
        except Exception as exc:
            fail_web_request("chat", user_id, operation_id, exc)

        LOGGER.info(
            "event=chat_completed request_id=%s user=%s duration_ms=%d history_messages=%d",
            operation_id,
            user_log_id(user_id),
            round((time.perf_counter() - started_at) * 1000),
            len(updated_history),
        )

        return updated_history, updated_history, ""

    def clear_conversation(
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
        """清空当前浏览器会话的模型上下文和页面记录。"""
        return [], [], ""
#生成知识库文档列表和统计信息
    def knowledge_rows(user_id: str) -> list[list[object]]:
        """把知识库文档统计转换成表格行。"""
        knowledge_base = knowledge_base_manager.for_user(user_id)
        return [
            [
                item["document_id"],
                item["source"],
                item["chunks"],
                item["created_at"],
            ]
            for item in knowledge_base.list_documents()
        ]
#生成知识库文档列表和统计信息
    def knowledge_summary(user_id: str, message: str = "") -> str:
        """生成知识库状态文本。"""
        knowledge_base = knowledge_base_manager.for_user(user_id)
        documents = knowledge_base.list_documents()
        chunks = sum(int(item["chunks"]) for item in documents)
        summary = f"当前知识库：{len(documents)} 份文档，{chunks} 个文本块。"
        return f"{message}\n\n{summary}" if message else summary
#把 PDF 或粘贴的文本交给 UserKnowledgeBaseManager，存入该用户专属的 RAG 库
    def import_knowledge_file(
        user_id: str,
        uploaded_file: object,
    ) -> tuple[str, list[list[object]], None]:
        """把网页上传的 PDF、Markdown 或 TXT 导入 RAG。"""
        operation_id = request_id()
        try:
            if uploaded_file is None:
                raise ValueError("请先选择攻略文件")
            file_path = getattr(uploaded_file, "name", uploaded_file)
            knowledge_base = knowledge_base_manager.for_user(user_id)
            result = knowledge_base.import_file(str(file_path))
        except Exception as exc:
            fail_web_request("knowledge_import", user_id, operation_id, exc)
        LOGGER.info(
            "event=knowledge_imported request_id=%s user=%s chunks=%s",
            operation_id,
            user_log_id(user_id),
            result["chunks"],
        )
        message = (
            f"**导入成功：** {result['document_id']}，"
            f"生成 {result['chunks']} 个文本块。"
        )
        return knowledge_summary(user_id, message), knowledge_rows(user_id), None
#把 PDF 或粘贴的文本交给 UserKnowledgeBaseManager，存入该用户专属的 RAG 库
    def save_guide_text(
        user_id: str,
        title: str,
        content: str,
    ) -> tuple[str, list[list[object]], str, str]:
        """把粘贴的旅行攻略导入 RAG。"""
        operation_id = request_id()
        try:
            knowledge_base = knowledge_base_manager.for_user(user_id)
            result = knowledge_base.add_guide(title, content)
        except Exception as exc:
            fail_web_request("knowledge_save", user_id, operation_id, exc)
        LOGGER.info(
            "event=knowledge_saved request_id=%s user=%s chunks=%s",
            operation_id,
            user_log_id(user_id),
            result["chunks"],
        )
        message = (
            f"**保存成功：** {result['document_id']}，"
            f"生成 {result['chunks']} 个文本块。"
        )
        return knowledge_summary(user_id, message), knowledge_rows(user_id), "", ""

    def refresh_knowledge(user_id: str) -> tuple[str, list[list[object]]]:
        """重新读取知识库统计和文档列表，供“刷新”按钮使用。"""
        return knowledge_summary(user_id), knowledge_rows(user_id)
#两步确认清空知识库
    def request_clear_knowledge() -> tuple[str, object]:
        """显示清空知识库的二次确认区域。"""
        return (
            "即将删除知识库中的全部攻略和向量文本块，此操作不可撤销。",
            gr.update(visible=True),
        )
#两步确认清空知识库,第二部要弹窗出现提示按钮
    def confirm_clear_knowledge(
        user_id: str,
    ) -> tuple[str, list[list[object]], object]:
        """确认后清空知识库。"""
        operation_id = request_id()
        try:
            knowledge_base_manager.for_user(user_id).clear()
        except Exception as exc:
            fail_web_request("knowledge_clear", user_id, operation_id, exc)
        LOGGER.info(
            "event=knowledge_cleared request_id=%s user=%s",
            operation_id,
            user_log_id(user_id),
        )
        return (
            knowledge_summary(user_id, "**旅行知识库已清空。**"),
            [],
            gr.update(visible=False),
        )

    def cancel_clear_knowledge() -> tuple[str, object]:
        """取消清空知识库。"""
        return "", gr.update(visible=False)

    def list_place_rows(city_filter: str = "") -> list[list[object]]:
        """读取全部或指定城市地点，并转换成表格刷新所需的二维列表。"""
        clean_city = city_filter.strip()
        places = (
            place_store.list_by_city(clean_city)
            if clean_city
            else place_store.list_all()
        )
        return [place_to_row(place) for place in places]
##有 ID 就更新，无 ID 就新增。
    def save_place(
        user_id: str,
        selected_id: int | None,
        city_value: str,
        name: str,
        environment: str,
        budget_level: str,
        recommended_hours: float,
        tags_text: str,
        description: str,
    ) -> tuple[object, ...]:
        """将表单转换为 Place：无 ID 新增，有 ID 更新，并返回刷新后的表格。"""
        operation_id = request_id()
        try:
            saved_place = place_store.upsert(
                Place(
                    place_id=(
                        int(selected_id)
                        if selected_id is not None
                        else None
                    ),
                    name=name,
                    city=city_value,
                    environment=environment,
                    budget_level=budget_level,
                    recommended_hours=float(recommended_hours),
                    tags=parse_tags(tags_text),
                    description=description,
                )
            )
        except Exception as exc:
            fail_web_request("place_save", user_id, operation_id, exc)
        LOGGER.info(
            "event=place_saved request_id=%s user=%s place_id=%s",
            operation_id,
            user_log_id(user_id),
            saved_place.place_id,
        )
        form_values = place_to_form(saved_place)
        gr.Info("地点已保存")
        return (
            list_place_rows(saved_place.city),
            saved_place.place_id,
            list(form_values),
            *form_values,
            f"**编辑模式：** ID {saved_place.place_id} · {saved_place.city} / {saved_place.name}",
        )

    def clear_place_form() -> tuple[object, ...]:
        """清空输入并退出编辑模式，下一次保存将创建新地点。"""
        return (
            None,
            None,
            "",
            "",
            "室内",
            "适中",
            2.0,
            "",
            "",
            "**新建模式：** 填写地点后点击“保存或更新”。",
        )

    def cancel_place_edit(snapshot: list[object] | None) -> tuple[object, ...]:
        """放弃未保存修改，恢复选中地点的原始表单值。"""
        if snapshot is None or len(snapshot) != 7:
            raise gr.Error("当前没有正在编辑的地点")
        return (
            *snapshot,
            "**编辑模式：** 已恢复选中地点尚未修改时的内容。",
        )
##同样是两步确认删除，防误触。
    def request_place_deletion(
        selected_id: int | None,
    ) -> tuple[int, str, str, object]:
        """读取待删除记录并显示内联确认区，不立即删除。"""
        if selected_id is None:
            raise gr.Error("请先从表格中选择要删除的地点")
        place = place_store.get_by_id(int(selected_id))
        if place is None:
            raise gr.Error("要删除的地点已经不存在")
        message = (
            f"即将删除 **ID {place.place_id} · {place.city} / {place.name}**。"
            "此操作会立即影响后续推荐。"
        )
        return int(selected_id), place.city, message, gr.update(visible=True)
##同样是两步确认删除，防误触。
    def confirm_place_deletion(
        user_id: str,
        pending_id: int | None,
        pending_city: str,
    ) -> tuple[object, ...]:
        """用户二次确认后按 place_id 删除地点。"""
        if pending_id is None:
            raise gr.Error("当前没有等待确认的删除操作")
        operation_id = request_id()
        try:
            deleted = place_store.delete_by_id(int(pending_id))
        except Exception as exc:
            fail_web_request("place_delete", user_id, operation_id, exc)
        if not deleted:
            raise gr.Error("要删除的地点已经不存在")
        LOGGER.info(
            "event=place_deleted request_id=%s user=%s place_id=%s",
            operation_id,
            user_log_id(user_id),
            pending_id,
        )
        gr.Info("地点已删除")
        return (
            list_place_rows(pending_city),
            *clear_place_form(),
            None,
            "",
            "",
            gr.update(visible=False),
        )

    def cancel_place_deletion() -> tuple[None, str, str, object]:
        """取消等待中的删除操作。"""
        return None, "", "", gr.update(visible=False)
##点击表格某行，自动把那一行的数据回填到上方的编辑表单里。
    def load_place_from_table(
        evt: gr.SelectData,
    ) -> tuple[object, ...]:
        """把用户点击的地点表格行回填到编辑表单。"""
        row = evt.row_value
        if row is None or len(row) != len(place_table_headers):
            raise gr.Error("无法读取所选地点，请重新点击表格中的数据行")
        place_id = int(row[0])
        form_values = (
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            float(row[5]),
            str(row[6]),
            str(row[7]),
        )
        return (
            place_id,
            list(form_values),
            *form_values,
            f"**编辑模式：** ID {place_id} · {form_values[0]} / {form_values[1]}",
        )

    with gr.Blocks(title="智能旅行助手") as app:
        # gr.State 只存在于当前浏览器会话，用来隔离身份、聊天历史和编辑状态。
        session_user_id = gr.State(value="")
        session_history = gr.State(value=[])
        selected_place_id = gr.State(value=None)
        selected_place_snapshot = gr.State(value=None)
        pending_delete_id = gr.State(value=None)
        pending_delete_city = gr.State(value="")
        gr.Markdown("# 智能旅行助手")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                user_identity = gr.Textbox(
                    label="当前用户",
                    value="正在初始化会话",
                    interactive=False,
                )
                city = gr.Textbox(label="城市", value="杭州")
                departure = gr.Textbox(
                    label="出发地",
                    value="杭州市中心",
                    placeholder="地址、地标或区域名称",
                )
                mood = gr.Dropdown(
                    ["心情很好", "想放松", "有点疲惫", "想尝鲜", "想安静"],
                    value="心情很好",
                    label="当前心情",
                )
                activity_preference = gr.Radio(
                    ["室内为主", "室内外皆可", "户外为主"],
                    value="室内为主",
                    label="活动偏好",
                )
                budget = gr.Radio(
                    ["经济", "适中", "宽松"],
                    value="适中",
                    label="预算",
                )
                duration_hours = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=1,
                    label="可用时间（小时）",
                )
                companions = gr.Dropdown(
                    ["独自出行", "朋友同行", "情侣出行", "亲子出行", "家庭出行"],
                    value="独自出行",
                    label="同行方式",
                )
                with gr.Row():
                    save_preferences_button = gr.Button("保存偏好")
                    load_preferences_button = gr.Button("读取偏好")

            with gr.Column(scale=2, min_width=420):
                chatbot = gr.Chatbot(
                    label="旅行建议",
                    height=480,
                    layout="panel",
                )
                question = gr.Textbox(
                    label="你的问题",
                    value="我今天心情很好，但不喜欢户外，适合去哪里玩？",
                    lines=2,
                )
                with gr.Row():
                    submit_button = gr.Button("发送", variant="primary")
                    clear_button = gr.Button("清空")

        with gr.Accordion("旅行知识库 RAG", open=False):
            gr.Markdown(
                "导入 PDF、Markdown、TXT，或直接粘贴旅行攻略。"
                "同名文档再次导入时会覆盖旧文本块。"
            )
            knowledge_status = gr.Markdown(value="正在加载当前用户的知识库...")
            with gr.Row():
                knowledge_file = gr.File(
                    label="攻略文件",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                    type="filepath",
                )
                with gr.Column():
                    import_knowledge_button = gr.Button(
                        "导入文件",
                        variant="primary",
                    )
                    refresh_knowledge_button = gr.Button("刷新知识库列表")
                    request_clear_knowledge_button = gr.Button(
                        "清空知识库",
                        variant="stop",
                    )
            with gr.Row():
                guide_title = gr.Textbox(
                    label="攻略标题",
                    placeholder="例如：杭州雨天室内游",
                )
                guide_content = gr.Textbox(
                    label="攻略内容",
                    placeholder="粘贴你整理的攻略正文",
                    lines=5,
                )
            save_guide_button = gr.Button("保存粘贴攻略")
            knowledge_table = gr.Dataframe(
                headers=["文档", "来源", "文本块", "导入时间"],
                value=[],
                interactive=False,
                wrap=True,
                label="已导入文档",
            )
            with gr.Group(visible=False) as clear_knowledge_confirmation:
                clear_knowledge_message = gr.Markdown()
                with gr.Row():
                    confirm_clear_knowledge_button = gr.Button(
                        "确认清空",
                        variant="stop",
                    )
                    cancel_clear_knowledge_button = gr.Button("取消")

        with gr.Accordion("地点目录管理", open=False):
            gr.Markdown("地点使用稳定 ID 保存；选中记录后可以修改城市或名称。")
            edit_status = gr.Markdown(
                "**新建模式：** 填写地点后点击“保存或更新”。"
            )
            with gr.Row():
                place_city = gr.Textbox(label="城市", value="杭州")
                place_name = gr.Textbox(label="地点名称")
                place_environment = gr.Dropdown(
                    ["室内", "户外", "混合"],
                    value="室内",
                    label="环境类型",
                )
                place_budget = gr.Dropdown(
                    ["免费", "经济", "适中", "宽松"],
                    value="适中",
                    label="消费等级",
                )
                place_hours = gr.Number(
                    value=2,
                    minimum=0.5,
                    maximum=24,
                    label="建议时长（小时）",
                )
            place_tags = gr.Textbox(
                label="标签",
                placeholder="例如：安静、文化、独自出行",
            )
            place_description = gr.Textbox(label="地点说明", lines=2)
            with gr.Row():
                save_place_button = gr.Button("保存或更新", variant="primary")
                clear_place_button = gr.Button("清空表单")
                cancel_edit_button = gr.Button("取消编辑")
                request_delete_button = gr.Button("删除地点", variant="stop")
            with gr.Row():
                query_place_button = gr.Button("查询城市")
                list_all_places_button = gr.Button("显示全部")
            with gr.Group(visible=False) as delete_confirmation:
                delete_message = gr.Markdown()
                with gr.Row():
                    confirm_delete_button = gr.Button(
                        "确认删除",
                        variant="stop",
                    )
                    cancel_delete_button = gr.Button("取消删除")
            place_table = gr.Dataframe(
                headers=place_table_headers,
                value=list_place_rows(),
                interactive=False,
                wrap=True,
                label="地点目录",
            )

        # 先集中定义输入输出列表，后面的事件绑定便能复用且保持参数顺序一致。
        inputs = [
            session_user_id,
            city,
            departure,
            mood,
            activity_preference,
            budget,
            duration_hours,
            companions,
            question,
            session_history,
        ]
        preference_inputs = [
            session_user_id,
            city,
            departure,
            mood,
            activity_preference,
            budget,
            duration_hours,
            companions,
        ]
        preference_outputs = [
            city,
            departure,
            mood,
            activity_preference,
            budget,
            duration_hours,
            companions,
        ]
        save_preferences_button.click(
            save_user_preferences,
            inputs=preference_inputs,
            api_name="save_preferences",
        )
        load_preferences_button.click(
            load_user_preferences,
            inputs=session_user_id,
            outputs=preference_outputs,
            api_name="load_preferences",
        )
        submit_button.click(
            submit_question,
            inputs=inputs,
            # 同一份历史分别更新可见聊天框和隐藏状态，空字符串用于清空输入框。
            outputs=[chatbot, session_history, question],
            api_name="submit_question",
        )
        question.submit(
            submit_question,
            inputs=inputs,
            outputs=[chatbot, session_history, question],
            api_name=False,
        )
        clear_button.click(
            clear_conversation,
            outputs=[chatbot, session_history, question],
            api_name="clear_conversation",
        )
        import_knowledge_button.click(
            import_knowledge_file,
            inputs=[session_user_id, knowledge_file],
            outputs=[knowledge_status, knowledge_table, knowledge_file],
            api_name="import_knowledge_file",
        )
        save_guide_button.click(
            save_guide_text,
            inputs=[session_user_id, guide_title, guide_content],
            outputs=[knowledge_status, knowledge_table, guide_title, guide_content],
            api_name="save_guide_text",
        )
        refresh_knowledge_button.click(
            refresh_knowledge,
            inputs=session_user_id,
            outputs=[knowledge_status, knowledge_table],
            api_name="list_knowledge_documents",
        )
        request_clear_knowledge_button.click(
            request_clear_knowledge,
            outputs=[clear_knowledge_message, clear_knowledge_confirmation],
            api_name=False,
        )
        confirm_clear_knowledge_button.click(
            confirm_clear_knowledge,
            inputs=session_user_id,
            outputs=[
                knowledge_status,
                knowledge_table,
                clear_knowledge_confirmation,
            ],
            api_name="clear_knowledge_base",
        )
        cancel_clear_knowledge_button.click(
            cancel_clear_knowledge,
            outputs=[clear_knowledge_message, clear_knowledge_confirmation],
            api_name=False,
        )
        place_inputs = [
            place_city,
            place_name,
            place_environment,
            place_budget,
            place_hours,
            place_tags,
            place_description,
        ]
        place_table.select(
            load_place_from_table,
            outputs=[
                selected_place_id,
                selected_place_snapshot,
                *place_inputs,
                edit_status,
            ],
            api_name=False,
        )
        save_place_button.click(
            save_place,
            inputs=[session_user_id, selected_place_id, *place_inputs],
            outputs=[
                place_table,
                selected_place_id,
                selected_place_snapshot,
                *place_inputs,
                edit_status,
            ],
            api_name="save_place",
        )
        clear_place_button.click(
            clear_place_form,
            outputs=[
                selected_place_id,
                selected_place_snapshot,
                *place_inputs,
                edit_status,
            ],
            api_name=False,
        )
        cancel_edit_button.click(
            cancel_place_edit,
            inputs=selected_place_snapshot,
            outputs=[*place_inputs, edit_status],
            api_name=False,
        )
        request_delete_button.click(
            request_place_deletion,
            inputs=selected_place_id,
            outputs=[
                pending_delete_id,
                pending_delete_city,
                delete_message,
                delete_confirmation,
            ],
            api_name=False,
        )
        confirm_delete_button.click(
            confirm_place_deletion,
            inputs=[session_user_id, pending_delete_id, pending_delete_city],
            outputs=[
                place_table,
                selected_place_id,
                selected_place_snapshot,
                *place_inputs,
                edit_status,
                pending_delete_id,
                pending_delete_city,
                delete_message,
                delete_confirmation,
            ],
            api_name="delete_place",
        )
        cancel_delete_button.click(
            cancel_place_deletion,
            outputs=[
                pending_delete_id,
                pending_delete_city,
                delete_message,
                delete_confirmation,
            ],
            api_name=False,
        )
        query_place_button.click(
            list_place_rows,
            inputs=place_city,
            outputs=place_table,
            api_name="list_places_by_city",
        )
        list_all_places_button.click(
            lambda: list_place_rows(),
            outputs=place_table,
            api_name="list_all_places",
        )
        app.load(
            initialize_session,
            outputs=[
                session_user_id,
                user_identity,
                knowledge_status,
                knowledge_table,
            ],
            api_name=False,
        )
    return app


def main() -> None:
    """完成依赖装配、安全检查和 Gradio 服务启动。"""
    parser = argparse.ArgumentParser(description="智能旅行助手")
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=7862, help="Web 服务端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="允许在非本机地址上无认证启动，仅用于受信任网络",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    try:
        auth = parse_auth_config(os.getenv("TRAVEL_AUTH"))
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.host not in {"127.0.0.1", "localhost", "::1"}
        and auth is None
        and not args.allow_unauthenticated
    ):
        parser.error(
            "非本机监听必须配置 TRAVEL_AUTH，或显式使用 --allow-unauthenticated"
        )

    concurrency = int(os.getenv("TRAVEL_CONCURRENCY", "2"))
    if concurrency <= 0:
        parser.error("TRAVEL_CONCURRENCY 必须大于 0")

    # 在应用入口统一创建依赖，业务类本身不偷偷依赖全局对象，便于测试和替换。
    place_store = PlaceStore(DATABASE_PATH)
    place_store.seed(create_default_places())
    # BGE 模型体积远大于普通业务对象，因此全应用只创建一份并按需懒加载。
    embedding = BGEEmbedding()
    shared_knowledge_base = TravelKnowledgeBase(
        KNOWLEDGE_BASE_DIR,
        embedding=embedding,
    )
    knowledge_base_manager = UserKnowledgeBaseManager(embedding=embedding)
    assistant = TravelAssistant(
        create_llm(),
        max_history_turns=3,
        place_catalog=SQLitePlaceCatalog(place_store),
        realtime_tools=RealTimeTravelTools.create_default(),
        knowledge_base=shared_knowledge_base,
    )
    preference_store = TravelPreferenceStore(DATABASE_PATH)
    app = create_web_app(
        assistant,
        preference_store,
        place_store,
        knowledge_base_manager,
    )
    LOGGER.info(
        "event=server_starting host=%s port=%d auth_enabled=%s log_path=%s",
        args.host,
        args.port,
        auth is not None,
        log_path,
    )
    # queue 控制同时执行的 LLM 请求数；blocked_paths 防止敏感本地文件被 Web 暴露。
    app.queue(default_concurrency_limit=concurrency).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=False,
        show_error=False,
        auth=auth,
        auth_message="请输入智能旅行助手账号",
        max_file_size=os.getenv("TRAVEL_MAX_UPLOAD_SIZE", "20mb"),
        root_path=os.getenv("TRAVEL_ROOT_PATH") or None,
        state_session_capacity=int(os.getenv("TRAVEL_SESSION_CAPACITY", "1000")),
        blocked_paths=[
            str(PROJECT_DIR / ".env"),
            str(DATABASE_PATH),
            str(KNOWLEDGE_BASE_DIR),
            str(LOG_DIR),
        ],
    )


if __name__ == "__main__":
    # 只有直接执行该文件时才启动服务；被测试代码 import 时不会自动占用端口。
    try:
        main()
    except Exception:
        if not LOGGER.handlers:
            configure_logging()
        LOGGER.exception("event=server_crashed")
        raise
