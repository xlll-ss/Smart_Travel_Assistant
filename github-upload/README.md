# Hello Agents 智能旅行助手

这是一个基于本地 Ollama、HelloAgents 框架、实时旅行工具和 RAG 知识库构建的智能旅行助手。项目使用 Gradio 提供 Web 界面，使用 SQLite 保存地点、偏好和向量数据，不需要云端 API Key。

## 功能

- 根据城市、心情、室内/户外偏好、预算、时长和同行方式生成结构化行程。
- Agent 自动判断是否需要地点检索、天气、驾车路线或旅行知识库。
- Photon 地理编码、Open-Meteo 当前天气、OSRM 驾车距离。
- 导入 PDF、Markdown、TXT 或粘贴攻略，使用本地 RAG 检索。
- 程序级检查城市、室内/户外、预算、候选白名单和总时长约束。
- 浏览器会话历史、持久化用户偏好和每用户独立知识库。
- 日志轮转、安全错误编号、自动测试和 Ollama 端到端评测。

## 执行流程

```text
用户问题 + 结构化偏好 + 最近对话
    -> TravelToolRouter
    -> TravelToolDecision
       ├── 地点目录检索
       ├── 天气与地理位置
       ├── OSRM 驾车路线
       └── 私人/共享攻略 RAG
    -> 合并可信上下文
    -> Ollama 生成结构化 JSON
    -> 白名单、环境、预算和时长校验
    -> Gradio Markdown 回答
```

模型只负责生成建议，硬约束由 Python 代码再次校验。即使模型尝试使用候选列表之外、户外或超预算地点，行程也不会直接通过。

## 项目结构

```text
hello-agents/
├── hello_agents/                 # Agent 框架源码
│   ├── core/                     # LLM、消息、配置和异常
│   ├── agents/                   # Simple/ReAct/Reflection/Plan-and-Solve
│   └── tools/                    # 工具基类、注册表、工具链和内置工具
├── evals/
│   └── travel_constraints.json   # 旅行约束评测集
├── tests/                        # pytest 自动测试
├── knowledge_base/               # 共享与私人 RAG SQLite
├── logs/                         # 轮转运行日志
├── travel_assistant.py           # 正式 Web 应用入口
├── travel_evaluation.py          # 离线/真实模型评测入口
├── wise_douctment.py             # 通用本地 RAG 实现
├── check_environment.py          # 环境、数据库和隔离检查
├── travel_assistant.db           # 本地运行数据，不提交 Git
├── .env.example                  # 配置模板
├── requirements.txt              # Python 依赖
├── INTERVIEW_NOTES.md             # 项目面试问题、回答要点与复盘清单
├── UPLOAD_CHECKLIST.md             # GitHub 上传范围和检查步骤
├── DEPLOYMENT.md                 # 详细部署和备份说明
└── README.md                     # 项目总文档
```

`wise_douctment.py` 的文件名保留了早期项目兼容性，正式旅行应用只导入其中的 `RAGTool`。

## 项目文档

- [面试问题笔记](INTERVIEW_NOTES.md)：138 道项目问题、企业级高并发设计、故障场景、简历表述和演示清单。
- [部署说明](DEPLOYMENT.md)：认证、反向代理、日志、备份与恢复。
- [GitHub 上传检查](UPLOAD_CHECKLIST.md)：应提交和必须排除的文件。

## `hello_agents` 包保留说明

以下源码全部保留，它们共同组成可复用 Agent 框架：

- `core/agent.py`：Agent 抽象基类。
- `core/llm.py`：OpenAI 兼容模型统一接口，当前连接 Ollama。
- `core/message.py`：消息结构和角色。
- `core/config.py`：通用配置。
- `core/exceptions.py`：异常体系。
- `agents/*.py`：四种 Agent 实现。
- `tools/*.py`：工具、注册、链式调用和异步执行。
- `tools/builtin/*.py`：计算与搜索工具。
- 所有 `__init__.py` 和 `version.py`：Python 包定义。

`__pycache__`、`.pyc` 和空的旧 `memory` 目录不是源码，已经从正式项目中移除，运行后产生的缓存由 `.gitignore` 排除。

## 环境要求

- Windows 10/11 或兼容 Linux 环境。
- Python 3.10 以上，当前验证版本为 Python 3.12。
- Ollama。
- 推荐至少 8 GB 内存；是否使用 GPU 取决于 Ollama 和显卡驱动。

安装 Ollama 模型：

```powershell
ollama pull qwen2.5:3b
ollama list
```

创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置

创建本地配置：

```powershell
Copy-Item .env.example .env
```

最小配置：

```env
LLM_MODEL_ID=qwen2.5:3b
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_TIMEOUT=180
```

本地 Ollama 不需要 API Key。不要提交 `.env`。

常用可选配置：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TRAVEL_TOOL_TIMEOUT` | `8` | 外部实时工具超时秒数 |
| `TRAVEL_CONCURRENCY` | `2` | Gradio 并发数 |
| `TRAVEL_MAX_UPLOAD_SIZE` | `20mb` | 文档上传上限 |
| `TRAVEL_LOG_LEVEL` | `INFO` | 日志级别 |
| `TRAVEL_AUTH` | 空 | `user:password` 登录配置 |
| `TRAVEL_ROOT_PATH` | 空 | 反向代理子路径 |

## 运行

先执行环境检查：

```powershell
python check_environment.py
python check_environment.py --live
```

启动 Web 应用：

```powershell
python travel_assistant.py --no-browser
```

访问：

```text
http://127.0.0.1:7862/
```

局域网部署必须先在 `.env` 设置 `TRAVEL_AUTH`，然后运行：

```powershell
python travel_assistant.py --host 0.0.0.0 --port 7862 --no-browser
```

完整的认证、反向代理、日志和备份说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 数据隔离

```text
认证用户名或浏览器会话
    -> 哈希内部 user_id
       ├── travel_assistant.db / travel_preferences
       ├── gr.State 对话历史
       └── knowledge_base/users/travel_guides_<hash>.db
```

- 启用认证时，同一用户名可以跨会话读取自己的偏好和攻略。
- 未启用认证时，只保证当前浏览器会话隔离。
- `knowledge_base/travel_guides.db` 是兼容旧数据的共享只读知识来源。
- `travel_assistant.db / places` 是系统共享地点目录。

## 测试与评测

运行稳定的离线测试：

```powershell
python -m pytest -q
```

运行确定性评测集：

```powershell
python travel_evaluation.py
```

调用本地 Ollama 运行端到端评测：

```powershell
python travel_evaluation.py --live
```

输出 JSON 报告：

```powershell
python travel_evaluation.py --live --output travel_eval_report.json
```

评测会检查错误工具选择、跨城市地点、超预算、违反室内/户外偏好、候选外地点、单地点时长、总时长和不应出现的行程。

## 日志与异常

日志默认写入：

```text
logs/travel_assistant.log
```

页面只显示安全的错误说明和 12 位错误编号。根据编号检索日志：

```powershell
Select-String -Path .\logs\travel_assistant.log* -Pattern "错误编号"
```

日志不会主动记录用户问题全文、攻略正文、密码或原始用户名。

## 运行数据库

需要保留的本地运行数据：

- `travel_assistant.db`：地点目录和用户偏好。
- `knowledge_base/travel_guides.db`：旧版共享攻略。
- `knowledge_base/users/*.db`：各用户私人攻略。

这些文件是运行数据，不应提交 Git。旧的 `memory_*`、`rag_demo*`、`vector_memory_demo.db`、`document_qa_demo.db` 和 `pdf_*` 数据库属于早期学习 Demo，不是正式旅行应用依赖。

## 当前限制

- Gradio 登录适合演示和小型内网服务，不是完整企业身份系统。
- 地点管理仍是共享功能，没有管理员角色权限。
- HashEmbedding 与 SQLite 全量向量检索只适合小型知识库。
- Photon、Open-Meteo 和 OSRM 是公共服务，可能超时或限流。
- 当前推荐单进程部署；多进程需要重新设计 SQLite 连接和进程锁。
