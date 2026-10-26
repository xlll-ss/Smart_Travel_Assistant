# 智能旅行助手部署说明

本文对应 `travel_assistant.py`。默认部署方式是 Windows 本机运行、Ollama 只监听本机、Gradio 监听 `127.0.0.1:7862`。

## 1. 数据与进程边界

| 内容 | 位置 | 隔离方式 |
| --- | --- | --- |
| 旅行偏好 | `travel_assistant.db` | 以内部用户 ID 分行保存 |
| 对话历史 | Gradio `gr.State` | 每个浏览器会话独立，不持久化 |
| 私人攻略 RAG | `knowledge_base/users/travel_guides_<hash>.db` | 每个用户独立 SQLite |
| 旧版共享攻略 | `knowledge_base/travel_guides.db` | 所有用户只读检索，兼容已有数据 |
| 地点目录 | `travel_assistant.db` | 系统共享数据 |
| 服务日志 | `logs/travel_assistant.log` | 5 MB 自动轮转，默认保留 5 份 |

启用登录认证后，内部用户 ID 由登录用户名的 SHA-256 摘要生成。同一账号刷新或重新登录后仍能读取自己的偏好和知识库。未启用认证时使用 Gradio `session_hash`，不同浏览器会话互相隔离，但刷新或会话失效后不会恢复原来的临时身份。

新上传或粘贴的攻略只写入当前用户的私人知识库。旧版 `knowledge_base/travel_guides.db` 保留为共享检索来源，避免升级后已有攻略突然不可见；页面不会向普通用户提供修改该共享库的入口。

地点目录是管理员维护的共享基础数据，不属于用户私有内容。当前页面没有细粒度角色系统，因此多用户部署时应只向可信用户开放“地点目录管理”。若要公开服务，应把管理功能拆到单独的管理员页面。

## 2. 环境准备

安装 Python 3.12、Ollama，并下载模型：

```powershell
ollama pull qwen2.5:3b
ollama list
```

创建虚拟环境并安装依赖：

```powershell
cd "D:\pytorch_xllll\github开源项目\hello-agents\hello-agents"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

从示例创建本地配置，再按机器情况修改：

```powershell
Copy-Item .env.example .env
```

`.env` 包含本机配置和登录凭据，已经加入 `.gitignore`，不要提交到代码仓库。

## 3. 本机启动

先确认 Ollama API 可用：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

运行自动检查：

```powershell
python check_environment.py
python -m pytest -q
python travel_evaluation.py
```

启动 Web 服务：

```powershell
python travel_assistant.py --no-browser
```

访问 `http://127.0.0.1:7862/`。启动后可以用以下命令做健康检查：

```powershell
(Invoke-WebRequest http://127.0.0.1:7862/ -UseBasicParsing).StatusCode
```

返回 `200` 表示 Gradio 服务可访问，但不等于 Ollama、天气和路线服务一定可用。完整功能应再运行 `python travel_evaluation.py --live`。

## 4. 局域网部署与认证

在 `.env` 中配置至少一个账号：

```env
TRAVEL_AUTH=alice:replace-with-a-long-password,bob:another-long-password
```

然后监听全部网卡：

```powershell
python travel_assistant.py --host 0.0.0.0 --port 7862 --no-browser
```

程序在非本机地址启动时默认要求 `TRAVEL_AUTH`。只有在完全受信任的隔离网络中，才可以显式使用：

```powershell
python travel_assistant.py --host 0.0.0.0 --allow-unauthenticated --no-browser
```

匿名模式只提供会话级隔离，不适合作为互联网公开部署方案。Windows 防火墙只应向需要访问的内网网段开放 7862 端口，不要将 Ollama 的 11434 端口暴露到公网。

## 5. 反向代理

建议由 Caddy、Nginx 或企业网关终止 HTTPS，再代理到 `127.0.0.1:7862`。部署在子路径 `/travel` 时设置：

```env
TRAVEL_ROOT_PATH=/travel
```

反向代理需要支持 WebSocket，并传递 `Host`、`X-Forwarded-For` 和 `X-Forwarded-Proto`。不要把 `.env`、SQLite、`knowledge_base` 或 `logs` 配置为静态目录。程序已经通过 Gradio `blocked_paths` 阻止这些路径被文件接口读取，但反向代理仍需正确配置。

## 6. 日志和错误排查

日志写入 `logs/travel_assistant.log`，格式包括时间、级别、事件名、请求 ID、哈希用户 ID和耗时。日志不会主动记录用户问题全文、攻略正文、密码或原始用户名。

默认配置：

```env
TRAVEL_LOG_LEVEL=INFO
TRAVEL_LOG_MAX_BYTES=5242880
TRAVEL_LOG_BACKUP_COUNT=5
```

页面发生异常时会显示类似 `错误编号：a1b2c3d4e5f6`。使用该编号搜索日志：

```powershell
Select-String -Path .\logs\travel_assistant.log* -Pattern "a1b2c3d4e5f6"
```

输入校验错误可以显示具体原因；数据库错误、模型错误和未知异常只显示通用提示，详细堆栈仅写入服务日志。

## 7. 备份与恢复

停止写入后备份以下内容：

```text
travel_assistant.db
knowledge_base/users/
.env
```

SQLite 使用 WAL 模式。在线备份不能只复制主 `.db` 文件，否则可能遗漏尚未合并的 WAL 内容。简单可靠的方式是先停止服务，再复制数据库及知识库目录；生产环境应使用 SQLite Backup API 或定期快照。

恢复时停止服务，将备份文件放回原路径，再重新启动。认证用户名变化会生成不同的内部用户 ID，因此恢复后应继续使用原用户名登录。

## 8. 当前限制

- Gradio 登录适合演示和小型内部服务，不替代企业身份系统。
- 单进程内通过锁保护每个 RAG 数据库；多进程部署需要重新设计连接和锁。
- 地点目录仍是共享管理数据，没有管理员角色权限。
- 本地 HashEmbedding 和 SQLite 全量向量检索适合小规模知识库。
- Open-Meteo、Photon 和 OSRM 是外部公共服务，可能超时或限流。
- 日志是本地文件，尚未接入集中式监控、指标和告警。

在这些限制解决前，推荐使用单进程、受认证保护的内网部署。
