# GitHub 上传检查清单

## 已包含

- `hello_agents/` Agent 框架源码
- `travel_assistant.py` 正式旅行助手
- `wise_douctment.py` 本地 RAG 实现
- `travel_evaluation.py` 和 `evals/` 评测代码与数据
- `tests/` 自动测试
- `README.md`、`DEPLOYMENT.md` 和 `INTERVIEW_NOTES.md`
- `requirements.txt`、`pytest.ini`、`.env.example` 和 `.gitignore`
- `knowledge_base/README.md` 运行数据库说明

## 已排除

- `.env`：可能包含本机配置和登录凭据
- `*.db`、`*.db-wal`、`*.db-shm`：运行数据库和用户数据
- `knowledge_base/users/`：私人攻略知识库
- `logs/`：运行日志
- `__pycache__/`、`.pytest_cache/`、`*.pyc`：运行缓存
- `.vscode/`：本机编辑器配置

## 上传前验证

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python check_environment.py
```

不要在上传目录中新建真实 `.env` 后执行 `git add -f`。如果曾经把密钥提交到 Git 历史，仅删除文件并不够，还需要撤销并更换对应密钥。

## 初始化仓库

```powershell
git init
git add .
git status
git commit -m "Initial commit: intelligent travel assistant"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

执行 `git status` 时应确认没有 `.env`、数据库、日志或私人知识库文件进入暂存区。
