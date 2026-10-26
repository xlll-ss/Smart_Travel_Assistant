# Knowledge Base Runtime Directory

该目录用于保存智能旅行助手运行时生成的 RAG SQLite 数据库。

首次启动应用或导入攻略时，程序会自动创建所需数据库，不需要提前提交 `.db` 文件。以下内容已由根目录 `.gitignore` 排除：

- `*.db`、`*.db-wal`、`*.db-shm`
- `knowledge_base/users/`

这样可以避免把本地用户攻略、向量文本块和会话数据上传到 GitHub。此说明文件用于让 Git 保留 `knowledge_base` 目录本身。
