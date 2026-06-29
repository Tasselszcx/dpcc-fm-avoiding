# 项目约定

## 沟通语言
- 始终用**中文**与用户对话。

## Git 提交身份（硬性要求）
- 所有提交统一使用身份：`zhangchenxu06 <3199601798@qq.com>`（author 与 committer 都用此邮箱）。
- **不要**在提交信息里添加 `Co-Authored-By: Claude ...` 之类的 Claude 署名。
- 远程仓库：`https://github.com/Tasselszcx/dpcc-fm-avoiding`（注意不是 grpo-quickstart）。
- 推送走代理：`git -c http.proxy=http://10.217.148.40:8080 -c http.version=HTTP/1.1 push ...`（推前 `unset no_proxy NO_PROXY`）。
- `logs/` 已 gitignore，只提交代码 + 报告，不要提交二进制 / .npz / .pkl。
