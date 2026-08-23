# A股多因子量化研究模型：源码快照

## 快照信息

- 打包日期：2026-08-23
- 生成方式：`generate_package.py`（从当前工作区实际文件生成，清单与磁盘自洽）
- 仓库状态：非 git 工作区（无提交/分支元数据）

## 测试声明（P0-1 修正，随工作区状态动态生成）

- **撤回历史 `782 passed` 声明**：该数字出自 2026-07-23 历史快照；当前不以任何离线快照数字自证。
- 工作区含 `tests/`（`2` 个测试文件），本地 Python312 环境 `python -m pytest tests/ -q` 实跑通过；
  未在干净环境 CI 复现，如实说明。
- 复现方式：`python -m venv .venv && .\.venv\Scripts\python.exe -m pip install -r requirements.txt`
  （开发/测试依赖见 `requirements-dev.txt`；未在干净环境 CI 实跑，如实说明）。

## 包含内容

- 核心回测、滚动训练、多因子排序、混合权重、无泄漏验证与原子发布代码
- 成本模型、前瞻跟踪、面板刷新、交易日历与观察模块
- 配置文件、文档、依赖声明

## 有意排除

- `external_data/`（历史行情面板，>1GB）、`outputs/`（结果产物，>500MB）
- `logs/`、`__pycache__/`、`.workbuddy/`、历史压缩包、`.log/.bak/.npz/.zip`

## 基础验证

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

完整每日流程仍需配置本地行情目录。本项目当前用于研究和实盘前置验证，
自动下单保持关闭，不构成投资建议或收益承诺。
