# Tasks: Model pipeline refactor (`model_train_predict.ipynb` → `model_pipeline`)

**Workspace**: `model_pipeline` | **Date**: 2025-03-22  
**Input**: `specs/model_pipeline/` 下的 `spec.md`、`plan.md`  
**Prerequisites**: plan.md ✓, spec.md ✓

**Note**: 源码包路径为仓库根目录下的 `model_pipeline/`（与 `plan.md` 一致）。超参模块文件名为 **`tune_model_params.py`**（与当前 `spec.md` / `plan.md` 一致）。

---

## ⚠️ 测试要求（强制）

**每个 Phase 结束时必须满足**：

- 该 Phase 内**全部**单元测试与验收测试通过  
- **覆盖率**（本项目尚无历史约定）：**行覆盖率 ≥ 50%，分支覆盖率 ≥ 50%**（与 `tasks` 技能默认一致；若后续团队采用更严阈值，可改为 80%/60% 并重跑 gate）  
- **User Story Phase 还须**：验收测试覆盖该 Story 下列 **Acceptance Scenarios** 与 **Edge Cases**

**不满足以上要求，不能进入下一个 Phase**

**验收测试方式（项目类型）**：`plan.md` 判定为 **纯库/SDK**（Python 模块 + pytest）→ 使用 **`pytest` 集成/验收用例**（`tests/acceptance/`），无需浏览器或独立 HTTP 服务。新建验收文件；若日后仓库已有统一 `tests/integration/`，可迁移并复用命名。

---

## 任务格式

```
- [ ] [TaskID] [P?] [Story?] 描述，含文件路径
```

---

## Phase 1: Setup（项目初始化）

**目的**: Python 包布局、依赖与 pytest/coverage 可运行。

- [x] T001 创建包目录与占位 `model_pipeline/__init__.py`
- [x] T002 在仓库根目录添加 `pyproject.toml`（或 `requirements.txt` + `requirements-dev.txt`），声明运行时依赖：`pandas`、`numpy`、`scikit-learn`、`lightgbm`；开发依赖：`pytest`、`pytest-cov`
- [x] T003 [P] 配置 pytest / coverage（`pyproject.toml` 中 `[tool.pytest.ini_options]` 与 `[tool.coverage.run]`），将 `model_pipeline` 纳入 `pythonpath` 或包可安装模式
- [x] T004 [P] 创建 `tests/__init__.py`

### 单元测试（必须）

- [x] T005 添加冒烟测试 `tests/test_smoke_imports.py`：可 `import model_pipeline` 及后续子模块占位不报错
- [x] T006 验证 Phase 1 覆盖率 gate（至少覆盖 `test_smoke_imports` 与配置相关文件的可统计范围）

**Phase Gate**: `pytest` 可执行，包可被导入，覆盖率 gate 满足

---

## Phase 2: Foundational（阻塞性基础设施）

**目的**: 所有用户故事共用的类型、指标注册与测试夹具。

**⚠️ 关键**: 完成前不得开始各 Story 的业务实现（除本 Phase 文件外）。

- [x] T007 在 **`model_pipeline/train_model.py`** 中定义 **`ModelType` 枚举**（至少 **LGB**；预留 **XGB** 等）、**`TrainResult`** 与 **`TuningReport`**（或等价 dataclass / NamedTuple，字段含 `model_type`、路径、可选最佳轮数等）；**本里程碑不增加单独 `types.py`**，与 `plan.md` 源码树一致
- [x] T008 [P] 实现 `model_pipeline/metrics.py`：metric 名称（如 `auc`）到 `sklearn.metrics` 可调用对象的注册/解析，供 `tune_model_params` 与训练日志对齐
- [x] T009 更新 `model_pipeline/__init__.py` 导出公共符号（`ModelType`、`train`、`load_dataset` 等按实现进度逐步加入，本 Phase 至少导出 `ModelType`）
- [x] T010 实现 `tests/conftest.py`：合成二分类 DataFrame 工厂、临时 parquet 路径、`tmp_path` 辅助

### 单元测试（必须）

- [x] T011 编写 `tests/test_types_and_metrics.py`：从 **`train_model`** 导入 `ModelType` 等与 **`metrics`** 联动（非法 metric 名报错等 Edge Cases）
- [x] T012 验证 Phase 2 覆盖率达标

**Phase Gate**: 基础设施 UT 全过，覆盖率达标

---

## Phase 3: User Story 1 — 从路径加载特征与标签 (Priority: P1) 🎯 MVP 起点

**目标**: 满足 **spec 需求 1**：给定路径读取 Parquet（文件或目录数据集），返回 DataFrame 与特征列 / 标签列约定；**禁止**模块内硬编码数据根路径。

**Acceptance Scenarios**（验收测试须覆盖）:

1. **Given** 有效 Parquet 文件路径与 `label_column`、`feature_columns`，**When** 调用 `load_dataset`，**Then** 返回的表包含所需列且行数与文件一致。  
2. **Given** 临时目录写入的分区/多文件 Parquet（若计划支持），**When** 调用 `load_dataset`，**Then** 成功合并读取（与 `pandas.read_parquet` 行为一致）。  

**Edge Cases**: 路径不存在；`feature_columns` 与数据列交集为空；`label_column` 不在表中。

### 实现

- [x] T013 [US1] 实现 `model_pipeline/load_data.py`：`load_dataset(path, *, label_column, feature_columns, exclude_columns=...)`（签名以 `plan.md` 为准，允许返回 NamedTuple `LoadedData`）

### 单元测试（必须）

- [x] T014 [US1] 编写 `tests/test_load_data.py`（临时 parquet、列校验、异常路径）

### 验收测试（必须）

- [x] T015 [US1] 编写 `tests/acceptance/test_load_data_acceptance.py`：端到端写入 parquet → `load_dataset` → 断言列与 shape

- [x] T016 [US1] 验证本 Phase 覆盖率达标

**Phase Gate**: US1 验收场景与 Edge Cases 通过，UT + 验收全过，覆盖率达标

---

## Phase 4: User Story 2 — 按比例与策略划分数据 (Priority: P1)

**目标**: 满足 **spec 需求 2**：支持 **random**（`train_test_split` 风格，可选 stratify）与 **time_window**（日期列 + 起止窗口）；策略通过参数选择，无硬编码路径。

**Acceptance Scenarios**:

1. **Given** 带日期字符串列的 DataFrame，**When** `strategy=time_window` 与 `train_start`/`train_end`，**Then** 训练集行均落在窗口内。  
2. **Given** 二分类标签列，**When** `strategy=random` 且 `stratify=True`，**Then** 训练/验证集正例比例近似一致（容差在测试中明确）。  

**Edge Cases**: 时间过滤后 0 行；`test_size` 非法；缺失 `date_column`。

### 实现

- [x] T017 [US2] 实现 `model_pipeline/split_data.py`：`SplitConfig`（或等价）与统一入口 `split_data(df, strategy, ...)`，至少 `random`、`time_window`

### 单元测试（必须）

- [x] T018 [US2] 编写 `tests/test_split_data.py`（含空结果与 stratify）

### 验收测试（必须）

- [x] T019 [US2] 编写 `tests/acceptance/test_split_data_acceptance.py`：合成月度数据，时间窗划分可重复断言边界

- [x] T020 [US2] 验证本 Phase 覆盖率达标

**Phase Gate**: US2 全部通过，覆盖率达标

---

## Phase 5: User Story 3 — 多后端训练入口，首期仅 LGB (Priority: P1)

**目标**: 满足 **spec 需求 3**：`train(..., model_type=...)` **默认 LGB**；**未实现的 `model_type` 显式失败**；LGB 路径支持 early stopping、`num_boost_round`、保存模型与 **gain** 特征重要性 CSV、可选 **refit_on_full_data**（mid 模型轮数 × `boost_round_multiplier`）；修复笔记本中无 `val` 且 early stopping 时使用错误标签变量（应为 `train_data[label]`）的问题。

**Acceptance Scenarios**:

1. **Given** 小样本 train/val 与 LGB `params`，**When** `train(..., model_type=LGB)`，**Then** 模型文件与重要性 CSV 写入指定路径。  
2. **Given** `val_data is None` 且 `is_early_stopping=True`，**When** 训练，**Then** 通过 **split_data** 的 random 策略产生验证集且标签来自同一 train 表（回归测试防 `data[label]` 类 bug）。  
3. **Given** `model_type` 为已声明但未实现的后端（如 XGB），**When** `train`，**Then** `NotImplementedError` 或带清晰信息的异常。  
4. **Given** `refit_on_full_data=True`，**When** 训练完成，**Then** 存在全量 refit 终模且轮数策略与 `plan.md` 一致。  

**Edge Cases**: `val_data is None` 且 `is_early_stopping=False` 仅单 Dataset；极大 `num_boost_round` 下 early stopping 提前停（可用小数据+高学习率触发）。

### 实现

- [x] T021 [US3] 实现 `model_pipeline/train_model.py`：对外 `train(...)` 分发 + `_train_lgb(...)`；`log_evaluation` / `early_stopping` 轮次参数可配置（默认值脱离魔法数）
- [x] T022 [US3] 内部 holdout 划分 **调用** `model_pipeline/split_data.py`，禁止复制粘贴 `train_test_split` 逻辑
- [x] T023 [US3] 实现 `refit_on_full_data` 与 mid 模型文件命名策略（与笔记本语义一致，可在 API 中显式化 `mid_model_file_stem` 等参数）

### 单元测试（必须）

- [x] T024 [US3] 编写 `tests/test_train_model.py`（LGB 小数据、未实现 backend、无 val early stopping 标签一致性）

### 验收测试（必须）

- [x] T025 [US3] 编写 `tests/acceptance/test_train_model_acceptance.py`：真实 **LGB** 训练（`lightgbm.train`）落盘模型 + 读取 `Booster` 树数量与重要性文件存在性

- [x] T026 [US3] 验证本 Phase 覆盖率达标

**Phase Gate**: US3 全部通过，覆盖率达标  

**MVP 建议**: 完成 **Phase 1–5** 即可在笔记本中用模块完成「加载 → 划分 → 训练」闭环。

---

## Phase 6: User Story 4 — 预测分数 (Priority: P1)

**目标**: 满足 **spec 需求 4**：基于模型路径与预测特征表打分；`model_type` 默认 LGB；可选写回 parquet（`id_column` + 分数字段）。

**Acceptance Scenarios**:

1. **Given** US3 产出的 LGB 模型路径与对齐特征列的 DataFrame，**When** `predict_scores`，**Then** 返回长度与行数一致的得分向量。  
2. **Given** `output_path`，**When** `predict_and_save_parquet`，**Then** 输出文件仅含 id 与分数字段且可读回。  

**Edge Cases**: 预测集缺列（应失败并提示缺失特征）；`num_threads` 传入 None 时使用库默认。

### 实现

- [x] T027 [US4] 实现 `model_pipeline/predict_model.py`：`predict_scores`、`predict_and_save_parquet`，`model_type` 分发（首期仅 LGB）

### 单元测试（必须）

- [x] T028 [US4] 编写 `tests/test_predict_model.py`（缺列异常、线程参数）

### 验收测试（必须）

- [x] T029 [US4] 编写 `tests/acceptance/test_predict_acceptance.py`：加载 US3 验收生成的模型，对同一分布小样本预测并校验 AUC 非 NaN

- [x] T030 [US4] 验证本 Phase 覆盖率达标

**Phase Gate**: US4 全部通过，覆盖率达标  

**MVP 建议**: 完成 **Phase 1–6** 即可「训练 + 预测」离线闭环。

---

## Phase 7: User Story 5 — 超参网格与交叉验证/调参 (Priority: P2)

**目标**: 满足 **spec 需求 5**：`build_param_grid(model_type, grid_spec)` 生成候选；`tune`（或 `cross_validate`）在 train/val 或 K 折上按 metric 评估；**首期仅 LGB**；与 `train_model` 共用 `ModelType`。

**Acceptance Scenarios**:

1. **Given** 小网格（≥2 组参数），**When** 在固定随机种子的小数据集上 `tune`，**Then** 返回结果表含每组 metric 且能指出最优参数组合。  
2. **Given** `n_splits=1` 与固定 `val_df`，**When** `tune`，**Then** 行为等价于单次验证评估（在测试中定义断言）。  

**Edge Cases**: 空网格；`model_type` 非 LGB 未实现时报错；metric 非法。

### 实现

- [x] T031 [US5] 实现 `model_pipeline/tune_model_params.py`：`build_param_grid`、`tune`（内部可调用 **`lightgbm.cv`** 或自研循环 + `train_model.train`，与 `plan.md` 决策 3 一致）

### 单元测试（必须）

- [x] T032 [US5] 编写 `tests/test_tune_model_params.py`（网格笛卡尔积、空网格、非法 metric）

### 验收测试（必须）

- [x] T033 [US5] 编写 `tests/acceptance/test_tune_acceptance.py`：2×2 极小网格在几十行数据上跑通并返回 `TuningReport` / DataFrame

- [x] T034 [US5] 验证本 Phase 覆盖率达标

**Phase Gate**: US5 全部通过，覆盖率达标

---

## Phase 8: Polish & 横切关注点

**目的**: 结构整理、文档与覆盖率收口。

- [ ] T035 [P] （可选）将 LGB 训练实现迁至 `model_pipeline/backends/lgb.py`（文件名简称；语义为 **LGB** 后端）并由 `train_model.py` 引用，保持对外 API 不变
- [x] T036 全量运行 `pytest --cov=model_pipeline --cov-report=term-missing`，确保**整体**行/分支覆盖率满足 gate
- [x] T037 新增 `specs/model_pipeline/quickstart.md`：最小示例（load → split → train → predict），与 `plan.md` Design Artifacts 一致
- [x] T038 在 `model_train_predict.ipynb` 中增加一节**示例**：由 `import model_pipeline` 调用替代内联函数（或新建 `examples/train_pipeline_script.py`，二选一写入 notebook 注释说明路径）

**Phase Gate**: 全部测试通过，整体覆盖率达标，准备合并/交付

---

## 依赖与执行顺序

### Phase 依赖

- **Phase 1 → Phase 2 → US1 → US2 → US3 → US4 → US5 → Polish**（**US1 与 US2 不并行**：须先完成 US1（Phase 3）再开始 US2（Phase 4）；US3 依赖 `split_data`，US4 依赖已保存模型，US5 依赖 `train_model`）

### 并行机会

- Phase 1 中 T003、T004 可并行  
- Phase 2 中 T008 与 T007 可并行（不同文件）  
- Phase 8 中 T035 与其他文档任务可并行  

---

## 实施策略

### MVP（最小可演示）

1. Phase 1–2  
2. Phase 3–6（US1–US4）  
3. 验收：**笔记本或脚本**可完成 加载 → 划分 → 训练 → 预测，无硬编码数据路径  

### 增量交付

- 加上 Phase 7（US5）→ 超参搜索能力  
- Phase 8 → 文档与示例收口  

---

## Notes

- **文件命名**: 超参模块为 **`tune_model_params.py`**（非 `cross_validation.py`）。  
- **术语**: 对外与文档统一 **LGB**（见 `spec.md`「Terminology」）；依赖包名仍为 **`lightgbm`**。  
- **XGB**: 任务中仅要求 **占位与 fail fast**；完整 XGB 训练/预测为后续里程碑，不在本 `tasks.md` 范围内。  
- 若仓库后续引入 `ruff`/`black`，可在 Polish 阶段追加格式化任务，不阻塞功能 Phase。  

---

## 报告统计（本文件生成时）

| 项 | 数量 |
|----|------|
| **总任务数** | 38（T001–T038） |
| **US1–US5 实现+测试任务** | 各 Story 含 实现 1 + UT 1 + 验收 1 + 覆盖率 1（US1–US2 略同结构） |
| **可并行 [P]** | **4** 个带 `[P]` 标记：T003, T004, T008, T035；**Phase 8**：T036、T037、T038 可与 T035 并行分配（T035 为可选重构，T037 无 `[P]` 但可与 T035 同时进行） |
| **建议 MVP 范围** | Phase 1–6（US1–US4） |

**建议下一步**: 可选运行 `analyze` 核对 spec / plan / tasks；再按 `.cursor/agents/implement.md` 或本地流程 **按 Phase 实现**，每 Phase 结束跑 pytest + coverage gate。
