# 非理想成像下牛顿环的稳健几何恢复：复现包

本包收录论文“Robust geometric recovery of Newton’s rings under non-ideal imaging”
所用的数据、冻结结果和代码。
无需 GPU 亦可核对本文的主要数据，也可从实测帧重新生成结果。

English version: [README.md](README.md)

## 运行环境

- Python 3.11（论文表格均在 3.11 下生成）
- 依赖版本见 `requirements.txt`：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

GPU 非必需的，仅用于加速 MA-Net 推理。部分脚本的终端输出为中文，对结果无影响

## 一键复现

自动创建 `.venv`、安装 `requirements.txt`、跑如下四条核对，并使日志出现 `Table 1 match.`。成功时会打印 **全部通过**。

```bat
reproduce.bat
reproduce.bat --with-manet
```

```bash
chmod +x reproduce.sh
./reproduce.sh
./reproduce.sh --with-manet
```

`--with-manet` 会再跑一帧 MA-Net 拟合（`run_comparison_now.py --limit 1`），需要 `weights/50distill_last.pt`。日志在 `results/reproduce_logs/`。

## 快速核对（无需 GPU 和权重）

与一键脚本相同的四条命令：

```bash
python scripts/reproduce_tables.py
python code/eval/stats.py
python code/bench/run_score_sensitivity.py
python code/eval/eval_c1_c4_reviews.py --limit 1
```

`reproduce_tables.py` 对冻结的圆名单重新判分，最后应输出 `Table 1 match.`，同时重算 3.6 节的多透镜半径。
`stats.py` 从同一批文件重算 Wilson 区间、精确 McNemar 检验、Wilcoxon 检验以及 Holm 校正。

## 目录结构

```
data/comparison/       93 帧标注数据：分析画布、原始帧（raw/）、子集 sidecar、
                       人工极点与暗环标注、暗纹带标注
data/stationary/       40 帧静止序列（表 7–8）
data/dual_blind_b15/   第二标注者的数据，15 帧开弧（第 3.2 节）
data/multi_lens/       五次多透镜试验的环列表（第 3.6 节）
results/fits/          四条路线的冻结圆名单
results/timing.json    逐帧耗时（表 6）
results/static.json    静止序列拟合结果（表 7–8）
results/easy_translate/     平移实验，d = 0–200 px（第 3.3 节）
results/score_sensitivity/  评分常数 ±20% 扰动（第 4 节）
results/hough_oracle/       Hough 24 组参数扫描（第 2.5 节）
results/c1_c4/              蒙特卡洛极点扰动；γ 缩放与消融
results/dual_blind/         两位标注者的极点差
code/vision/           三段式求解器与背景减除前端
code/newmodel/         MA-Net 推理
code/bench/            判分、对照路线（Hough、FRFT 型）、各实验入口
code/eval/             统计检验与蒙特卡洛评估
docs/criteria.md       规则定义
weights/               MA-Net 推理权重（单独发布，见 weights/README.md）
```

结果文件中的路线名与正文的对应关系：

| 文件名 | 正文中的路线 |
|--------|--------------|
| `classical` | 背景减除前端 + 三段式求解器 |
| `v1.3` | MA-Net 前端 + 三段式求解器 |
| `Hough` | 圆 Hough 变换 |
| `FRFT` | 全局 r² 谱（FRFT 型 / r²DFT）搜索 |

## 正文条目 → 命令 → 产物

| 正文条目 | 命令 | 产物 |
|----------|------|------|
| 表 1（0.5 Λ / 1 Λ / 2 Λ / 不测极点下的通过数） | `python scripts/reproduce_tables.py` | 由 `results/fits/*` 重新判分 |
| 表 2–5、表 9 | 冻结结果（`reproduce_tables.py` 只检查文件是否存在） | `results/fits/`、`results/c1_c4/` |
| 表 6 | 冻结结果 | `results/timing.json` |
| 表 7–8 | 冻结结果 | `results/static.json`、`data/stationary/` |
| 第 3.2 节 第二标注者（中位 0.54 Λ，1 Λ 内 73.3%） | `python code/bench/score_dual_blind.py` | `results/dual_blind/annotator_AB_pole_delta.json` |
| 第 3.2 节 蒙特卡洛极点扰动（种子 42） | `python code/eval/eval_c1_c4_reviews.py` | `results/c1_c4/part1_results.json` |
| 第 3.3 节 平移实验 | `python code/bench/run_easy_translate.py`（需要权重） | `results/easy_translate/` |
| 第 3.6 节 γ 缩放 | 冻结结果 | `results/c1_c4/part2_results.json` → `radial_scaling_generalization` |
| 第 3.6 节 多透镜 R | `python scripts/reproduce_tables.py` | `data/multi_lens/confidence_db_multilens.json` |
| 第 4 节 评分常数敏感性 | `python code/bench/run_score_sensitivity.py` | `results/score_sensitivity/` |
| 第 2.5 节 Hough 参数扫描 | `python code/bench/run_hough_oracle.py` | `results/hough_oracle/` |
| Wilson / McNemar / Wilcoxon / Holm | `python code/eval/stats.py` | 由 `results/fits/` 现场计算 |

## 重新生成拟合结果

`code/bench/` 下的脚本都通过 `code/bench/pkg_paths.py` 在包内定位路径。不跑 MA-Net、只跑一帧 smoke 测试：

```bash
python code/bench/run_comparison_now.py --limit 1 --skip-manet
python code/bench/run_v162_timing_static.py --limit 1 --skip-manet
```

smoke 写入 `results/smoke_*`，不改动冻结结果。
完整运行会重新生成表 1–5 的圆名单（`run_comparison_now.py`），或表 6–8 的耗时和静止序列拟合（`run_v162_timing_static.py`），并写入 `results/`，覆盖冻结副本。
MA-Net 路线和平移实验需要把权重放在 `weights/50distill_last.pt`。
耗时取决于机器，表 6 的数值不会完全一致。
MA-Net 重跑的结果也可能因 GPU 和库版本与冻结结果有亚像素级差异（例如帧 000000 的极点相差 0.01 px），在检查过的帧上不影响通过与否的判定。

## MA-Net

只提供蒸馏学生网络的轻量包（`weights/50distill_last.pt`，约 607 MB / 0.565 GiB；
MD5 `5941203ce7b5384ba7a389c0a2606c83`）。去掉了教师权重、优化器状态和训练元数据；与完整训练
checkpoint 在单帧上逐元素一致（见 `weights/README.md`），以合乎 GitHub 单文件 2G 的上限。训练代码与 333 张训练帧不收录。
学生推理使用 `code/newmodel/models/baseline.py`。

## 仪器常数

| 量 | 数值 |
|----|------|
| 波长 λ | 589.3 nm（钠 D 线） |
| 像素比例尺 p | 7.189 µm/像素（768 × 768 分析画布） |

两者皆为仪器软件使用的工作常数，不是对中继光路的独立计量标定，
故由它们换算的透镜半径与正文一样，只作合理性检验。

## 帧号与子集

- 文件名在可用时使用 6 位帧号（如 `000018_Open-arc_canvas.png`），正文缩写为 4 位（帧 0018）
- 子集（`Easy`、`Open-arc`、`Illegible`）以各 sidecar `*.json` 的 `subset` 字段为准，不看文件名。有 13 个文件名为 `*_Open-arc*` 的帧标为 `Illegible`。

## 多透镜试验（第 3.6 节）

五次试验导出自仪器软件的结果记录。该记录保存每次拟合的极点和环列表，不保存干涉图。每次试验的透镜半径为

R = 相邻级 Δ(r²) 的中位数 × p² / λ，

其中 r 是 `fit.rings` 中以像素计的环半径，p 是像素比例尺。
正文报告的是每块样品两次试验的中位数（1.091 m 和 1.075 m），以及换回原透镜后的结果（1.916 m）。
`scripts/reproduce_tables.py` 会打印五次试验各自的数值。
字段 `R_m_from_platform` 是仪器软件按入库记录平均得到的值，不是正文使用的数值。

## 未收录的内容

- MA-Net 训练数据、训练代码和教师网络

## 许可

- 代码（`code/`、`scripts/`、`reproduce.bat`、`reproduce.sh`）：GNU GPL v3.0 或更高版本，见 [LICENSE](LICENSE)。
  可以使用、修改和再发布（包括商用），但发布的衍生作品必须以同一许可证开放源代码。
- 数据、结果和推理权重：CC BY 4.0，见 [LICENSE-DATA](LICENSE-DATA)

## 引用

引用信息见 [CITATION.cff](CITATION.cff)。
