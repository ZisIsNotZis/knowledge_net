<p align="center">
  <img src="assets/logo.svg" width="112" alt="knowledge_net 标志">
</p>

<h1 align="center">knowledge_net</h1>

<p align="center">
  <em>面向序列模型的注意力读取、门控写入键值记忆。</em>
</p>

<p align="center">
  <strong>用少量可读的 PyTorch notebook 搭出机制，并在答案可精确计算的任务上检验它。</strong>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  研究存档，冻结于 2026 年 2 月 · 没有已发表结果 · AGPL-3.0-only
</p>

少量记忆槽能否比一个不透明的隐藏状态更好地保存离散绑定？这些 notebook 用 PyTorch 搭出一个「注意力读取、门控写入」的记忆库，并在答案可以精确计算的任务上检验它：指针链、合成算术字符串，以及字节级中文下一符号预测。

这是一份**研究草稿，不是库，也不是新架构**。该机制属于 Neural Turing Machine / Memory Networks 一脉，追加式变体已经接近在不断增长的 KV cache 上做注意力。仓库里没有基线、消融或对比实验。如实的状态、证据和局限见 [`docs/project-status.md`](docs/project-status.md)。

## 概览

| | |
|---|---|
| **状态** | 已归档的研究草稿；不再维护，没有发布版本 |
| **仓库内容** | 11 个 Jupyter notebook + 1 个可导入模块（`kb.py`） |
| **核心想法** | 一个可学习的键值槽记忆库，由注意力读取，由学习到的门控写入 |
| **任务** | 指针链、合成算术字符串、字节级中文下一符号预测 |
| **结果** | 没有可复现结果；notebook 未保存输出，只有内嵌的曲线截图 |
| **运行要求** | PyTorch、NumPy、Matplotlib、Jupyter，以及一块 CUDA GPU |
| **许可证** | [AGPL-3.0-only](LICENSE) |

## 快速开始

这里没有包、命令行入口、`main`、测试套件或 `requirements.txt`。notebook 需要在 Jupyter 中自上而下运行。

```bash
git clone https://github.com/ZisIsNotZis/knowledge_net.git
cd knowledge_net
pip install torch numpy matplotlib jupyterlab
jupyter lab
```

然后选择一个 notebook：

- **`Untitled.ipynb`** —— 最快的自包含入口。指针链任务直接定义在 notebook 内，不需要外部语料。它能端到端运行，但很慢：在这台机器上，1024 步的 Python 级记忆循环大约需要 15 秒一个优化步。
- **`Untitled-Copy8.ipynb`** —— 最完整、最典型的快照（`KbCell` / `Kb` / `Akb`、数据管线、训练曲线）。它**不能直接运行**：需要 `../nlpdat/inews/{train,dev,test}.txt`，而这份语料不在仓库里。
- **`Untitled-Copy5/6/7.ipynb`** —— 不依赖外部数据的变体，自己生成合成算术字符串批次。

> **默认需要 CUDA。** 每个 notebook 开头都是 `torch.set_default_tensor_type(torch.cuda.FloatTensor)`。较新的 PyTorch 已弃用或移除这条路径，所以这通常是最先要改的一行。

## 现在能做什么

- **固定大小记忆库** —— `KbCell` 通过注意力模块（`Notice`）从槽记忆库读取，将读出的内容与输入混合，生成每个槽的写门控，并更新记忆库。
- **追加式记忆** —— `Akb` 从空记忆开始，每一步对所有更早的槽做注意力，并追加一个新槽。
- **双向记忆注意力** —— `kb.py` 中较早的 `Bitention` / `Kb` 形式在输入流与记忆之间做双向读写注意力。
- **序列编码器与解码器** —— `Seq2Vec`、`Vec2Seq`、`Seq2Seq`、残差 `Res` / `Exp` 块、`ConvSpace`，以及接受多路输入的 `Linear` 子类。
- **玩具任务生成器** —— 指针链、带短语的 `qa` 绑定、合成算术字符串，都定义在 notebook 内。
- **字节级中文管线** —— `Dat`、`DatNoL`、`DatMock`，以及日志和熵辅助函数。

它**不**提供：包、稳定 API、测试、配置、checkpoint、可复现训练脚本或留出集评估。

## 架构

核心循环刻意保持很小：

```text
                ┌─────────────── 读取 ───────────────┐
                │                                    │
输入 x_t ───► 混合 ───► 输出 y_t          记忆槽 [k_1 … k_n]
                │                                    ▲
                └─────────────── 写门控 ─────────────┘
                         新槽 / 门控更新
```

读头把查询、记忆槽和位置信号拼接起来，经过 MLP 后对槽维度做 softmax，从而给每个槽打分。写路径生成一个门控和一个候选槽；固定大小变体把候选槽混合进记忆库，`Akb` 则把它追加进去。任务需要序列到序列建模时，编码器和解码器包在这个循环外面。

具体定义从 [`Untitled-Copy8.ipynb`](Untitled-Copy8.ipynb) 看起；想读抽出来的模块版本，打开 [`kb.py`](kb.py)。

## 如何开发

没有构建步骤。直接在 Jupyter 里改，把每个 notebook 当作实验快照，而不是模块。

建议阅读顺序（想法的演进）：

1. [`Untitled.ipynb`](Untitled.ipynb) —— 最早的完整形态：`Bitention` + `Kb`，跑指针链任务。
2. [`kb.py`](kb.py) —— 把同一想法抽成一个可导入文件。
3. [`Untitled-Copy4.ipynb`](Untitled-Copy4.ipynb) —— `Seq2Vec` / `Vec2Seq` + `Kb`，跑带短语的 `qa` 任务。
4. [`Untitled-Copy5.ipynb`](Untitled-Copy5.ipynb)、[`Untitled-Copy6.ipynb`](Untitled-Copy6.ipynb)、[`Untitled-Copy7.ipynb`](Untitled-Copy7.ipynb) —— 重构成 `KbCell` / `Kb`，在合成算术字符串上评估。
5. [`Untitled-Copy8.ipynb`](Untitled-Copy8.ipynb) —— 最终快照：`Akb` 做字节级中文下一符号预测，并附有记录的曲线。

改代码前先知道这些已知粗糙处：`kb.py` 丢掉了记忆长度参数（`s.l = h`），若干循环遮蔽了自己的迭代变量，`Untitled-Copy1.ipynb` 调用了未定义的 `qas(...)`，`DatMock` 没有 `__len__`，但模型构建代码会调用 `len(a)`。这些是记录在案的，不是藏起来的。

## 如何贡献

仓库已归档，没有活跃维护或发布流程。仍然欢迎 issue 和 fork，尤其是：

- 复现或替换缺失的 `inews` 语料，
- 保存 notebook 输出，让结论可以被检查，
- 加一个小测试，固定住记忆库的更新规则，
- 把这份草稿整理成一个干净、能在 CPU 上运行的实验。

如果复活它，请保留这个项目最好的习惯：每个结论都应当可计算或可复现。

## 路线图

**已实现**

- [x] 固定大小的可学习记忆库，注意力读取 + 门控写入
- [x] 追加式注意力记忆（`Akb`）
- [x] `kb.py` 中抽出的 `Bitention` / `Kb`
- [x] 合成指针链、绑定和算术字符串任务
- [x] 字节级中文数据管线与训练循环
- [x] 早期运行留下的内嵌曲线截图

**计划**

- [ ] 仓库归档期间没有计划。没有活跃路线图，没有承诺的发布版本，也没有受维护的 API。

**未来设想** —— 如果复活，有用的下一步不是更大的模型，而是更小、更可检查的实验：把近重复的 notebook 合并成一个可运行实验，恢复缺失数据或换成生成语料，加入留出集评估，并把记忆库与普通循环状态、KV-cache 基线做对比。那样才能把一份有苗头的草稿变成证据。

## 来源与许可证

notebook 和 `kb.py` 均为作者原创；没有引入第三方代码。仓库从本地只读存档迁移而来，不包含密钥或绝对家目录路径。

以 [GNU Affero General Public License v3.0](LICENSE)（`AGPL-3.0-only`）授权。各源文件内没有单独的许可证头。
