<p align="center">
  <img src="assets/logo.svg" width="112" alt="knowledge_net logo">
</p>

<h1 align="center">knowledge_net</h1>

<p align="center">
  <em>Attention-read, gate-written key–value memory for sequence models.</em>
</p>

<p align="center">
  <strong>Small, readable PyTorch notebooks that build the mechanism and test it on tasks with exactly computable answers.</strong>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  Research archive, frozen February 2026 · no published results · AGPL-3.0-only
</p>

Can a handful of memory slots hold discrete bindings better than one opaque hidden state? These notebooks build an attention-read, gate-written memory bank in PyTorch and test it on tasks where the correct answer is exactly computable: pointer chains, synthetic arithmetic strings, and byte-level Chinese next-symbol prediction.

This is a **research sketch, not a library and not a novel architecture**. The mechanism belongs to the Neural Turing Machine / Memory Networks lineage, and the append-only variant is close to attention over a growing KV cache. There is no baseline, ablation, or comparison in the repository. The honest status, evidence, and limitations live in [`docs/project-status.md`](docs/project-status.md).

## At a glance

| | |
|---|---|
| **Status** | Archived research sketch; not maintained, no releases |
| **What is here** | 11 Jupyter notebooks + one importable module (`kb.py`) |
| **Core idea** | A learned bank of key–value slots, read by attention and written through a learned gate |
| **Tasks** | Pointer chains, synthetic arithmetic strings, byte-level Chinese next-symbol prediction |
| **Results** | No reproducible results; notebooks have no saved outputs, only embedded curve screenshots |
| **Requirements** | PyTorch, NumPy, Matplotlib, Jupyter, and a CUDA GPU |
| **License** | [AGPL-3.0-only](LICENSE) |

## Quick start

There is no package, CLI, `main`, test suite, or `requirements.txt`. The notebooks are meant to be opened and run top-to-bottom in Jupyter.

```bash
git clone https://github.com/ZisIsNotZis/knowledge_net.git
cd knowledge_net
pip install torch numpy matplotlib jupyterlab
jupyter lab
```

Then pick a notebook:

- **`Untitled.ipynb`** — the fastest self-contained starting point. It defines the synthetic pointer-chain task in-notebook, so it needs no external corpus. It runs end-to-end but is slow: the 1024-step Python-level memory loop takes roughly 15 seconds per optimization step on the machine where it was checked.
- **`Untitled-Copy8.ipynb`** — the canonical, most complete snapshot (`KbCell` / `Kb` / `Akb`, data pipeline, training curves). It does **not** run as-is: it expects `../nlpdat/inews/{train,dev,test}.txt`, which is not in this repository.
- **`Untitled-Copy5/6.ipynb`** — data-free variants that generate their own synthetic arithmetic-string batches. **`Untitled-Copy7.ipynb`** trains the same self-generated task, but its trailing cells poke at the missing `../nlpdat` corpora, so run only the cells above them.

> **CUDA is required by default.** Every notebook begins with `torch.set_default_tensor_type(torch.cuda.FloatTensor)`. On newer PyTorch releases that path is deprecated or removed, so this is usually the first line to change.

## What it can do now

- **Fixed-size memory banks** — `KbCell` reads from a slot bank with an attention module (`Notice`), mixes the read with the input, emits a per-slot write gate, and updates the bank.
- **Append-only memory** — `Akb` starts empty, reads over all earlier slots at each step, and appends a new slot.
- **Bidirectional memory attention** — the earlier `Bitention` / `Kb` form in `kb.py` reads and writes attention between the input stream and memory.
- **Sequence encoders and decoders** — `Seq2Vec`, `Vec2Seq`, `Seq2Seq`, residual `Res` / `Exp` blocks, `ConvSpace`, and a `Linear` subclass that accepts multiple inputs.
- **Toy task generators** — pointer chains, phrased `qa` bindings, and synthetic arithmetic strings defined inside the notebooks.
- **A byte-level Chinese pipeline** — `Dat`, `DatNoL`, and `DatMock`, plus logging and entropy helpers.

What it does **not** provide: a package, a stable API, tests, configs, checkpoints, a reproducible training script, or a held-out evaluation.

## Architecture

The core loop is deliberately small:

```text
                ┌─────────────── read ───────────────┐
                │                                    │
input x_t ──► mix ──► output y_t          memory slots [k_1 … k_n]
                │                                    ▲
                └─────────────── write gate ─────────┘
                         new slot / gated update
```

The read head scores every memory slot by concatenating the query, the slot, and a position signal, passing them through an MLP, and softmaxing over the slot axis. The write path produces a gate and a candidate slot; the fixed-size variant blends the candidate into the bank, while `Akb` appends it. Encoders and decoders sit around this loop when the task needs sequence-to-sequence modeling.

For the concrete definitions, start with [`Untitled-Copy8.ipynb`](Untitled-Copy8.ipynb); for the extracted module version, read [`kb.py`](kb.py).

## How to develop

There is no build step. Work directly in Jupyter, and treat each notebook as an experiment snapshot rather than a module.

Suggested reading order — the idea's development:

1. [`Untitled.ipynb`](Untitled.ipynb) — earliest complete form: `Bitention` + `Kb` on the pointer-chain task.
2. [`kb.py`](kb.py) — the same idea extracted into one importable file.
3. [`Untitled-Copy4.ipynb`](Untitled-Copy4.ipynb) — `Seq2Vec` / `Vec2Seq` + `Kb` on a phrased `qa` task.
4. [`Untitled-Copy5.ipynb`](Untitled-Copy5.ipynb), [`Untitled-Copy6.ipynb`](Untitled-Copy6.ipynb), [`Untitled-Copy7.ipynb`](Untitled-Copy7.ipynb) — refactor to `KbCell` / `Kb`, evaluated on synthetic arithmetic strings.
5. [`Untitled-Copy8.ipynb`](Untitled-Copy8.ipynb) — final snapshot: `Akb` on byte-level Chinese next-symbol prediction, plus the recorded curves.

Before changing code, know the known rough edges: `kb.py` drops the memory-length argument (`s.l = h`), several loops shadow their iterators, `Untitled-Copy1.ipynb` calls an undefined `qas(...)`, and `DatMock` lacks `__len__` even though the model builder calls `len(a)`. These are documented, not hidden.

## How to contribute

The repository is archived, so there is no active maintenance or release process. Issues and forks are welcome, especially:

- reproducing or replacing the missing `inews` corpus,
- saving notebook outputs so claims can be checked,
- adding a small test that pins the memory-bank update rule,
- reviving the sketch as a clean, CPU-runnable experiment.

If you revive it, keep the project's best habit: every claim should be computable or reproducible.

## Roadmap

**Implemented**

- [x] Fixed-size learned memory bank with attention reads and gated writes
- [x] Append-only attention memory (`Akb`)
- [x] `Bitention` / `Kb` module extraction in `kb.py`
- [x] Synthetic pointer-chain, binding, and arithmetic-string tasks
- [x] Byte-level Chinese data pipeline and training loops
- [x] Embedded curve screenshots from early runs

**Planned**

- [ ] Nothing while the repository is archived. There is no active roadmap, no promised release, and no maintained API.

**Future vision** — If revived, the useful next step is not a bigger model but a smaller, checkable one: unify the near-duplicate notebooks into a single runnable experiment, restore the missing data or replace it with a generated corpus, add a held-out evaluation, and compare the bank against a plain recurrent state and a KV-cache baseline. That would turn a promising sketch into evidence.

## Provenance and license

The notebooks and `kb.py` are the author's own work; there is no vendored third-party code. The repository was migrated from a local read-only archive, and no secrets or absolute home-directory paths are present.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0-only`). There are no license headers in the individual source files.
