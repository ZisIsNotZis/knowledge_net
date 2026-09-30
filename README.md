# knowledge_net

**What:** A scratch research directory (notebooks, Feb 2026; one module from Dec 2024)
exploring a single idea — give a sequence model a small **learned key–value memory bank**
that it reads from by attention and writes to through a learned gate — and test it on
tasks whose correct answer is *exactly computable*, plus a byte-level Chinese
next-symbol prediction task.

**Why it exists / the problem it was attacking:** the usual ways a sequence model
carries state are either a single opaque hidden vector (GRU, conv stack) or a KV cache
that grows with the input (attention). This directory asks whether a small explicit
bank of memory slots, read selectively by attention and updated by a gate, can hold and
retrieve discrete bindings (`k → v`, "is `k` still `v`?", follow a pointer chain)
instead of an opaque state. The tasks are deliberately toy and verifiable so that
success or failure is unambiguous.

**Status:** research / archive. Frozen as of Feb 2026; not maintained, never published.
See `docs/project-status.md` for the classification and the evidence behind it.

**Is it a novel architecture?** No. Stated plainly: this is **a sketch of a known family
of ideas**, not a novel architecture and not even a complete experiment. The mechanism
is an attention-based external memory — the Neural-Turing-Machine / Memory-Networks
lineage (a read head over a slot bank plus a learned write gate), and the final `Akb`
variant, which appends a slot every step and attends over everything so far, is close to
re-deriving attention over a growing KV cache. There is **no citation, no baseline, no
ablation, and no comparison** to any of that prior work anywhere in the directory, so
nothing here establishes novelty. It is **a pile of near-duplicate `Untitled-CopyN`
experiments** — nine manual forks of an `Untitled.ipynb` original (11 notebooks in total
including `dat.ipynb`) that were edited forward one after another (not byte-identical
duplicates — each first cell differs; only 6 cell bodies are byte-identical across
notebooks, all within the closest sibling pairs `Copy6`/`Copy7`, `Copy2`/`Copy3` and
`Untitled`/`Copy1`). It never got past toy training runs, and the runs that were done were
not saved with their outputs.

## How to run

**There is no single entrypoint.** No package, no CLI, no `main`, no tests, no configs,
no `requirements.txt`. It is notebooks only, and they are meant to be run top-to-bottom
in Jupyter.

```bash
pip install torch numpy matplotlib
jupyter lab          # or: jupyter notebook
```

- **A CUDA GPU is required.** Every notebook's first cell runs
  `torch.set_default_tensor_type(torch.cuda.FloatTensor)`. On the torch installed here
  (2.14) this still works but emits a deprecation warning; newer torch releases have
  removed the CUDA default-tensor-type path, so this call is the first thing that will
  need editing.
- **Canonical notebook: `Untitled-Copy8.ipynb`.** It is the last and most complete
  snapshot: `Nn` / `Linear`(over `Cat`) / `Notice`, `KbCell` / `Kb` / `Akb`, `Red` / `Res`,
  the `Dat` / `DatNoL` / `DatMock` data pipeline, `Log`, and the **only recorded training
  curves** in the directory. Its 277 KB is almost entirely three embedded PNGs; the code
  is ~8.5 KB. It **will not run as-is**: `Dat` loads `../nlpdat/inews/dev.txt`, and cell 4
  additionally needs `../nlpdat/inews/{test,train}.txt`. That corpus is **not in this
  directory and not anywhere on the archive** — see below.
- **Self-contained alternative: `Untitled.ipynb`.** The cleanest end-to-end example: the
  synthetic pointer-chain task (`qa` / `qas` / `gets` / `unpack`) is defined inside the
  notebook, so it needs no external data. Verified to execute here (forward, backward and
  an optimizer step all run). Note that as written it is **impractically slow**: `Kb`
  loops over the 1024-step sequence in Python with autograd, which measured ~15 s per
  optimization step on this machine, i.e. hours for the 10,000 steps the scheduler asks
  for — so I did not run it to convergence, and the first step is at chance. That cost is
  a plausible reason no results were ever recorded for this notebook.
- `Untitled-Copy5/6/7.ipynb` are also data-free — they generate their own synthetic
  arithmetic-string batches with `dat(l, n)`.

Suggested reading order (the idea's development, chronologically by file mtime all are
Feb 2026; `kb.py` is Dec 2024 and is the odd one out):

1. `Untitled.ipynb` — earliest complete form: `Bitention` + `Kb` on the pointer-chain task.
2. `kb.py` — the same idea extracted into one importable file (see the `cznn` note below).
3. `Untitled-Copy4.ipynb` — `Seq2Vec` / `Vec2Seq` + `Kb` on a phrased `qa` task.
4. `Untitled-Copy5/6/7.ipynb` — refactor to `KbCell` / `Kb`, evaluated on synthetic
   arithmetic strings.
5. `Untitled-Copy8.ipynb` — final: `Akb` (append-only attention memory) on byte-level
   Chinese next-symbol prediction, plus the recorded curves.

### The architecture, concretely (`Untitled-Copy8.ipynb`)

- `Notice(q, k, h, head)` — attention over memory slots: concatenates `[q, k, position]`
  (`position = torch.linspace(0, 1, k.size(-2))`) through an MLP and softmaxes over the
  slot axis.
- `KbCell(i, h, o, headin, headout)` — `read` (`Notice`) reads the bank; `compute` (MLP)
  mixes the input with the read; `write` (`Notice`) emits a per-slot gate; `save` (MLP)
  produces the new slot. Update is `x * w + (1 - w) * a`.
- `Kb(i, h, o, l, ...)` — fixed-size bank `torch.zeros(l, h)`, consumed step by step.
- `Akb(i, h, o, head)` — append-only variant: the bank starts empty, each step reads over
  all earlier slots, emits an output plus a new slot, and appends it.
- `Bitention` (early form, in `Untitled*`, `kb.py`) — bidirectional read/write attention
  between the input stream and the memory.
- Encoders/decoders around the bank: `Seq2Vec` (Conv1d + pooling, or GRU), `Vec2Seq`
  (ConvTranspose1d upsampling with residual `Exp` blocks), `Seq2Seq`, `Red`, `Res`,
  `ConvSpace`, and a `Linear` subclass that accepts a `Cat` (multi-input linear =
  block-sliced weight).

## What is here vs. not

**Here**

- 11 `.ipynb` notebooks + 1 `.py` module (`kb.py`), 12 files, ~1.5k lines of effective code
  (the largest file is large only because of embedded PNGs).
- The architecture definitions, the toy-task generators, and the training loops.
- Four pasted training-curve screenshots (three in `Untitled-Copy8.ipynb`, one in
  `dat.ipynb`) and one hand-pasted list of 16 unlabelled numbers
  (`Untitled-Copy8.ipynb` cell 3, values −7.67 … −19.80 — note cross-entropy cannot be
  negative, so these are not self-explanatory).

**Not here**

- **No written-up results.** No paper, no report, no results table, no test-set metric, no
  baseline, no ablation, no seeding, no logs.
- **No reproducible run records.** In every one of the 11 notebooks, every code cell has
  `execution_count == 0` and **zero stored outputs**. The notebooks were saved without
  outputs, so the only results are the hand-attached screenshots above. Nothing here can
  be checked by re-opening a notebook.
- **No data.** Five notebooks depend on `../nlpdat/inews/{dev,test,train}.txt`. That
  corpus is absent from this directory and I could not find it anywhere on the archive —
  so the main training runs, including the ones whose curves are pasted in, **cannot be
  reproduced**.
- **No checkpoints, weights, configs, environment files, or tests.** No `LICENSE`.

**What the screenshots appear to show** (unlabelled, no test set, so treat as anecdote,
not result): one byte-level next-symbol run improving loss ≈4.7 → 2.6 with per-position
accuracy ≈0.6; one run loss ≈9.7 → 6.2 with per-position accuracy ≈0.9; one plateaued at
loss ≈2.5 / accuracy ≈0.6. The one run I could re-execute is the data-free `DatMock`
path: it confirms the graph runs end-to-end, but `DatMock` generates **uniformly random**
sequences, so ~5.62 nats (≈ `ln 256` = 5.545) with ~0 accuracy is the expected floor and
says nothing about learning. `dat.ipynb`'s no-memory baseline on real Chinese news barely
moved (loss 5.038 → 5.025, accuracy 0.067 → 0.0685), i.e. it did not beat a unigram prior.

**Rough edges worth knowing before reading the code** (a representative sample, not a
review): `kb.py`'s `Kb.__init__(s, i, h, l)` sets `s.l = h` — the memory length argument is
silently dropped. Several loops shadow their own iterator (`for t in range(...): for t in
range(100)`) and reuse `l` for both "loss" and "length". `Untitled-Copy1.ipynb` calls
`qas(...)` without defining it. `DatMock` has no `__len__` even though the model builder
calls `len(a)`. `Untitled-Copy3.ipynb` contains `p = torch.tensor([0.9])` pinned over the
model's own probability output. `Untitled-Copy3/7.ipynb` contain `%reset -f in out` and
stray scratch cells (`ord('~')`, `a[[1,2,3,4]] = 4`).

## Relationship to the author's published `cznn`

**No notebook here imports a local `nn` module, and there is no `nn.py` in this directory
— nothing was deleted.** The dependency runs in neither direction.

The stylistic lineage is real, though. `kb.py` and the notebooks use the same author's
signature idiom: `s` for `self`, `from torch.nn import *`, `Sequential` subclasses with a
one-line `forward`, a `Cat`/`Linear` pair for multi-input linear layers, and `F` as a
callable-wrapper module. Those same symbols appear in the author's published library
[`cznn`](https://github.com/ZisIsNotZis/cznn), specifically in its verbatim `neuronloop`
fork (`cznn/variants_neuronloop.py`: `F`, `Map`, `Cat`, `Nn`, …). But this is **shared
style and shared vocabulary, not shared code**: the concrete definitions differ (for
instance `cznn`'s `Map` is a `Sequential` composition wrapper; here `Map` is a
character/byte tokenizer), and `cznn` contains **no** `Kb`, `Akb`, `KbCell`, `Bitention`
or `Notice` at all. Treat `knowledge_net` as an **independent sibling** of the `cznn`
idiom rather than a consumer of `cznn`.

## Provenance and license

- Source of truth for this copy: the read-only archive directory
  `<archive>/knowledge_net`, copied with `rsync -a` (excluding `.git`, `__pycache__`,
  `*.pyc`, `.venv`, `.ipynb_checkpoints`). The copy is byte-identical to the source.
- **License: none.** There is no `LICENSE` file in the source directory and no license
  header in any file, so the default applies — all rights reserved. This is a public
  *repo layout*, not a grant of rights. Do not treat it as open source.
- **Sanitisation:** the credential gate over this directory returns nothing, and there
  were **no absolute, user-specific home-directory paths in any copied file** (all paths
  in the code are relative, e.g. `../nlpdat/inews/dev.txt`), so nothing needed
  genericising. All paths in this README and in `docs/project-status.md` are written
  relative or as `<archive>/...` placeholders on purpose.
