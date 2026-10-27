# knowledge_net project status

## Classification

Research scratch / sketch. A personal notebook directory (plus one module) exploring a
single mechanism — a small learned key–value **memory bank** that a sequence model reads
from by attention and writes to through a learned gate — on toy, exactly-checkable tasks
and on byte-level Chinese next-symbol prediction. It is an experiment on a **known**
family of ideas (attention-based external memory: NTM / DNC / Memory Networks; the
append-only variant is close to re-deriving attention over a growing KV cache), not a
novel architecture, and not a finished experiment either.

## Status

Dead / archived. Frozen at Feb 2026 (notebooks) and Dec 2024 (`kb.py`); not maintained,
never written up, no releases. Development consisted of manually forking
the notebook forward — `Untitled.ipynb` → `Untitled-Copy1..8b.ipynb` — with each fork
carrying small architectural edits. Two lineages are visible: an early `Bitention`/`Kb`
form (`Untitled`, `Copy1`–`Copy4`, `kb.py`) and a later `KbCell`/`Kb`/`Akb` form
(`Copy5`–`Copy8b`). The end state is `Untitled-Copy8.ipynb`.

## Evidence

**Architecture (implemented, executable).**

- `Untitled-Copy8.ipynb` — canonical and most complete: `Nn`, `Linear` (multi-input via
  `Cat`), `BatchNormFlat`, `Notice`, `KbCell`, `Kb`, `Akb`, `Red`, `Res`, plus the
  `Dat` / `DatNoL` / `DatMock` pipeline, `Log`, and `entropy`/`entropy2` helpers.
- `Untitled.ipynb` — the only fully self-contained end-to-end example (synthetic
  pointer-chain task defined in-notebook). Verified here to execute (forward, backward,
  optimizer step) without external data, but the 1024-step Python-level `Kb` loop makes
  it ~15 s per optimization step, so it was not run to convergence.
- `kb.py` — the idea extracted into a single importable module
  (`Linears`, `F`, `Lift`, `Nn`, `Notice`, `Bitention`, `Kb`).
- `Untitled-Copy5/6/7.ipynb` — a `Seq2Vec` / `Kb` / `Vec2Seq` `Seq2Seq` stack trained on
  self-generated synthetic arithmetic strings (`dat(l, n)`), so also data-free.

**Recorded results (weak).**

- Three hand-attached training-curve PNGs in `Untitled-Copy8.ipynb` cell 5 and one in
  `dat.ipynb` cell 2; one hand-pasted list of 16 unlabelled numbers in
  `Untitled-Copy8.ipynb` cell 3 (values −7.67 … −19.80, which cannot be cross-entropy).
- Re-executed here: the `DatMock` path runs, but `DatMock` emits uniformly random
  sequences, so its ~5.62 nat floor is uninformative about learning. `dat.ipynb`'s
  no-memory baseline on real news barely moved (5.038 → 5.025, accuracy 0.067 → 0.0685).

**Absent.**

- No write-up, results table, test-set metric, baseline, ablation, seed, or log.
- No stored notebook outputs: all 11 notebooks have `execution_count == 0` in every code
  cell and zero stored outputs, so the screenshots are the entire result record.
- No data: `../nlpdat/inews/{dev,test,train}.txt` is referenced by five notebooks and
  exists neither in this directory nor elsewhere on the archive, so the principal runs
  cannot be reproduced.
- No checkpoints, weights, configs, `requirements.txt`, tests, or CI.

## Boundaries

- **Unclaimed novelty.** Nothing in the directory cites, compares against, or even names
  the prior art it resembles (NTM/DNC/Memory Networks/linear attention). Do not present
  `Kb`/`Akb` as a new architecture.
- **Do not cite the curves as results.** They are unlabelled screenshots from runs whose
  data is gone, with no held-out evaluation.
- **Requires CUDA and edits to run.** Every notebook starts with
  `torch.set_default_tensor_type(torch.cuda.FloatTensor)`; it still works on torch 2.14
  with a deprecation warning but is removed/deprecated on newer torch. No entrypoint, no
  packaging — notebook-only.
- **Open source under AGPL-3.0-only.** The repository ships the full
  [`LICENSE`](../LICENSE) text. Individual source files carry no license headers.
- **Known rough edges** (representative): `kb.py`'s `Kb.__init__` sets `s.l = h`, dropping
  the memory-length argument; iterator shadowing and `l` reused for loss/length in several
  loops; `Untitled-Copy1.ipynb` calls `qas(...)` without defining it; `DatMock` lacks
  `__len__` though the builder calls `len(a)`; `Untitled-Copy3.ipynb` hardcodes
  `p = torch.tensor([0.9])` over the model's own probability output.
- **Third-party code:** none. All files are the author's own; there are no vendored
  dependencies and no `nn.py` to preserve (no notebook imports a local `nn` module).
- **Sanitisation:** credential gate returns nothing; no absolute, user-specific
  home-directory paths existed in any copied file, so none needed genericising.
