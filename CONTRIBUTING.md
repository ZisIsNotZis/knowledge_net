# Contributing

This repository is a frozen research sketch, not a maintained library. It takes fixes
and evidence, not features: the value here is that every claim can be checked. If you
want to revive it as a real experiment, read [`docs/project-status.md`](docs/project-status.md)
first — it lists exactly what is missing (data, outputs, baselines, a held-out evaluation).

## What a change may touch

- One notebook, one idea. The notebooks are forked forward (`Untitled.ipynb` →
  `Untitled-Copy8b.ipynb`); add a new copy rather than rewriting an existing snapshot,
  because the old ones are the record.
- `kb.py` and `docs/` take corrections of fact.
- Do **not** add dependencies, packaging, or a `requirements.txt` in a drive-by PR; open an
  issue with the reason first. Every dependency is one more thing a reader has to install
  before they can check a claim.

## Before opening a pull request

There is no test suite, linter, or CI in this repository, and nothing here is run
automatically. So the pull request has to carry its own evidence:

1. Say which notebook (and which cells) you ran, top to bottom or not.
2. Say which PyTorch version and whether it ran on CUDA or CPU. Every notebook opens with
   `torch.set_default_tensor_type(torch.cuda.FloatTensor)`; a CPU run means you changed that
   line, and the pull request should show the change.
3. If you attach outputs or a curve, say what produced them and over how many steps. An
   unlabelled screenshot is not evidence — that is the main flaw in this repository.
4. Keep the diff to one subject. A whitespace reflow of a notebook makes the real change
   unreadable.

## Issues

Useful: a traceback with the notebook name, torch version, and the cell that failed; a
replacement for the missing `../nlpdat` corpora; a measured number where a claim is vague.

Not useful: "does not run" without the error, or a request to make it a package.

## Licensing

Contributions are accepted under [AGPL-3.0-only](LICENSE), the license of the repository.
