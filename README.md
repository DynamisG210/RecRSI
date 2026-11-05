# RecRSI

**Mechanism-Guided Discovery of Recommendation Algorithms**

RecRSI is an LLM-agent framework for discovering better recommendation algorithms. Given a starting model, a dataset, and an experiment budget, it proposes changes, implements candidate algorithms, and tests them through training and evaluation.

The search is organized around **mechanism hypotheses**: what to change, why it might help, and what the experiment should show.

- **MechSpace** builds a task-specific mechanism space from the starting model and research literature.
- **Research Line** develops hypotheses, implements candidates, and selects experiments.
- **Mechanism Feedback** connects experimental results to the tested hypotheses to guide the next round.

The default research task is SASRec on Amazon CDs and Vinyl. For graph recommenders, see the [task format](docs/task-format.md).

## Quick start

Download or clone this repository and run the following from its root directory with Python 3.10 or 3.11. This CPU example needs no API key or GPU:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[train,test]'
python -m pip check
recrsi demo --output ../recrsi-demo
```

The example trains SASRec for two epochs on a small synthetic dataset, evaluates NDCG@5 and Hit@5, and reloads the checkpoint. Results are in `../recrsi-demo/DEMO_RESULT.json`; learning curves are in `../recrsi-demo/baseline/worker/LEARNING_CURVE.jsonl`. This demonstrates training and evaluation, not the LLM-driven search described below.

## Run a research task

Full research runs need a Linux x86-64 CUDA host and an LLM API endpoint. See [host setup](docs/host-setup.md) for CUDA installation, worker setup, and the required model IDs. Use a CUDA-enabled environment rather than the CPU-only PyTorch installation above.

### 1. Create a project

```bash
recrsi init --output ../my-research
```

Fill in `../my-research/environment.json` following the host setup guide. Then edit `../my-research/inputs/binding.json` for your dataset, model, and training settings.

### 2. Prepare data and literature

For a new sequential task, start with a TSV containing `user_id`, `item_id`, and integer `timestamp` columns:

```bash
recrsi split --interactions ../interactions.tsv --output ../my-data \
  --test-output ../heldout/cds.test.inter --dataset cds --max-length 50
```

This creates `cds.train.inter`, `cds.dev.inter`, `cds.user`, and `cds.item` in `../my-data/`, with TEST saved separately. The splitter uses each user's last two distinct items for DEV and TEST and skips users with fewer than four distinct items. Match `MAX_ITEM_LIST_LENGTH` in the binding to `--max-length`. Existing splits can also be supplied in the documented [task format](docs/task-format.md).

Create `../literature.json` with relevant papers and summaries, for example:

```json
{
  "documents": [
    {
      "title": "Self-Attentive Sequential Recommendation",
      "year": 2018,
      "summary": "Self-attention models sequential item dependencies."
    }
  ]
}
```

### 3. Start the search

Run these commands in the training environment:

```bash
recrsi doctor --environment ../my-research/environment.json
recrsi assets --project ../my-research --data-dir ../my-data \
  --literature ../literature.json --description "Sequential recommendation on CDs"
recrsi baseline --project ../my-research --output ../my-baseline --device cuda
recrsi check --config ../my-research/task.json
recrsi prepare --config ../my-research/task.json --output ../my-run
recrsi inspect ../my-run
recrsi run ../my-run
```

`assets` builds the item mapping and fixed evaluation candidates. `baseline` trains the starting model. The remaining commands check the setup, prepare a run directory, and start the hypothesis–experiment–feedback loop. A search allows up to 12 candidate evaluations by default; budget and repeat settings are described in [task format](docs/task-format.md).

To resume an interrupted run, use `recrsi run ../my-run --resume`. If `PAUSED.json` is present, check its message and the [troubleshooting guide](docs/host-setup.md#troubleshooting) first.

## Read the outputs

Results are saved under `../my-run/task/research_line/runs/<run_id>/`:

- `RESULT.json`: the final search result and retained candidates.
- Candidate directories: source code, learning curves, checkpoints, and validation scores.
- Research records: the mechanism space, hypotheses, experiment choices, feedback, and API/GPU usage.

NDCG@5 and Hit@5 are evaluated on fixed 100-item DEV candidate sets. Search uses TRAIN and DEV; keep TEST separate for final evaluation.

## Development

Run the tests with:

```bash
python -m pytest tests -q
```

Further documentation: [task format](docs/task-format.md), [host setup](docs/host-setup.md), and [anonymous source export](docs/anonymous-release.md).

Released under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for the included SASRec implementation.
