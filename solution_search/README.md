# Solution Search

This package implements a repo-local autonomous search loop for Parameter Golf style experimentation:

1. propose a candidate via an LLM API
2. apply code edits and/or environment overrides in an isolated workspace
3. run an evaluator
4. score the result against hard constraints
5. reflect on the run
6. update strategy and mistake archives so the loop learns over time

It is intentionally smaller and more specialized than the systems that inspired it:

- `karpathy/autoresearch`
- `sakanaai/ai-scientist-v2`
- `algorithmicsuperintelligence/openevolve`

The goal here is not a full general-purpose research platform. It is a practical search engine for this repository and this class of optimization problem.

## Design

The search state is persisted under a campaign directory such as:

```text
solution_search_runs/parameter_golf_proxy/
├── archive/
│   ├── meta.json
│   ├── mistakes.json
│   ├── runs.jsonl
│   └── strategies.json
└── candidates/
    └── candidate_0001/
        ├── combined.log
        ├── proposal.json
        ├── result.json
        └── snapshot/
```

Each candidate stores the resulting editable-file snapshot, so future mutations can branch from the best prior candidate without preserving an entire workspace tree.

The branching loop is lightweight MCTS-style rather than full textbook MCTS:

- a UCB-like score selects a small parent pool from prior feasible candidates
- the proposer emits a batch of sibling candidates each iteration
- each sibling is executed, reflected on, and added back to the archive
- later iterations can branch from any successful child

This keeps the search tree simple enough for code experiments while still exploring multiple local moves per round.

## LLM Interface

The live provider uses an OpenAI-compatible `POST /chat/completions` API. Configure it in the campaign JSON and provide the key via the configured environment variable.

For local verification, the package also ships with a `stub` provider that returns deterministic no-op proposals and reflections.

## Edit Protocol

The proposer returns structured JSON rather than raw diffs. Supported edit types:

- `replace`
- `insert_after`
- `insert_before`
- `append`
- `write_file`
- `regex_replace`
- `insert_after_regex`
- `insert_before_regex`

This is less flexible than unrestricted patch generation, but it is much easier to validate and replay.

## Knowledge Base

The search loop is not limited to its own run history. Each campaign can maintain a cached external knowledge base under `solution_search_runs/<campaign>/knowledge/cards.json`.

Supported source kinds:

- `url`: fetch any public URL and strip HTML if needed
- `github_readme`: fetch a repo README from GitHub
- `github_markdown_linked_readmes`: fetch a GitHub markdown page, extract linked markdown files, and ingest the top matching READMEs
- `arxiv_search`: fetch public arXiv search results and treat the returned abstracts as paper notes

The provided Parameter Golf config uses this to ingest:

- the public Parameter Golf README / leaderboard
- leaderboard-linked top record READMEs from `records/track_10min_16mb/`
- external research repos such as `autoresearch`, `AI-Scientist-v2`, and `OpenEvolve`
- BitNet / ternary QAT paper search results from arXiv

## Usage

Smoke-test the loop without a live model:

```bash
python3 -m solution_search run \
  --config solution_search/examples/parameter_golf_syntax_smoke.json \
  --iterations 2
```

Run the real proxy search loop against the current ternary experiment:

```bash
export LLM_API_KEY=...
python3 -m solution_search run \
  --config solution_search/examples/parameter_golf_proxy.json \
  --iterations 5
```

Inspect the current best feasible candidates:

```bash
python3 -m solution_search leaderboard \
  --config solution_search/examples/parameter_golf_proxy.json \
  --limit 10
```

Refresh or inspect the external knowledge cache:

```bash
python3 -m solution_search sync-knowledge \
  --config solution_search/examples/parameter_golf_proxy.json \
  --force

python3 -m solution_search show-knowledge \
  --config solution_search/examples/parameter_golf_proxy.json \
  --limit 6
```

## Parameter Golf Notes

The provided example campaign is targeted at:

- `records/track_non_record_16mb/2026-03-23_Ternary_QAT_vs_Int5/train_gpt.py`
- the FineWeb-based proxy workflow already developed in this repo
- the current mixed-ternary branch

The seeded strategy and mistake archives already encode local findings such as:

- full ternary currently loses too much quality
- larger mixed ternary is the most promising branch
- naive layer recurrence has already looked bad in a prior non-record run
- the last mixed MLP projection can collapse to all-zero

That gives the loop a useful starting memory instead of forcing it to relearn every local failure from scratch.
