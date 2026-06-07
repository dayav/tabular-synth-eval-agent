# Synthetic Data Evaluation Agent

An LLM agent that evaluates the quality of synthetic tabular data against a real reference dataset. Given two CSV files, the agent autonomously selects and runs fidelity and utility evaluations, then writes a plain-English assessment and saves it as a Markdown report.

It is built on the evaluation methodology from my master's thesis on privacy-preserving synthetic tabular data ([paper](https://arxiv.org/abs/2602.06390) · [thesis pipeline](https://github.com/dayav/pipeline_synthetic_tabular_data)) and is powered by Claude through the Anthropic SDK's native tool use.

> **Status:** Work in progress (MVP).

## How it works

You give the agent the path to a real dataset and the path to a synthetic version of it. The agent then calls three evaluation tools and synthesizes the results into a readable summary. It currently covers two dimensions of quality:

### Univariate fidelity — does each column *look* like the real thing?

- **Categorical columns — Jensen–Shannon distance.** Measures how closely each synthetic categorical column's distribution matches the real one. `0` means identical, larger values mean more divergence (typically `< 0.1` is good).
- **Numerical columns — Cohen's *d*.** A standardized effect size for the difference in means between the real and synthetic values of each numerical column. Smaller is better (`< 0.2` is a small effect, indicating good fidelity).

### Utility — is the synthetic data *useful* for a downstream model?

- **Train on Synthetic, Test on Real (TSTR).** An XGBoost classifier is trained on the synthetic data and evaluated on held-out real data. Its accuracy is compared against a **Train on Real, Test on Real (TRTR)** baseline. The resulting **utility gap** (`TRTR − TSTR`) quantifies how much predictive value, if any, is lost by training on synthetic data instead of real data.

## Installation

```bash
pip install -r requirements.txt
```

The agent calls the Anthropic API, so it needs an API key. Copy the example file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env` and set `ANTHROPIC_API_KEY` to your [Anthropic API key](https://console.anthropic.com/).

## Usage

The simplest way to run an end-to-end evaluation on the bundled sample data:

```bash
python src/agent.py
```

This evaluates `examples/synth_sample.csv` against `examples/real_sample.csv`, prints the assessment, and writes it to `report.md`.

To run it on your own data, use the two-function API directly:

```python
from agent import run_agent
from report import save_report

prompt = (
    "Evaluate the synthetic dataset at path/to/synth.csv "
    "against the real dataset at path/to/real.csv. "
    "The target column for utility evaluation is 'target'."
)

# run_agent returns the agent's plain-English assessment as a string
assessment = run_agent(prompt)
print(assessment)

# save_report writes that assessment to a timestamped Markdown file (report.md by default)
path = save_report(assessment)
print(f"Report saved to: {path}")
```

## Example output

Run on the bundled sample data, the agent produces something like this:

> ### Categorical Fidelity (Jensen-Shannon Distance)
>
> | Column | JS Distance | Rating |
> |---|---|---|
> | `gender` | 0.174 | ⚠️ Moderate |
> | `education` | 0.000 | ✅ Perfect |
> | `city` | 0.044 | ✅ Good |
> | **Mean** | **0.073** | ✅ Acceptable |
>
> The overall mean JS distance of 0.073 is just under the 0.1 threshold... `gender` stands out with a score of 0.174 and warrants attention. `education` is perfectly replicated.
>
> *[...continues with numerical fidelity and utility evaluation. Full report at [report.md](report.md).]*

## Project structure

```
src/
  agent.py     # Agent loop: sends the prompt to Claude and dispatches tool calls
  tools.py     # The three evaluation tools (JS distance, Cohen's d, TSTR/XGBoost)
  report.py    # Writes the agent's assessment to a Markdown report
examples/      # Sample real and synthetic CSVs
notes/         # Project scope and design notes
```

## Roadmap


Likely v2: 
- Add a privacy evaluation tool (HEOM-based identifiability epsilon, from the thesis pipeline). 

Out of scope for v1 (planned for later):

- Multivariate fidelity (correlation matrices, joint distributions)
- Framework-based orchestration (e.g. LangGraph)
- A web UI / API endpoint

## License

See [LICENSE](LICENSE).
