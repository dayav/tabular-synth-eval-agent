# Scope

- Goal : Given a synthetic tabular dataset generated from an original dataset, the goal is to create an agent capable of evaluating both the univariate resemblance and the utility of the synthetic data compared to the original data.

- Implementation : 
    - LLM : For the first MVP, we will use the Claude API due to the availability of Anthropic's tool-use capabilities.
    - LLM Integration: The first version will use the Anthropic SDK with native tool support.
- First MVP : Create a terminal-based application that outputs the evaluation results and generates a report.md file summarizing the findings.
    - Univariate Resemblance Evaluation :
        - Compute the Jensen–Shannon distance for all categorical features.
        - Compute Cohen's d for all numerical features.
    - Utility Evaluation : 
        - compare the performance of a model trained synthetic data and tested on real(TSTR) against the performance of a model trained on real data and tested on real data (trtr) 


## Out of scope (v1)
- Multivariate fidelity (correlation matrices, joint distributions)
- Privacy evaluation
- Non-tabular data
- Web UI / API endpoint
- Framework-based orchestration (LangGraph, etc.) — kept for v2