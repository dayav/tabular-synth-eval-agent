"""Agent orchestration: dispatches user prompts to Claude with tool use."""
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import json
from report import save_report
from tools import js_divergence_categorical, cohen_d_numerical, tstr_xgboost

load_dotenv()
client = Anthropic() 

SYSTEM_PROMPT = """You are an expert in evaluating synthetic tabular data.

You have three tools available:
- js_divergence_categorical: fidelity of categorical columns
- cohen_d_numerical: fidelity of numerical columns
- tstr_xgboost: predictive utility (train-on-synthetic vs train-on-real)

When asked to evaluate a synthetic dataset, call all three tools in sequence,
then write a short plain-English assessment summarizing:
1. Categorical fidelity (JS distance — smaller is better, typically < 0.1 is good).
2. Numerical fidelity (Cohen's d — under 0.2 is small effect, indicates good fidelity).
3. Utility gap (positive means synthetic loses utility; under 0.05 is excellent).

Be honest. If the data is poor on any axis, say so.
Note that the target column should not be interpreted as a fidelity issue
when it appears in Cohen's d output — it's a label, not a feature."""

TOOLS = [
    {
        "name": "js_divergence_categorical",
        "description": (
            "Compute the Jensen-Shannon distance between the real and synthetic "
            "datasets for each categorical column. Returns a per-column dict, "
            "the mean across columns, and the column count. Smaller values mean "
            "the synthetic categorical distributions are closer to real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
            },
            "required": ["real_csv_path", "synth_csv_path"],
        },
    },
    {
        "name": "cohen_d_numerical",
        "description": (
            "Compute absolute Cohen's d effect size between the real and synthetic "
            "datasets for each numerical column. Returns a per-column dict, the mean "
            "across columns, and the column count. Cohen's d roughly interprets as: "
            "0.2 = small effect, 0.5 = medium, 0.8 = large. Smaller values mean the "
            "synthetic numerical distributions are closer to real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
            },
            "required": ["real_csv_path", "synth_csv_path"],
        },
    },
    {
        "name": "tstr_xgboost",
        "description": (
            "Compute a utility evaluation of the synthetic dataset using XGBoost. "
            "Trains one XGBoost classifier on the synthetic data and one on the real "
            "data, then evaluates both on a held-out portion of the real data. "
            "Returns: tstr_accuracy (model trained on synthetic), trtr_accuracy "
            "(model trained on real, as baseline), utility_gap (= trtr_accuracy - "
            "tstr_accuracy; positive means synthetic loses utility, near zero means "
            "synthetic is as useful as real), target_column, n_test_samples, n_classes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the column to predict (classification target).",
                },
                "test_csv_path": {
                    "type": "string",
                    "description": (
                        "Optional path to a held-out test CSV. "
                        "If omitted, the real dataset is split 80/20 internally."
                    ),
                },
            },
            "required": ["real_csv_path", "synth_csv_path", "target_column"],
        },
    },    
]

# in src/agent.py


TOOL_REGISTRY = {
    "js_divergence_categorical": js_divergence_categorical,
    "cohen_d_numerical": cohen_d_numerical,
    "tstr_xgboost": tstr_xgboost,
}

def run_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call by name. Returns the tool's result dict."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return TOOL_REGISTRY[tool_name](**tool_input)
    except Exception as e:
        return {"error": str(e)}


def run_agent(user_prompt: str, max_turns: int = 10) -> str:
    """Run the agent loop until Claude stops calling tools."""
    messages = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        # Append Claude's response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        # If Claude is done calling tools, return the text
        if response.stop_reason != "tool_use":
            # Pull text out of the final response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "(no text response)"

        # Otherwise, run each tool and append the results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return "(max turns reached)"

if __name__ == "__main__":

    user_prompt = (
        "Evaluate the synthetic dataset at examples/synth_sample.csv "
        "against the real dataset at examples/real_sample.csv. "
        "The target column for utility evaluation is 'target'."
    )
    final_response = run_agent(user_prompt)
    print(final_response)

    path = save_report(final_response)
    print(f"\nReport saved to: {path}")