from pathlib import Path
import json
import argparse
import models
import lm_eval

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tasks", type=str, nargs="+", default=["hellaswag", "openbookqa", "winogrande", "arc_challenge", "arc_easy", "boolq", "piqa"])
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output", type=str, default="harness/result.json")

    args = parser.parse_args()

    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model="hf",
        model_args=f"pretrained={args.model_name_or_path},dtype={args.dtype},attn_implementation=flash_attention_2",
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        log_samples=False,
        task_manager=task_manager,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, default=lambda o: '<not serializable>')

if __name__ == "__main__":
    main()
