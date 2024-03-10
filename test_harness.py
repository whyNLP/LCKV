import os

os.environ['LCKV_FLASH_ATTN'] = '1'
os.environ['LCKV_FUSED_RMSNORM'] = '1'
os.environ['LCKV_FUSED_CROSSENTROPY'] = '1'
os.environ['LCKV_FUSED_ROTARY'] = '1'
os.environ['LCKV_FUSED_SWIGLU'] = '1'

import models
import lm_eval

lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.

results = lm_eval.simple_evaluate( # call simple_evaluate
    model="hf",
    model_args="pretrained=TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T,dtype=bfloat16",
    tasks=[
        "hellaswag",
        "openbookqa",
        "winogrande",
        "arc_challenge",
        "arc_easy",
        "boolq",
        "piqa",
    ],
    num_fewshot=0,
    log_samples=False,
)

with open("harness/result-ref-2.5T.json", "w") as f:
    import json
    json.dump(results, f, indent=4)
