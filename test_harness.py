import os

os.environ['ALGPT_FLASH_ATTN'] = '1'
os.environ['ALGPT_FUSED_RMSNORM'] = '1'
os.environ['ALGPT_FUSED_CROSSENTROPY'] = '1'
os.environ['ALGPT_FUSED_ROTARY'] = '1'
os.environ['ALGPT_FUSED_SWIGLU'] = '1'

import models
import lm_eval

lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.

results = lm_eval.simple_evaluate( # call simple_evaluate
    model="hf",
    # model_args="pretrained=outputs/llama-tiny-original-test,dtype=bfloat16",
    model_args="pretrained=/tmp/test-clm-opt-8-7-l1,dtype=bfloat16",
    tasks=["mathqa"],
    num_fewshot=0,
    log_samples=False,
)

with open("result.json", "w") as f:
    import json
    json.dump(results, f, indent=4)
