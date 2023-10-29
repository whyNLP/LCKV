from transformers.integrations import WandbCallback as OriginWandbCallback
from transformers.integrations import INTEGRATION_TO_CALLBACK, rewrite_logs

class WandbCallback(OriginWandbCallback):
    """Read _custom_log from model and add to the log."""
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            custom_log = getattr(model, "_custom_log", None)
            if isinstance(custom_log, dict):
                logs = {**logs, **custom_log}
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

# register to transformer callback
INTEGRATION_TO_CALLBACK["wandb"] = WandbCallback
