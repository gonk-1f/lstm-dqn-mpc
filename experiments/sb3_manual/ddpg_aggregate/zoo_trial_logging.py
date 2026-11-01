"""Record actual trial model settings without changing Zoo's training flow."""
import json
from pathlib import Path

from rl_zoo3.callbacks import TrialEvalCallback
from stable_baselines3.common.callbacks import BaseCallback


class TrialParametersCallback(BaseCallback):
    def _on_training_start(self):
        # OffPolicyAlgorithm.learn supplies its initialized CallbackList in locals.
        evaluation = next(
            cb for cb in self.locals["callback"].callbacks
            if isinstance(cb, TrialEvalCallback)
        )
        model = self.model
        parameters = {
            "learning_rate": float(model.lr_schedule(1.0)),
            "gamma": model.gamma,
            "batch_size": model.batch_size,
            "buffer_size": model.buffer_size,
            "learning_starts": model.learning_starts,
            "train_freq": model.train_freq.frequency,
            "train_freq_unit": model.train_freq.unit.value,
            "gradient_steps": model.gradient_steps,
            "tau": model.tau,
            "policy": type(model.policy).__name__,
            "policy_kwargs": model.policy_kwargs,
            "actor": str(model.actor.mu),
            "critic": str(model.critic.qf0),
            "optimizer": type(model.actor.optimizer).__name__,
            "noise_type": type(model.action_noise).__name__,
            "noise_std": model.action_noise._sigma.tolist(),
            "n_envs": model.n_envs,
            "device": str(model.device),
            "model_seed": model.seed,
            "training_split": self.training_env.get_attr("split"),
            "evaluation_split": evaluation.eval_env.get_attr("split"),
            "eval_episodes": evaluation.n_eval_episodes,
            "eval_freq": evaluation.eval_freq,
            "deterministic_eval": evaluation.deterministic,
        }
        evaluation.trial.set_user_attr("effective_parameters", parameters)
        if evaluation.best_model_save_path is not None:
            path = Path(evaluation.best_model_save_path) / "effective_parameters.json"
            path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
        print(f"Trial {evaluation.trial.number} effective DDPG parameters: "
              + json.dumps(parameters, sort_keys=True))

    def _on_step(self):
        return True
