"""Record actual trial model settings without changing Zoo's training flow."""
import json
from pathlib import Path

from rl_zoo3.callbacks import TrialEvalCallback
from stable_baselines3.common.callbacks import BaseCallback

from .economic_reward import REWARD_VERSION


def ensure_study_reward_identity(study, trial_number, identity):
    """Do not reuse trials from an unknown or different economic reward."""
    if identity.get("reward_version") != REWARD_VERSION:
        raise ValueError("Optuna environment has the wrong economic reward version")
    recorded = study.user_attrs.get("ddpg_reward_identity")
    if recorded is None:
        if any(trial.number != trial_number for trial in study.get_trials(deepcopy=False)):
            raise ValueError("Existing Study has trials without reward identity; start a new economic Study")
        study.set_user_attr("ddpg_reward_identity", identity)
    elif recorded != identity:
        raise ValueError("Study reward/calibration identity differs; do not mix reward definitions")


class TrialParametersCallback(BaseCallback):
    def _on_training_start(self):
        # OffPolicyAlgorithm.learn supplies its initialized CallbackList in locals.
        evaluation = next(
            cb for cb in self.locals["callback"].callbacks
            if isinstance(cb, TrialEvalCallback)
        )
        model = self.model
        identities = self.training_env.get_attr("reward_identity")
        identity = identities[0]
        if any(item != identity for item in identities + evaluation.eval_env.get_attr("reward_identity")):
            raise ValueError("Optuna Train and Validation reward identities differ")
        study = getattr(evaluation.trial, "study", None)
        if study is not None:
            ensure_study_reward_identity(study, evaluation.trial.number, identity)
        model._ddpg_reward_identity = dict(identity)
        evaluation.trial.set_user_attr("reward_identity", identity)
        self.logger.record("reward/version", identity["reward_version"], exclude=("tensorboard",))
        self.logger.record("reward/c_ref", identity["c_ref"])
        self.logger.record("reward/fc_off_reference_efficiency", identity["fc_off_reference_efficiency"])
        parameters = {
            "learning_rate": float(model.lr_schedule(1.0)),
            "gamma": model.gamma,
            "batch_size": model.batch_size,
            "buffer_size": model.buffer_size,
            "learning_starts": model.learning_starts,
            "train_freq": model.train_freq.frequency,
            "train_freq_unit": model.train_freq.unit.value,
            "gradient_steps": model.gradient_steps,
            "critic_loss": identity["critic_loss"],
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
            "reward_identity": identity,
        }
        evaluation.trial.set_user_attr("effective_parameters", parameters)
        if evaluation.best_model_save_path is not None:
            path = Path(evaluation.best_model_save_path) / "effective_parameters.json"
            path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
        print(f"Trial {evaluation.trial.number} effective DDPG parameters: "
              + json.dumps(parameters, sort_keys=True))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for name in ("equivalent_fc_efficiency", "fc_efficiency_actual", "c_bat_eq_cny", "battery_current_a"):
                if name in info:
                    self.logger.record(f"economics/{name}", info[name])
        return True
