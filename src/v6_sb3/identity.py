"""Small explicit identity records for reproducible, Train/Val-only resume."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(dataset):
    files = ['metadata/active_split.json', dataset._metadata['manifest_relative_path']]
    rows = dataset._manifest.loc[dataset._manifest['split'].isin(('train','validation'))]
    files += sorted(rows['relative_path'].tolist())
    return dict(contract=dataset.task_contract,
                train_validation_files={relative:file_digest(dataset.root/relative) for relative in files},
                test_payloads_opened=0)


def execution_identity(root, data, *, reward_mode=None):
    from v4.independent_economics import ZhouEvaluationConfig
    from v4.failure_replay import FailurePenalty
    root = Path(root)
    sources = sorted((root/'src').rglob('*.py'))
    identity = dict(schema=1, data=data, python_version=platform.python_version(),
        source_files={path.relative_to(root).as_posix():file_digest(path) for path in sources},
        dependencies={name:version(name) for name in (
            'numpy','pandas','scipy','torch','stable-baselines3','gymnasium','optuna')},
        economics=ZhouEvaluationConfig.from_formal_curve().identity())
    if reward_mode is not None:
        from .rewards import INVERSE_ONE_PLUS_ECONOMIC, reward_contract
        identity['reward_contract'] = reward_contract(reward_mode)
        if reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
            identity['failure_penalty'] = None
            return identity
    identity['failure_penalty'] = asdict(FailurePenalty.from_reference())
    return identity
