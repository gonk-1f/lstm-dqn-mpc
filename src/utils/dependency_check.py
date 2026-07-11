from __future__ import annotations

import importlib
import sys


REQUIRED_RUNTIME_PACKAGES = ["numpy", "pandas", "matplotlib"]
REQUIRED_DQN_PACKAGES = ["torch"]
OPTIONAL_MPC_PACKAGES = ["casadi", "scipy"]
OPTIONAL_CONFIG_PACKAGES = ["yaml"]


def check_packages(package_names: list[str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for package in package_names:
        try:
            importlib.import_module(package)
            status[package] = True
        except ModuleNotFoundError:
            status[package] = False
    return status


def build_dependency_report() -> dict[str, dict[str, bool]]:
    config_status = check_packages(OPTIONAL_CONFIG_PACKAGES)
    config_status["project_yaml_fallback"] = True
    report = {
        "runtime": check_packages(REQUIRED_RUNTIME_PACKAGES),
        "dqn": check_packages(REQUIRED_DQN_PACKAGES),
        "mpc_optional": check_packages(OPTIONAL_MPC_PACKAGES),
        "config_optional": config_status,
        "python_executable": sys.executable,
    }
    if report["dqn"].get("torch", False):
        try:
            import torch

            report["dqn_details"] = {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            report["dqn_details"] = {"error": str(exc)}
    return report
