"""Independent RL Zoo vessel route; import the env module to register its Gym ID."""
from pathlib import Path
import sys

_source = str(Path(__file__).resolve().parents[3]/'src')
if _source not in sys.path:
    sys.path.insert(0, _source)

from .env import ENV_ID, register_envs

register_envs()

__all__ = ['ENV_ID', 'register_envs']
