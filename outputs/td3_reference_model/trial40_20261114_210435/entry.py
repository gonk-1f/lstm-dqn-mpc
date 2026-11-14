from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src" / "scripts"))
import train_td3_reference_model as experiment

experiment.EPISODES = 40
experiment.TOTAL_TIMESTEPS = experiment.EPISODES * experiment.TRAIN_WINDOW_STEPS
experiment.OUTPUT_ROOT = Path(__file__).resolve().parent
experiment.main()
