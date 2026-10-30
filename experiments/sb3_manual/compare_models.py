
from pathlib import Path
import statistics

import gymnasium as gym
from stable_baselines3 import DQN

# 已经训练好的模型所在目录
MODEL_DIR = Path(
    "experiments/sb3_manual/zoo_smoke/dqn/CartPole-v1_1"
)

# 使用相同的20个随机种子评估两个模型
TEST_SEEDS = range(1000, 1020)


def evaluate(model_path):
    model = DQN.load(str(model_path), device="cpu")
    env = gym.make("CartPole-v1")
    rewards = []

    try:
        for seed in TEST_SEEDS:
            obs, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs, deterministic=True
                )

                obs, reward, terminated, truncated, info = (
                    env.step(int(action))
                )
                total_reward += reward

            rewards.append(total_reward)
    finally:
        env.close()

    return rewards


best = evaluate(MODEL_DIR / "best_model.zip")
final = evaluate(MODEL_DIR / "CartPole-v1.zip")

print("\n===== 20 Episode 统一评估 =====")

for name, rewards in [
    ("Best", best),
    ("Final", final),
]:
    print(f"\n{name} Model")
    print(f"平均奖励: {statistics.mean(rewards):.2f}")
    print(f"标准差: {statistics.pstdev(rewards):.2f}")
    print(f"最小奖励: {min(rewards):.0f}")
    print(f"最大奖励: {max(rewards):.0f}")

print("\n===== 同随机种子对比 =====")
print(f"Best 胜出: {sum(a > b for a, b in zip(best, final))}")
print(f"Final 胜出: {sum(a < b for a, b in zip(best, final))}")
print(f"两者相同: {sum(a == b for a, b in zip(best, final))}")
