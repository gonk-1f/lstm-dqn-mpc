from gymnasium.envs.registration import register

register(
    id="AggregateShipDDPG-v0",
    entry_point="experiments.sb3_manual.ddpg_aggregate.env:AggregateDDPGEnv",
)
