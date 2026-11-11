"""Start a fresh TD3 v7 run from PyCharm or PowerShell."""

from run_td3_train_economic_ems import main


if __name__ == "__main__":
    raise SystemExit(main(
        env_id="AggregateShipTD3EconomicV7-v0",
        config_name="zoo_td3_economic_ems_v7.yml",
        output_name="td3_v7_fixed_parent005_n32_prevfc_500ep",
    ))
