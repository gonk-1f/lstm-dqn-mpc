"""Start a fresh TD3 V8 run from PyCharm or PowerShell."""

from run_td3_train_economic_ems import main


if __name__ == "__main__":
    raise SystemExit(main(
        env_id="AggregateShipTD3EconomicV8-v0",
        config_name="zoo_td3_economic_ems_v8.yml",
        output_name="td3_v8_fixed_parent005_n32_prevfc_soc_potential_500ep",
    ))
