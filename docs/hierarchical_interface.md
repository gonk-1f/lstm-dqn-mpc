# Hierarchical Interface

## Upper MPC outputs to lower DQN

Current project interface supports both total references and side-aware
references:

- `mpc_fuel_cell_ref_kw`
- `mpc_battery_ref_kw`
- `mpc_soc_ref`
- `mpc_fuel_cell_ref_left_kw`
- `mpc_fuel_cell_ref_right_kw`
- `mpc_battery_ref_left_kw`
- `mpc_battery_ref_right_kw`

The design intent is:

1. the upper layer computes economically reasonable total source references
2. the upper layer also provides side-aware allocations for left/right vessel
   devices
3. the lower DQN tracks and refines these references under faster time scales

## Lower DQN inputs

The dual-side lower environment currently exposes:

- left/right load
- left/right SOC
- upper-layer `SOC_ref`
- left/right actual fuel-cell power
- left/right fuel-cell references
- left/right battery references
- total load and total source powers
- left/right reference tracking errors

The simple lower environment currently exposes:

- total load
- vessel speed
- actual SOC
- upper-layer `SOC_ref`
- actual fuel-cell and battery power
- upper-layer fuel-cell and battery references
- tracking errors and load ramp

## Important modeling note

This interface is vessel-specific and should not be interpreted as the original
microgrid buy/sell formulation. The project has been redirected toward a
hydrogen vessel microgrid with battery clusters, fuel-cell groups, inverters,
DC/DC converters, and propulsion-related demand.
