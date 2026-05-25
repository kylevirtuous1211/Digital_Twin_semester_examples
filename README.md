# Digital Twin Semester Examples

Hands-on examples developed during the Digital Twin / Isaac Sim semester course. Designed to run against the [isaac-sim-quickstart](https://github.com/kylevirtuous1211/isaac-sim-quickstart) Docker environment.

## Usage

Clone this repo alongside `isaac-sim-quickstart`, then send a script to the running Isaac Sim container:

```bash
# From the isaac-sim-quickstart directory, with the container running:
./run_in_isaac.py /path/to/Digital_Twin_semester_examples/hand_on_1_amr.py --wait
```

Or mount this directory into the container by adding a bind mount to `docker-compose.yml`.

## Examples

| Script | Description |
|--------|-------------|
| `hand_on_1_amr.py` | JetBot navigates 4 colored waypoints in a square |
| `hand_on_2_franka.py` | Franka Panda picks 3 random cubes, stacks a pyramid |
| `hand_on_3_rmpflow.py` | RMPflow obstacle-avoidance demo |
| `hand_on_4_cortex.py` | Cortex stacking pipeline |
| `hand_on_5_domain_randomization.py` | Replicator scatters 6 YCB props around Franka and captures RGB + semantic + instance segmentation to `./output/` |
| `hands_on_6_visualization/` | Plotting and tracing helpers |
| `cosmos_writer_pick_place.py` | CosmosWriter pick-and-place data capture |
