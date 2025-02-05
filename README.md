# Minecraft Building Assistance Game

This repository contains code for the paper *Scalably Solving Assistance Games*. It includes the implementation of the Minecraft Building Assistance Game (MBAG), which we introduce to study more complex assistance games. It also includes code for running all experiments in the paper, including training assistants for MBAG with AssistanceZero.

MBAG is a multiagent environment that can run on its own within Python. It can also connect to running instances of Minecraft via [Project Malmo](https://github.com/microsoft/malmo) for visualization or for interaction with human players.

## Setup

This section describes how to set up your environment for running MBAG.

### Installing dependencies

First, install Python 3.8, 3.9, or 3.10. Then, run one of the following commmands:

  * Install just the environment: `pip install -e .`
  * Also install the RLlib dependencies for training and running assistants: `pip install -e .[rllib]`
  * Also install the Malmo interface to run assistants in Minecraft: `pip install -e .[rllib,malmo]`

    *Note: the interface with Minecraft via Malmo is currently only supported on macOS and Linux.*

To run assistants in Minecraft, you will also need Java 8 installed. On macOS, you specifically need to install the Java JDK 8u152, which is available [here](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html). See [https://github.com/microsoft/malmo/issues/907] for details about why the specific version for macOS is necessary.

### Linting, testing, and type checking

The project uses various tools to maintain code quality. To install them all, run

    pip install --upgrade -e .[dev]

Then, run linting and testing with the following commands:

    ./lint.sh
    pytest -m "not uses_malmo"

To run fewer tests, you can some or all of these additional filters:

    pytest -m "not uses_malmo and not slow"
    pytest -m "not uses_malmo and not uses_rllib"
    pytest -m "not uses_malmo and not uses_cuda"

## Running assistants in Minecraft

Playing with assistants in Minecraft and/or collecting data of humans playing takes two steps: first, starting Minecraft instances, and second, connecting to those instances to run an episode within the MBAG environment.

### Starting Minecraft

To start Minecraft instances, run the following command, assuming you've installed the Malmo dependencies (see [setup](#setup)):

    python -m malmo.minecraft launch --num_instances 2 --goal_visibility True False

Set the number of Minecraft instances to launch with the `--num_instances` option. You need at least two instances to play with an assistant (one for the human and one for the AI assistant); if you want to record video of the game, start an additional instance for a "spectator" player.

The `--goal_visibility` argument controls which instances show the goal house as a transparent blueprint within the game. Generally you should set this to True for the first instances and False for additional instances (e.g., `--goal_visibility True False False` for three instances).

Once you start 

### Running an MBAG episode

Once the Minecraft instances are running, you can use the following command to start an episode playing with an assistant:

    python -m mbag.scripts.evaluate with human_with_assistant assistant_checkpoint=data/assistancezero_assistant/checkpoint_002000 num_simulations=1

*Note: `num_simulations` is used to control how long MCTS runs. If you are on a machine with a GPU, you can probably set `num_simulations=20`. If you do not have a GPU, you may want to set `num_simulations=1`, which will sample directly from the policy network. If you notice errors like `WARNING - mbag.environment.mbag_env - environment step took longer than Malmo action_delay; action_delay may need to be increased to achieve consistent step rate`, then reduce `num_simulations`.*

To play an episode without an assistant, run:

    python -m mbag.scripts.evaluate with human_alone assistant_checkpoint=data/assistancezero_assistant/checkpoint_002000

The `assistant_checkpoint` argument is still needed in this case to load the environment configuration.

Once the episode starts, press <kbd>Return</kbd> (<kbd>Enter</kbd>) to enable movement and <kbd>Delete</kbd> (<kbd>Fn</kbd> + <kbd>Backspace</kbd> on Mac) to enable flying.

The episode will automatically terminate when the house is completed, but if you want to end it sooner, use <kbd>Ctrl</kbd>+<kbd>C</kbd>. At the end of the episode, the episode data will be saved and metrics will be printed out.
