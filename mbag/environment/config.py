from typing import List, Optional, Type, TypedDict, Union

from .goals import GoalGenerator, GoalGeneratorConfig, TransformedGoalGenerator
from .types import WorldSize


class MalmoConfigDict(TypedDict, total=False):
    use_malmo: bool
    """
    Whether to connect to a real Minecraft instance with Project Malmo.
    """

    use_spectator: bool
    """
    Adds in a spectator player to observe the game from a 3rd person point of view.
    """

    restrict_players: bool
    """
    Places a group of barrier blocks around players that prevents them from leaving
    the test world
    """

    video_dir: Optional[str]
    """
    Optional directory to record video from the game into.
    """

    ssh_args: Optional[List[Optional[List[str]]]]
    """
    If one of the Malmo instances is running over an SSH tunnel, then the entry in this
    list of the corresponding player should be set to a list of arguments that are
    passed to ssh in order to access the remote machine. This will be used to
    automatically set up necessary port forwarding.
    """

    start_port: int
    """
    Port to start looking for Malmo instances at (default 10000).
    """

    action_delay: float
    """
    The number of seconds to wait after each step to allow actions to complete
    in Malmo.
    """


class RewardsConfigDict(TypedDict, total=False):
    noop: float
    """
    The reward for doing any action which does nothing. This is usually either zero,
    or negative to discourage noops.
    """

    action: float
    """
    The reward for doing any action which is not a noop. This could be negative to
    introduce some cost for acting.
    """

    place_wrong: float
    """
    The reward for placing a block which is not correct, but in a place where a block
    should go. The negative of this is also given for breaking a block which is not
    correct.
    """

    own_reward_prop: float
    """
    A number from 0 to 1. At 0, it gives the normal reward function which takes into
    account all players actions. At 1, it gives only reward for actions that the
    specific player took.
    """

    own_reward_prop_horizon: Optional[int]
    """
    Decay own_reward_prop to 0 over this horizon. This requires calling
    set_global_timestep on the environment to update the global timestep.
    """

    get_resources: float
    """
    The reward for getting a resource block from the palette that the player
    did not have in their inventory previously.
    """


class AbilitiesConfigDict(TypedDict):
    teleportation: bool
    """
    Whether the agent can teleport or must move block by block.
    """

    flying: bool
    """
    Whether the agent can fly or if the agent must be standing on a block at all times.
    Not implemented yet!
    """

    inf_blocks: bool
    """
    True - agent has infinite blocks to build with
    False - agent must manage resources and inventory
    """


class EnchantmentDict(TypedDict, total=False):
    id: int
    """
    String id of Enchantment
    """

    level: int
    """
    The level of the enchantment to give to the item
    """


class ItemDict(TypedDict):
    id: str
    """
    String id of a Minecraft item.
    """

    count: int
    """
    The number of this item to place in the player inventory
    """

    enchantments: List[EnchantmentDict]


class MbagPlayerConfigDict(TypedDict, total=False):
    player_name: Optional[str]
    """A player name that will be displayed in Minecraft if connected via Malmo."""

    goal_visible: bool
    """Whether the player can observe the goal."""

    is_human: bool
    """
    Whether this player is a human interacting via Malmo. Setting this to True requires
    Malmo to be configured.
    """

    timestep_skip: int
    """
    How often the player can interact with the environment, i.e. 1 means every
    timestep, 5 means only every 5th timestep.
    """

    rewards: RewardsConfigDict
    """
    Optional per-player reward configuration. Any unpopulated keys are overridden by
    values from the overall rewards config dict.
    """

    give_items: List[ItemDict]
    """
    A list of items to give to the player at the beginning of the game.
    """


class MbagConfigDict(TypedDict, total=False):
    num_players: int
    horizon: int
    world_size: WorldSize
    random_start_locations: bool

    goal_generator: Union[Type[GoalGenerator], str]
    goal_generator_config: GoalGeneratorConfig

    players: List[MbagPlayerConfigDict]
    """List of player configuration dictionaries."""

    malmo: MalmoConfigDict
    """Configuration options for connecting to Minecraft with Project Malmo."""

    rewards: RewardsConfigDict
    """
    Configuration options for environment reward. To configure on a per-player basis,
    use rewards key in player configuration dictionary.
    """

    abilities: AbilitiesConfigDict
    """
    Configuration for limits placed on the players (e.g., can they teleport, do they
    have to gather resources, etc.).
    """

    randomize_first_episode_length: bool
    """
    If True, the first episode will have a random length between 1 and horizon. This
    can be useful when training with an algorithm like PPO so that fragments
    of episodes are not strongly correlated across environments.
    """


DEFAULT_PLAYER_CONFIG: MbagPlayerConfigDict = {
    "player_name": None,
    "goal_visible": True,
    "is_human": False,
    "timestep_skip": 1,
    "rewards": {},
    "give_items": [],
}


DEFAULT_CONFIG: MbagConfigDict = {
    "num_players": 1,
    "horizon": 50,
    "world_size": (5, 5, 5),
    "random_start_locations": False,
    "goal_generator": TransformedGoalGenerator,
    "goal_generator_config": {
        "goal_generator": "random",
        "goal_generator_config": {},
        "transforms": [
            {"transform": "add_grass"},
        ],
    },
    "players": [{}],
    "malmo": {
        "use_malmo": False,
        "use_spectator": False,
        "restrict_players": False,
        "video_dir": None,
        "ssh_args": None,
        "start_port": 10000,
        "action_delay": 0.3,
    },
    "rewards": {
        "noop": 0.0,
        "action": 0.0,
        "place_wrong": 0.0,
        "own_reward_prop": 0.0,
        "own_reward_prop_horizon": None,
        "get_resources": 0,
    },
    "abilities": {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    },
}
