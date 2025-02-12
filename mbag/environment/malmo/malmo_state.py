from typing import (
    ItemsView,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from ..blocks import MinecraftBlocks
from ..config import MbagConfigDict
from ..types import (
    BlockLocation,
    MbagInventory,
    WorldLocation,
    get_block_counts_in_inventory,
)
from .malmo_client import INVENTORY_SLOT_NAMES, MalmoObservationDict


class BlockDiff(NamedTuple):
    location: BlockLocation
    prev_block: int
    new_block: int


class InventoryDiff(NamedTuple):
    player_index: int
    block_id: int
    prev_count: int
    new_count: int


class LocationDiff(NamedTuple):
    player_index: int
    prev_location: WorldLocation
    new_location: WorldLocation


MalmoStateDiff = Union[BlockDiff, InventoryDiff, LocationDiff]


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class ReadOnlyList(Sequence[T]):
    def __init__(self, initial_items: Sequence[T]) -> None:
        self._items = tuple(initial_items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items})"


class ReadOnlyDict(Mapping[K, V]):
    def __init__(self, initial_items: Mapping[K, V]) -> None:
        self._items = dict(initial_items)

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def items(self) -> ItemsView[K, V]:
        return self._items.items()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items})"


class MalmoState(object):
    """
    Keeps track of important state of the Malmo environment (i.e., what's going
    on in Minecraft).
    """

    _blocks: MinecraftBlocks
    """The current blocks in the world."""

    _player_inventories: ReadOnlyList[MbagInventory]
    """The inventories of the players."""

    _player_locations: ReadOnlyList[WorldLocation]
    """The locations of the players."""

    _player_blocks_looking_at: ReadOnlyList[Optional[BlockLocation]]
    """The blocks the players are looking at, if any."""

    _player_is_breaking: ReadOnlyList[bool]
    """Whether the players are currently breaking blocks (i.e., is left click down)."""

    _player_is_placing: ReadOnlyList[bool]
    """Whether the players are currently placing blocks (i.e., is right click down)."""

    _player_last_breaking: np.ndarray
    """
    For each block location, the player index of which human was last holding
    break on it. This is used to determine who broke a block when it disappears.
    """

    _player_last_placing: np.ndarray
    """
    For each block location, the player index of which human was last holding
    place on it.
    """

    def __init__(
        self,
        blocks: MinecraftBlocks,
        player_inventories: Sequence[MbagInventory],
        player_locations: Sequence[WorldLocation],
        player_blocks_looking_at: Sequence[Optional[BlockLocation]],
        player_is_breaking: Sequence[bool],
        player_is_placing: Sequence[bool],
        player_last_breaking: np.ndarray,
        player_last_placing: np.ndarray,
    ):
        # Make all attributes read-only.
        self._blocks = blocks
        self._blocks.make_immutable()
        self._player_inventories = ReadOnlyList(player_inventories)
        for inventory in player_inventories:
            inventory.setflags(write=False)
        self._player_locations = ReadOnlyList(player_locations)
        self._player_blocks_looking_at = ReadOnlyList(player_blocks_looking_at)
        self._player_is_breaking = ReadOnlyList(player_is_breaking)
        self._player_is_placing = ReadOnlyList(player_is_placing)
        self._player_last_breaking = player_last_breaking
        self._player_last_breaking.setflags(write=False)
        self._player_last_placing = player_last_placing
        self._player_last_placing.setflags(write=False)

    @property
    def blocks(self) -> MinecraftBlocks:
        return self._blocks

    @property
    def player_inventories(self) -> Sequence[MbagInventory]:
        return self._player_inventories

    @property
    def player_locations(self) -> Sequence[WorldLocation]:
        return self._player_locations

    @property
    def player_blocks_looking_at(self) -> Sequence[Optional[BlockLocation]]:
        return self._player_blocks_looking_at

    @property
    def player_is_breaking(self) -> Sequence[bool]:
        return self._player_is_breaking

    @property
    def player_is_placing(self) -> Sequence[bool]:
        return self._player_is_placing

    @property
    def player_last_breaking(self) -> np.ndarray:
        return self._player_last_breaking

    @property
    def player_last_placing(self) -> np.ndarray:
        return self._player_last_placing


def get_initial_malmo_state(
    initial_blocks: MinecraftBlocks,
    initial_player_locations: List[WorldLocation],
    env_config: MbagConfigDict,
) -> MalmoState:
    num_players = env_config["num_players"]
    world_size = env_config["world_size"]

    player_inventories = [
        np.zeros((len(INVENTORY_SLOT_NAMES), 2), dtype=int) for _ in range(num_players)
    ]

    # Set initial inventory if the user has infinite blocks
    if env_config["abilities"]["inf_blocks"]:
        for player_index in range(env_config["num_players"]):
            for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
                player_inventories[player_index][block_id][0] = block_id
                player_inventories[player_index][block_id][1] = 1

    player_blocks_looking_at = [None for _ in range(num_players)]
    player_is_breaking = [False for _ in range(num_players)]
    player_is_placing = [False for _ in range(num_players)]
    player_last_breaking = np.full(world_size, -1, dtype=np.int8)
    player_last_placing = np.full(world_size, -1, dtype=np.int8)
    return MalmoState(
        initial_blocks.copy(),
        player_inventories,
        initial_player_locations,
        player_blocks_looking_at,
        player_is_breaking,
        player_is_placing,
        player_last_breaking,
        player_last_placing,
    )


def _update_malmo_blocks(
    previous_state: MalmoState,
    malmo_observation: MalmoObservationDict,
    env_config: MbagConfigDict,
) -> Tuple[MinecraftBlocks, List[MalmoStateDiff]]:
    state_diffs: List[MalmoStateDiff] = []
    prev_blocks = previous_state.blocks
    if "world" in malmo_observation:
        new_blocks = MinecraftBlocks.from_malmo_grid(
            env_config["world_size"], malmo_observation["world"]
        )
        for location in cast(
            Sequence[BlockLocation],
            map(
                tuple,
                np.argwhere(new_blocks.blocks != prev_blocks.blocks),
            ),
        ):
            state_diffs.append(
                BlockDiff(
                    location,
                    prev_blocks.blocks[location],
                    new_blocks.blocks[location],
                )
            )
    else:
        new_blocks = prev_blocks

    return new_blocks, state_diffs


def _update_malmo_player_locations(
    previous_state: MalmoState,
    player_index: int,
    malmo_observation: MalmoObservationDict,
) -> Tuple[List[WorldLocation], List[MalmoStateDiff]]:
    state_diffs: List[MalmoStateDiff] = []
    prev_player_locations = previous_state.player_locations
    prev_player_location = prev_player_locations[player_index]
    new_player_location = (
        malmo_observation.get("XPos", prev_player_location[0]),
        malmo_observation.get("YPos", prev_player_location[1]),
        malmo_observation.get("ZPos", prev_player_location[2]),
    )
    if new_player_location != prev_player_location:
        state_diffs.append(
            LocationDiff(
                player_index,
                prev_player_location,
                new_player_location,
            )
        )
    new_player_locations = list(prev_player_locations)
    new_player_locations[player_index] = new_player_location

    return new_player_locations, state_diffs


def _update_breaking_placing(
    previous_state: MalmoState,
    player_index: int,
    malmo_observation: MalmoObservationDict,
    env_config: MbagConfigDict,
    new_player_location: WorldLocation,
) -> Tuple[
    List[bool], List[bool], np.ndarray, np.ndarray, List[Optional[BlockLocation]]
]:
    # Update is breaking and is placing.
    placing_this_timestep = previous_state.player_is_placing[player_index]
    breaking_this_timestep = previous_state.player_is_breaking[player_index]
    new_is_placing = list(previous_state.player_is_placing)
    new_is_breaking = list(previous_state.player_is_breaking)
    if env_config["players"][player_index]["is_human"]:
        for event in malmo_observation.get("events", []):
            if event.get("command") == "use":
                new_is_placing[player_index] = event["pressed"]
                placing_this_timestep = placing_this_timestep or event["pressed"]
            elif event.get("command") == "attack":
                new_is_breaking[player_index] = event["pressed"]
                breaking_this_timestep = breaking_this_timestep or event["pressed"]
    else:
        new_is_placing[player_index] = False
        new_is_breaking[player_index] = False
        for command in malmo_observation.get("CommandsSinceLastObservation", []):
            if command == "use 1":
                new_is_placing[player_index] = True
                placing_this_timestep = True
            elif command == "attack 1":
                new_is_breaking[player_index] = True
                breaking_this_timestep = True

    # Update looking at.
    new_blocks_looking_at = list(previous_state.player_blocks_looking_at)
    new_player_x, new_player_y, new_player_z = new_player_location
    block_looking_at = None
    if "LineOfSight" in malmo_observation:
        line_of_sight = malmo_observation["LineOfSight"]
        if line_of_sight.get("inRange") and line_of_sight["hitType"] == "block":
            looking_x = line_of_sight["x"]
            looking_y = line_of_sight["y"]
            looking_z = line_of_sight["z"]
            if looking_x <= new_player_x and looking_x.is_integer():
                looking_x -= 1
            if looking_y <= new_player_y + 1.6 and looking_y.is_integer():
                looking_y -= 1
            if looking_z <= new_player_z and looking_z.is_integer():
                looking_z -= 1

            width, height, depth = env_config["world_size"]
            if (
                0 <= looking_x < width
                and 0 <= looking_y < height
                and 0 <= looking_z < depth
            ):
                block_looking_at = (
                    int(looking_x),
                    int(looking_y),
                    int(looking_z),
                )
    new_blocks_looking_at[player_index] = block_looking_at

    # Update last breaking and last placing.
    new_last_breaking = previous_state.player_last_breaking.copy()
    new_last_placing = previous_state.player_last_placing.copy()
    block_looking_at = new_blocks_looking_at[player_index]
    if block_looking_at is not None:
        if breaking_this_timestep:
            new_last_breaking[block_looking_at] = player_index
        if placing_this_timestep:
            new_last_placing[block_looking_at] = player_index

    return (
        new_is_breaking,
        new_is_placing,
        new_last_breaking,
        new_last_placing,
        new_blocks_looking_at,
    )


def _update_malmo_inventories(
    previous_state: MalmoState,
    player_index: int,
    malmo_observation: MalmoObservationDict,
) -> Tuple[List[MbagInventory], List[MalmoStateDiff]]:
    state_diffs: List[MalmoStateDiff] = []
    prev_inventory = previous_state.player_inventories[player_index]
    new_inventory: MbagInventory = np.zeros((len(INVENTORY_SLOT_NAMES), 2), dtype=int)
    for slot_index, slot in enumerate(INVENTORY_SLOT_NAMES):
        item_name = malmo_observation[f"InventorySlot_{slot}_item"]  # type: ignore
        block_id = MinecraftBlocks.NAME2ID.get(item_name)
        if block_id is None:
            new_inventory[slot_index, 0] = 0
            new_inventory[slot_index, 1] = 0
        else:
            new_inventory[slot_index, 0] = block_id
            new_inventory[slot_index, 1] = malmo_observation[f"InventorySlot_{slot}_size"]  # type: ignore

    new_inventories = list(previous_state.player_inventories)
    new_inventories[player_index] = new_inventory

    prev_counts = get_block_counts_in_inventory(prev_inventory)
    new_counts = get_block_counts_in_inventory(new_inventory)
    for block_id_with_diff in np.nonzero(new_counts != prev_counts)[0]:
        state_diffs.append(
            InventoryDiff(
                player_index,
                int(block_id_with_diff),
                prev_counts[block_id_with_diff],
                new_counts[block_id_with_diff],
            )
        )

    return new_inventories, state_diffs


def update_malmo_state(
    previous_state: MalmoState,
    player_index: int,
    malmo_observation: MalmoObservationDict,
    env_config: MbagConfigDict,
) -> Tuple[MalmoState, List[MalmoStateDiff]]:
    """
    Updates the Malmo state with the given observation from the given player index.
    Returns the new state and a list of MalmoStateDiffs that represent the differences
    between the previous state and the new state.
    """

    state_diffs: List[MalmoStateDiff] = []

    # Important to get block diffs before inventory diffs because of the way they're
    # processed by get_human_actions.
    if player_index == 0:
        # We only process block diffs from player 0 because all players should receive
        # the same blocks.
        new_blocks, block_diffs = _update_malmo_blocks(
            previous_state, malmo_observation, env_config
        )
        state_diffs.extend(block_diffs)
    else:
        new_blocks = previous_state.blocks

    new_player_locations, location_diffs = _update_malmo_player_locations(
        previous_state, player_index, malmo_observation
    )
    state_diffs.extend(location_diffs)

    (
        new_is_breaking,
        new_is_placing,
        new_last_breaking,
        new_last_placing,
        new_blocks_looking_at,
    ) = _update_breaking_placing(
        previous_state,
        player_index,
        malmo_observation,
        env_config,
        new_player_locations[player_index],
    )

    new_inventories, inventory_diffs = _update_malmo_inventories(
        previous_state, player_index, malmo_observation
    )
    state_diffs.extend(inventory_diffs)

    return (
        MalmoState(
            new_blocks,
            new_inventories,
            new_player_locations,
            new_blocks_looking_at,
            new_is_breaking,
            new_is_placing,
            new_last_breaking,
            new_last_placing,
        ),
        state_diffs,
    )
