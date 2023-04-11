import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .blocks import MinecraftBlocks
from .types import (
    BlockLocation,
    MbagAction,
    MbagActionTuple,
    MbagActionType,
    MbagInfoDict,
    MbagInventory,
    MbagInventoryObs,
    WorldLocation,
)

if TYPE_CHECKING:
    from .malmo import MalmoObservationDict
    from .mbag_env import MbagConfigDict


logger = logging.getLogger(__name__)


class HumanActionDetector(object):
    human_block_looking_at: List[Optional[BlockLocation]]
    """For each player, which block they are currently looking at."""
    human_blocks_on_ground: List[Dict[int, int]]
    """For each player, number of blocks of each inventory type that is on the ground"""
    human_is_breaking: List[bool]
    human_is_placing: List[bool]
    """For each player, whether they are currently holding the place/break keys."""
    human_locations: List[BlockLocation]
    """The current location of each human in Malmo (may be ahead of env state)."""
    human_last_placing: np.ndarray
    """For each block, the player index of which human was last holding break on it."""
    human_last_breaking: np.ndarray
    """For each block, the player index of which human was last holding place on it."""
    malmo_blocks: List[MinecraftBlocks]
    """
    The last blocks from Malmo; kept separate for each player for consistency with
    other attributes.
    """
    malmo_inventories: List[MbagInventory]
    """The last inventories from Malmo"""
    num_pending_human_interactions: np.ndarray
    """
    How many human interactions have occurred with each block that have yet to be
    reflected in the Python env. This is incremented each time a new human action is
    generated and decremented each time an action is passed to the env and updates
    that location. For locations where this is nonzero, discrepancies between Malmo
    and the Python env are ignored since they reflect human changes.
    """

    def __init__(self, env_config: "MbagConfigDict"):
        self.env_config = env_config

    def reset(
        self,
        initial_player_locations: List[WorldLocation],
        initial_blocks: MinecraftBlocks,
        palette_x: int,
    ):
        """
        This should be called at the beginning of a new episode.
        """

        from .mbag_env import MbagEnv

        self.human_block_looking_at = [
            None for _ in range(self.env_config["num_players"])
        ]
        self.human_blocks_on_ground = [
            defaultdict(int) for _ in range(self.env_config["num_players"])
        ]
        self.human_is_breaking = [False for _ in range(self.env_config["num_players"])]
        self.human_is_placing = [False for _ in range(self.env_config["num_players"])]
        self.human_locations = [
            (int(x), int(y), int(z)) for x, y, z in initial_player_locations
        ]
        self.human_last_breaking = np.full(
            self.env_config["world_size"], -1, dtype=np.int8
        )
        self.human_last_placing = np.full(
            self.env_config["world_size"], -1, dtype=np.int8
        )
        self.malmo_blocks = [
            initial_blocks.copy() for _ in range(self.env_config["num_players"])
        ]
        self.malmo_inventories = [
            np.zeros((MbagEnv.INVENTORY_NUM_SLOTS, 2), dtype=int)
            for _ in range(self.env_config["num_players"])
        ]

        self.num_pending_human_interactions = np.zeros(
            self.env_config["world_size"], dtype=np.int8
        )
        self.num_pending_human_movements = np.zeros(self.env_config["num_players"])

        self.palette_x = palette_x

    def get_human_actions(
        self,
        human_players: List[int],
        infos: List[MbagInfoDict],
    ) -> List[Tuple[int, MbagActionTuple]]:
        actions = []
        human_actions = {}
        timestamps: List[datetime] = []

        for player_index in human_players:
            one_human_action = self.get_one_human_actions(
                player_index, infos[player_index]["malmo_observations"]
            )
            timestamps.extend(one_human_action.keys())
            human_actions[player_index] = one_human_action

        timestamps = sorted([*set(timestamps)])

        # For each timestamp, pad each category of movement
        for time in timestamps:
            for index in range(3):
                largest_size = max(
                    [
                        len(human_actions[player_index][time][index])
                        for player_index in human_players
                    ]
                )
                for player_index in human_players:
                    total_actions = human_actions[player_index][time][index] + [
                        (player_index, (MbagAction.NOOP, 0, 0))
                        for _ in range(
                            largest_size - len(human_actions[player_index][time][index])
                        )
                    ]
                    actions.extend(total_actions)
                    # actions.extend(human_actions[player_index][time][index])

        for player_index, action_tuple in actions:
            if action_tuple[0] == MbagAction.NOOP:
                logger.info(f"padding action from player {player_index}")
            else:
                logger.info(
                    f"human action from player {player_index}: "
                    + str(MbagAction(action_tuple, self.env_config["world_size"]))
                )

        return actions

    def get_one_human_actions(
        self,
        player_index: int,
        malmo_observations: List[Tuple[datetime, "MalmoObservationDict"]],
    ) -> Dict[datetime, List[List[Tuple[int, MbagActionTuple]]]]:
        """
        Given a player index and a list of observation dictionaries from Malmo,
        determines which human actions have taken place since the last time the
        method was called. Returns a list of (player_index, action_tuple) pairs
        since occasionally an action must be generated for another player; for instance,
        if this player just picked up a block another player dropped, then it generates
        a GIVE_BLOCK action for the other player.
        """

        actions: Dict[datetime, List[List[Tuple[int, MbagActionTuple]]]] = defaultdict(
            lambda: [[], [], []]
        )

        for observation_time, malmo_observation in sorted(malmo_observations):
            timestamp_actions: List[List[Tuple[int, MbagActionTuple]]] = [[], [], []]
            block_discrepancies = self._get_block_discrepancies(
                player_index, malmo_observation
            )
            dropped_blocks, picked_up_blocks = self._get_dropped_picked_up_blocks(
                player_index, malmo_observation
            )
            timestamp_actions[0] = self._get_movement_actions(
                player_index, malmo_observation
            )
            timestamp_actions[1] = self._get_place_break_actions(
                player_index,
                malmo_observation,
                block_discrepancies,
                dropped_blocks,
            )

            timestamp_actions[2] = self._handle_dropped_picked_up_blocks(
                player_index, dropped_blocks, picked_up_blocks
            )

            actions[observation_time] = timestamp_actions

        return actions

    def _copy_palette(
        self,
        palette_x: int,
        palette_blocks: np.ndarray,
        palette_block_states: np.ndarray,
    ):
        pass

    def sync_human_state(self, player_index, player_location, player_inventory):
        if (
            self.num_pending_human_interactions.sum() > 0
            or self.num_pending_human_movements.sum() > 0
        ):
            logger.info(
                "Skipping human action detector sync because of outstanding human actions"
            )
            return

        # Make sure inventory has the same number of blocks as the environment's env
        human_inventory_obs = self._get_simplified_inventory(
            self.malmo_inventories[player_index]
        )
        player_inventory_obs = self._get_simplified_inventory(player_inventory)
        for slot in np.nonzero(human_inventory_obs != player_inventory_obs)[0]:
            logger.warning(
                f"inventory discrepancy for player {player_index} for {MinecraftBlocks.ID2NAME[slot]}: "
                f"expected {player_inventory_obs[slot]} "
                f"but received {human_inventory_obs[slot]} "
                "from human action detector"
            )

        # Make sure position is the same as the environment
        human_location = self.human_locations[player_index]
        player_location = (
            int(player_location[0]),
            int(player_location[1]),
            int(player_location[2]),
        )
        if any(
            abs(malmo_coord - stored_coord) > 1e-4
            for malmo_coord, stored_coord in zip(human_location, player_location)
        ):
            logger.warning(
                f"location discrepancy for player {player_index}: "
                f"expected {player_location} but received "
                f"{human_location} from human action detector"
            )
            self.human_locations[player_index] = player_location

    def _get_simplified_inventory(
        self, player_inventory: MbagInventory
    ) -> MbagInventoryObs:
        """
        Gets the array representation of the given player's inventory.
        """

        inventory_obs: MbagInventoryObs = np.zeros(
            MinecraftBlocks.NUM_BLOCKS, dtype=int
        )  # 10 total blocks
        for i in range(player_inventory.shape[0]):
            inventory_obs[player_inventory[i][0]] += player_inventory[i][1]

        inventory_obs[MinecraftBlocks.AIR] = 0
        return inventory_obs

    def _get_block_discrepancies(
        self, player_index: int, malmo_observation: "MalmoObservationDict"
    ) -> Dict[BlockLocation, Tuple[int, int]]:
        """
        Given an observation dict from Malmo, returns a dict mapping from all
        block locations that have changed since the last Malmo observation
        to a tuple of (current_block, prev_block). Also updates self.malmo_blocks.
        """

        block_discrepancies: Dict[BlockLocation, Tuple[int, int]] = {}
        if "world" in malmo_observation:
            prev_malmo_blocks = self.malmo_blocks[player_index]
            current_malmo_blocks = MinecraftBlocks.from_malmo_grid(
                self.env_config["world_size"], malmo_observation["world"]
            )
            for location in cast(
                Sequence[BlockLocation],
                map(
                    tuple,
                    np.argwhere(
                        current_malmo_blocks.blocks != prev_malmo_blocks.blocks
                    ),
                ),
            ):
                block_discrepancies[location] = (
                    current_malmo_blocks.blocks[location],
                    prev_malmo_blocks.blocks[location],
                )
            self.malmo_blocks[player_index] = current_malmo_blocks
        return block_discrepancies

    def _get_movement_actions(
        self, player_index: int, malmo_observation: "MalmoObservationDict"
    ) -> List[Tuple[int, MbagActionTuple]]:
        """
        Given an observation dict from Malmo for a particular player, compares the
        current player's
        """

        movement_actions: List[Tuple[int, MbagActionTuple]] = []
        current_x, current_y, current_z = self.human_locations[player_index]
        malmo_x = int(malmo_observation.get("XPos", current_x))
        malmo_y = int(malmo_observation.get("YPos", current_y))
        malmo_z = int(malmo_observation.get("ZPos", current_z))
        while current_x < malmo_x:
            movement_actions.append((player_index, (MbagAction.MOVE_POS_X, 0, 0)))
            current_x += 1
        while current_x > malmo_x:
            movement_actions.append((player_index, (MbagAction.MOVE_NEG_X, 0, 0)))
            current_x -= 1
        while current_y < malmo_y:
            movement_actions.append((player_index, (MbagAction.MOVE_POS_Y, 0, 0)))
            current_y += 1
        while current_y > malmo_y:
            movement_actions.append((player_index, (MbagAction.MOVE_NEG_Y, 0, 0)))
            current_y -= 1
        while current_z < malmo_z:
            movement_actions.append((player_index, (MbagAction.MOVE_POS_Z, 0, 0)))
            current_z += 1
        while current_z > malmo_z:
            movement_actions.append((player_index, (MbagAction.MOVE_NEG_Z, 0, 0)))
            current_z -= 1
        self.human_locations[player_index] = (current_x, current_y, current_z)
        self.num_pending_human_movements[player_index] += len(movement_actions)
        return movement_actions

    def _get_place_break_actions(
        self,
        player_index: int,
        malmo_observation: "MalmoObservationDict",
        block_discrepancies: Dict[BlockLocation, Tuple[int, int]],
        dropped_blocks: Dict[int, int],
    ) -> List[Tuple[int, MbagActionTuple]]:
        # Update human_is_placing and human_is_breaking.
        placing_this_timestep = self.human_is_placing[player_index]
        breaking_this_timestep = self.human_is_breaking[player_index]
        for event in malmo_observation.get("events", []):
            if event.get("command") == "use":
                self.human_is_placing[player_index] = event["pressed"]
                placing_this_timestep = placing_this_timestep or event["pressed"]
            elif event.get("command") == "attack":
                self.human_is_breaking[player_index] = event["pressed"]
                breaking_this_timestep = breaking_this_timestep or event["pressed"]

        # Update human_last_placing and human_last_breaking.
        block_looking_at = self.human_block_looking_at[player_index]
        if block_looking_at is not None:
            if placing_this_timestep:
                self.human_last_placing[block_looking_at] = player_index
            if breaking_this_timestep:
                self.human_last_breaking[block_looking_at] = player_index

        # Update human_block_looking_at.
        current_x, current_y, current_z = self.human_locations[player_index]
        block_looking_at = None
        if "LineOfSight" in malmo_observation:
            line_of_sight = malmo_observation["LineOfSight"]
            if line_of_sight.get("inRange") and line_of_sight["hitType"] == "block":
                looking_x = line_of_sight["x"]
                looking_y = line_of_sight["y"]
                looking_z = line_of_sight["z"]
                if current_x >= looking_x and looking_x.is_integer():
                    looking_x -= 1
                if looking_y <= current_y + 1.6 and looking_y.is_integer():
                    looking_y -= 1
                if looking_z <= current_z and looking_z.is_integer():
                    looking_z -= 1

                width, height, depth = self.env_config["world_size"]
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
        self.human_block_looking_at[player_index] = block_looking_at

        # Handle place/break actions.
        place_break_actions: List[Tuple[int, MbagActionTuple]] = []
        for block_location, (new_block_id, current_block_id) in list(
            block_discrepancies.items()
        ):
            action_type: Optional[MbagActionType] = None
            if (
                self.human_last_placing[block_location] == player_index
                and new_block_id != MinecraftBlocks.AIR
                and block_location[0] != self.palette_x
            ):
                action_type = MbagAction.PLACE_BLOCK
                dropped_blocks[new_block_id] -= 1
            if (
                self.human_last_breaking[block_location] == player_index
                and new_block_id == MinecraftBlocks.AIR
            ):
                action_type = MbagAction.BREAK_BLOCK
                dropped_blocks[current_block_id] += 1
            if action_type is not None:
                place_break_actions.append(
                    (
                        player_index,
                        (
                            action_type,
                            int(
                                np.ravel_multi_index(
                                    block_location,
                                    self.env_config["world_size"],
                                )
                            ),
                            new_block_id,
                        ),
                    )
                )

                del block_discrepancies[block_location]
                self.num_pending_human_interactions[block_location] += 1

        return place_break_actions

    def _get_dropped_picked_up_blocks(
        self, player_index: int, malmo_observation: "MalmoObservationDict"
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compares the player's latest Malmo inventory to the last one received and
        determines if there are any blocks that seem to have been dropped or picked
        up. Returns two dictionaries which map from block IDs to the number of blocks
        of that type that have been dropped and picked up, respectively. Also
        updates malmo_inventories for the given player.
        """

        if "InventorySlot_0_item" not in malmo_observation:
            return {}, {}

        from .mbag_env import MbagEnv

        past_malmo_inventory = self.malmo_inventories[player_index]
        malmo_inventory: MbagInventory = np.zeros(
            (MbagEnv.INVENTORY_NUM_SLOTS, 2), dtype=int
        )
        for slot in range(MbagEnv.INVENTORY_NUM_SLOTS):
            item_name = malmo_observation[f"InventorySlot_{slot}_item"]  # type: ignore
            malmo_inventory[slot, 0] = MinecraftBlocks.NAME2ID.get(item_name, 0)
            malmo_inventory[slot, 1] = malmo_observation[f"InventorySlot_{slot}_size"]  # type: ignore

        # Mapping id of each block --> number of each block we think is dropped
        dropped_blocks: Dict[int, int] = defaultdict(int)

        # Mapping id of each block --> number of each block we think is picked up
        picked_up_blocks: Dict[int, int] = defaultdict(int)

        for slot in np.nonzero(np.any(malmo_inventory != past_malmo_inventory, axis=1))[
            0
        ]:
            if MinecraftBlocks.ID2NAME[malmo_inventory[slot, 0]] == "air" or (
                past_malmo_inventory[slot, 0] == malmo_inventory[slot, 0]
                and past_malmo_inventory[slot, 1] > malmo_inventory[slot, 1]
            ):
                dropped_blocks[past_malmo_inventory[slot, 0]] += (
                    past_malmo_inventory[slot, 1] - malmo_inventory[slot, 1]
                )

            if MinecraftBlocks.ID2NAME[past_malmo_inventory[slot, 0]] == "air" or (
                past_malmo_inventory[slot, 0] == malmo_inventory[slot, 0]
                and past_malmo_inventory[slot, 1] < malmo_inventory[slot, 1]
            ):
                picked_up_blocks[malmo_inventory[slot, 0]] += (
                    malmo_inventory[slot, 1] - past_malmo_inventory[slot, 1]
                )
        self.malmo_inventories[player_index] = malmo_inventory

        return dropped_blocks, picked_up_blocks

    def _handle_dropped_picked_up_blocks(
        self,
        player_index: int,
        dropped_blocks: Dict[int, int],
        picked_up_blocks: Dict[int, int],
    ) -> List[Tuple[int, MbagActionTuple]]:
        """
        Based on the blocks dropped and picked up, updates human_blocks_on_ground
        and also returns a list of actions which may include GIVE_BLOCK if one of the
        picked up blocks came from another player.
        """

        # Write remaining dropped blocks into the player's dropped blocks log
        for dropped_block_id, dropped_block_quantity in list(dropped_blocks.items()):
            self.human_blocks_on_ground[player_index][
                dropped_block_id
            ] += dropped_block_quantity

        # Handle effects of picking up a block
        actions: List[Tuple[int, MbagActionTuple]] = []
        for picked_block_id, picked_block_quantity in list(picked_up_blocks.items()):
            players_to_iterate = [player_index] + list(
                range(self.env_config["num_players"])
            )
            for other_player_index in players_to_iterate:
                player_picked_blocks = min(
                    self.human_blocks_on_ground[other_player_index][picked_block_id],
                    picked_block_quantity,
                )

                picked_block_quantity -= player_picked_blocks
                self.human_blocks_on_ground[other_player_index][
                    picked_block_id
                ] -= player_picked_blocks

                if other_player_index != player_index:
                    logger.info(
                        (
                            MbagAction.GIVE_BLOCK,
                            int(
                                np.ravel_multi_index(
                                    self.human_locations[player_index],
                                    self.env_config["world_size"],
                                )
                            ),
                            picked_block_id,
                        )
                    )
                    player_tag = player_index + 1
                    if player_index < other_player_index:
                        player_tag += 1
                    actions.extend(
                        [
                            (
                                other_player_index,
                                (
                                    MbagAction.GIVE_BLOCK,
                                    player_tag,
                                    picked_block_id,
                                ),
                            )
                            for _ in range(player_picked_blocks)
                        ]
                    )

        return actions

    def record_human_movement(self, player_id: int):
        if self.num_pending_human_movements[player_id] > 0:
            self.num_pending_human_movements[player_id] -= 1
        else:
            logger.error(f"unexpected movement action from human player {player_id}")

    def record_human_interaction(self, block_location: BlockLocation):
        if self.num_pending_human_interactions[block_location] > 0:
            self.num_pending_human_interactions[block_location] -= 1
        else:
            logger.error(
                f"unexpected block action from human player at location {block_location}"
            )

    @property
    def blocks_with_no_pending_human_interactions(self):
        return self.num_pending_human_interactions == 0
