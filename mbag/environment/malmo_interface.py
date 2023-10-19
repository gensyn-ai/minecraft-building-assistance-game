import logging
import random
import time
from threading import Condition, Lock, Thread
from typing import List, Optional, Tuple, cast

import numpy as np
from git import Sequence

from mbag.environment.blocks import MalmoState, MinecraftBlocks
from mbag.environment.human_actions import HumanActionDetector
from mbag.environment.types import (
    BlockDiff,
    BlockLocation,
    FacingDirection,
    InventoryDiff,
    LocationDiff,
    MalmoStateDiff,
    MbagAction,
    MbagGiveAIAction,
    MbagInventory,
    MbagInventoryObs,
    MbagMalmoAIAction,
    MbagPlaceBreakAIAction,
    WorldLocation,
)

from .malmo import MalmoClient, MalmoObservationDict

logger = logging.getLogger(__name__)


# TODO: This should be it's own type in MbagTypes at some point
NO_ONE = 0
CURRENT_PLAYER = 1
OTHER_PLAYER = 2
NO_INTERACTION = -1
INVENTORY_NUM_SLOTS = 36
STACK_SIZE = 64


class MalmoInterface:
    def __init__(self, config):
        self.config = config

        self.ai_action_queue = []
        self.ai_action_lock = Condition()
        self.ai_diff_queue = []
        self.ai_diff_queue_lock = Lock()

        self.human_action_queue = []
        self.human_action_lock = Lock()
        self.malmo_client = MalmoClient()
        self.malmo_lock = Lock()
        self.human_action_detector = HumanActionDetector(self.config)

        self.malmo_state_lock = Lock()
        self.done = False

        self.palette_x = self.config["world_size"][0] - 1

    def get_malmo_client(self):
        """
        Dummy method to expose underlying malmo_client API
        Will get deleted after migration is complete
        """
        return self.malmo_client

    def get_malmo_state(self):
        with self.malmo_state_lock:
            return self.malmo_state

    def finish_mission(self):
        # Wait for a second for the final block to place and then end mission.
        logger.info("Ending mission")

        time.sleep(self.config["malmo"]["action_delay"])
        self.done = True

        self.ai_thread.join()
        self.human_thread.join()

        with self.malmo_lock:
            time.sleep(1)
            self.malmo_client.end_mission()

        logger.warning(self.ai_diff_queue)
        logger.info("Ended mission")

    def reset(
        self,
        current_blocks: MinecraftBlocks,
        goal_blocks: MinecraftBlocks,
        last_interacted: np.ndarray,
        player_locations: List[WorldLocation],
        player_directions: List[FacingDirection],
        player_inventories: List[MbagInventory],
    ):
        with self.malmo_state_lock:
            self.malmo_state: MalmoState = {
                "blocks": current_blocks.copy(),
                "player_inventories": [
                    inventory.copy() for inventory in player_inventories
                ],
                "player_locations": player_locations[:],
                "player_directions": list(player_directions),
                "last_interacted": last_interacted.copy(),
                "player_currently_breaking_placing": [
                    False for _ in range(self.config["num_players"])
                ],
            }

        with self.malmo_lock:
            self.malmo_client.start_mission(self.config, current_blocks, goal_blocks)
            time.sleep(1)  # Wait a second for the environment to load.

            # Pre-episode setup in Malmo.
            for player_index in range(self.config["num_players"]):
                player_config = self.config["players"][player_index]
                if not player_config["is_human"]:
                    # Make players fly.
                    for _ in range(2):
                        self.malmo_client.send_command(player_index, "jump 1")
                        time.sleep(0.1)
                        self.malmo_client.send_command(player_index, "jump 0")
                        time.sleep(0.1)
                self.malmo_client.send_command(
                    player_index,
                    "tp "
                    + " ".join(
                        map(str, self.malmo_state["player_locations"][player_index])
                    ),
                )

                # Give items to players.
                for item in self.config["players"][player_index]["give_items"]:
                    if "enchantments" not in item:
                        item["enchantments"] = []

                    for enchantment in item["enchantments"]:
                        assert "id" in enchantment
                        if "level" not in enchantment:
                            enchantment["level"] = 32767

                    enchantments_str = ",".join(
                        [
                            "{{id: {}, lvl: {}}}".format(
                                enchantment["id"], enchantment["level"]
                            )
                            for enchantment in item["enchantments"]
                        ]
                    )

                    self.malmo_client.send_command(
                        player_index,
                        "chat /give {} {} {} {} {}".format(
                            "@p",
                            item["id"],
                            item["count"],
                            0,
                            "{{ench: [{}]}}".format(enchantments_str),
                        ),
                    )

                    time.sleep(0.2)
            # Convert players to survival mode.
            # if not self.config["abilities"]["inf_blocks"]:
            for player_index in range(self.config["num_players"]):
                if self.config["players"][player_index]["is_human"]:
                    self.malmo_client.send_command(player_index, "chat /gamemode 0")

                # Disable chat messages from the palette
                self.malmo_client.send_command(
                    player_index, "chat /gamerule sendCommandFeedback false"
                )

        # Copy goal blocks over
        print("Starting")
        self.copy_palette_from_goal()

        # Start both AI action and human action thread
        self.done = False
        self.ai_thread = Thread(target=self.run_ai_actions)
        self.ai_thread.start()

        self.human_thread = Thread(target=self.run_human_actions)
        self.human_thread.start()

        # TODO: Do the palette here
        time.sleep(self.config["malmo"]["action_delay"])

    def run_human_actions(self):
        # Clear the observations created during the setup stage
        malmo_timestep_obs = self.get_malmo_obs()

        # Continue the loop if there are more observations to process
        # even if the process is done
        while not self.done or len(malmo_timestep_obs) != 0:
            malmo_timestep_obs = self.get_malmo_obs()

            for malmo_observations in malmo_timestep_obs:
                new_state = convert_malmo_obs_to_state(malmo_observations, self.config)
                # print("New State Observations")
                # print(new_state)

                # TODO: this may not be necessary
                if not new_state:
                    new_state = self.malmo_state

                with self.malmo_state_lock:
                    state_diffs = generate_state_diffs(self.malmo_state, new_state)
                for state_diff in state_diffs:
                    print(f"state diff found: {state_diff}")

                    if (
                        isinstance(state_diff, BlockDiff)
                        and state_diff.location[0] == self.palette_x
                    ):
                        if state_diff.received_block == MinecraftBlocks.AIR:
                            self.copy_palette_from_goal()
                        else:
                            continue

                    with self.ai_diff_queue_lock:
                        try:
                            if state_diff in self.ai_diff_queue:
                                self.ai_diff_queue.remove(state_diff)
                            else:
                                logger.warning(
                                    f"Found Diffs between Malmo State and Malmo Observation that were not accounted for by AI Actions: {state_diff}"
                                )
                                # TODO: do the human actions here

                                # human_actions_queue.add_all(
                                #     get_human_actions(self.malmo_state, state_diff)
                                # )
                        except ValueError as e:
                            # raise
                            # TODO: Very occasionally, there will be a value error
                            # when comparing tuples, I'm really not sure why, maybe we should
                            # just ignore it since having extra AI actions sitting around
                            # isn't the worst thing in the world? Or go to default search?

                            logger.warn("Issue with comparing queues")
                            logger.warn(state_diff)
                            logger.warn(self.ai_diff_queue)
                            for i in self.ai_diff_queue:
                                logger.warn(i)
                                logger.warn(state_diff == i)
                                logger.warn(state_diff in [i])
                            raise

                self.update_malmo_state(new_state)

            time.sleep(0)

    def get_human_actions(self):
        pass

    def handle_move(self, player_index, ai_action) -> List[LocationDiff]:
        """
        Executes a player's movement in Malmo. Returns the expected location difference
        """
        # print("handling move")
        # print(ai_action)
        action_type = ai_action.action.action_type
        player_location = self.malmo_state["player_locations"][player_index]

        with self.malmo_lock:
            if MbagAction.MOVE_ACTION_MASK[action_type][1] != "tp":
                self.malmo_client.send_command(
                    player_index, MbagAction.MOVE_ACTION_MASK[action_type][1]
                )
                dx, dy, dz = MbagAction.MOVE_ACTION_MASK[action_type][0]
                new_player_location: WorldLocation = (
                    player_location[0] + dx,
                    player_location[1] + dy,
                    player_location[2] + dz,
                )
                movement_diff = LocationDiff(
                    player_index,
                    player_location,
                    new_player_location,
                )
            else:
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, ai_action.player_location)),
                )
                movement_diff = LocationDiff(
                    player_index,
                    player_location,
                    ai_action.player_location,
                )
            return [movement_diff]

    def remove_block(self, player_index, block_id):
        assert not self.config["abilities"]["inf_blocks"]
        player_name = self.malmo_client.get_player_name(player_index, self.config)
        block_name = MinecraftBlocks.ID2NAME[block_id]

        self.malmo_client.send_command(
            player_index, f"chat /clear {player_name} {block_name} 0 1"
        )

    def add_block(self, player_index, block_id):
        player_name = self.malmo_client.get_player_name(player_index, self.config)
        block_name = MinecraftBlocks.ID2NAME[block_id]

        self.malmo_client.send_command(
            player_index, f"chat /give {player_name} {block_name}"
        )

    def handle_give(
        self, player_index: int, ai_action: MbagGiveAIAction
    ) -> List[InventoryDiff]:
        action, giver_index, receiver_index = (
            ai_action.action,
            ai_action.giver_index,
            ai_action.receiver_index,
        )
        print("handling give")
        print(action, giver_index, receiver_index)
        assert not self.config["abilities"]["inf_blocks"]

        with self.malmo_lock:
            self.remove_block(giver_index, action.block_id)
            self.add_block(receiver_index, action.block_id)

            giver_inventory = get_inventory_obs(
                self.malmo_state["player_inventories"][giver_index]
            )
            receiver_inventory = get_inventory_obs(
                self.malmo_state["player_inventories"][receiver_index]
            )
            return [
                InventoryDiff(
                    giver_index,
                    action.block_id,
                    giver_inventory[action.block_id],
                    giver_inventory[action.block_id] - 1,
                ),
                InventoryDiff(
                    receiver_index,
                    action.block_id,
                    receiver_inventory[action.block_id],
                    receiver_inventory[action.block_id] + 1,
                ),
            ]

    def handle_place_break(self, player_index: int, ai_action: MbagPlaceBreakAIAction):
        print("handling break place")

        with self.malmo_lock:
            expected_diffs: List[MalmoStateDiff] = []
            action, player_location, yaw, pitch, inventory_slot = (
                ai_action.action,
                ai_action.player_location,
                ai_action.yaw,
                ai_action.pitch,
                ai_action.inventory_slot,
            )

            if self.config["abilities"]["teleportation"]:
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, player_location)),
                )
                expected_diffs.append(
                    LocationDiff(
                        player_index,
                        self.malmo_state["player_locations"][player_index],
                        player_location,
                    )
                )

            self.malmo_client.send_command(player_index, f"setYaw {yaw}")
            self.malmo_client.send_command(player_index, f"setPitch {pitch}")

            if action.action_type == MbagAction.PLACE_BLOCK:
                if self.config["abilities"]["inf_blocks"]:
                    self.malmo_client.send_command(
                        player_index,
                        f"swapInventoryItems 0 {action.block_id}",
                    )
                    hotbar_slot = 0
                else:
                    player_inventory = self.malmo_state["player_inventories"][
                        player_index
                    ]
                    if inventory_slot < 9:
                        hotbar_slot = inventory_slot
                    else:
                        # Block is not in hotbar, need to swap it in.
                        hotbar_slot = random.randrange(9)
                        self.malmo_client.send_command(
                            player_index,
                            f"swapInventoryItems {hotbar_slot} {inventory_slot}",
                        )
                        (
                            player_inventory[hotbar_slot],
                            player_inventory[inventory_slot],
                        ) = (
                            player_inventory[inventory_slot].copy(),
                            player_inventory[hotbar_slot].copy(),
                        )

                self.malmo_client.send_command(
                    player_index, f"hotbar.{hotbar_slot + 1} 1"
                )
                self.malmo_client.send_command(
                    player_index, f"hotbar.{hotbar_slot + 1} 0"
                )
                time.sleep(0.1)  # Give time to swap item to hand and teleport.
                self.malmo_client.send_command(player_index, "use 1")
                time.sleep(0.1)  # Give time to place block.
                if self.config["abilities"]["inf_blocks"]:
                    self.malmo_client.send_command(
                        player_index,
                        f"swapInventoryItems 0 {action.block_id}",
                    )

                expected_diffs.append(
                    BlockDiff(
                        action.block_location, MinecraftBlocks.AIR, action.block_id
                    )
                )

                player_inventory_obs = get_inventory_obs(
                    self.malmo_state["player_inventories"][player_index]
                )

                if not self.config["abilities"]["inf_blocks"]:
                    self.remove_block(player_index, action.block_id)
                    expected_diffs.append(
                        InventoryDiff(
                            player_index,
                            action.block_id,
                            player_inventory_obs[action.block_id],
                            player_inventory_obs[action.block_id] - 1,
                        )
                    )

            else:
                time.sleep(0.1)  # Give time to teleport.
                self.malmo_client.send_command(player_index, "attack 1")
                block_id_broken = self.malmo_state["blocks"][action.block_location][0]
                expected_diffs.append(
                    BlockDiff(
                        action.block_location,
                        block_id_broken,
                        MinecraftBlocks.AIR,
                    )
                )

                player_inventory_obs = get_inventory_obs(
                    self.malmo_state["player_inventories"][player_index]
                )

                expected_diffs.append(
                    InventoryDiff(
                        player_index,
                        block_id_broken,
                        player_inventory_obs[block_id_broken],
                        player_inventory_obs[block_id_broken] + 1,
                    )
                )

                # TODO: This was in here to fix a bug with blocks and silk touch
                # Figure it out later.
                # if not self.config["abilities"]["inf_blocks"]:
                #     self.add_block(player_index, action.block_id)

            return expected_diffs

    def add_ai_action(self, player_index: int, action: MbagMalmoAIAction):
        # print("Adding AI Action")
        # print(action)
        with self.ai_action_lock:
            self.ai_action_queue.append((player_index, action))
            # print(self.ai_action_queue)
            self.ai_action_lock.notify()

    def running_ai_actions(self):
        return len(self.ai_action_queue) == 0 and not self.malmo_lock.locked()

    def get_current_malmo_state(self):
        return self.malmo_state

    def get_malmo_obs(self) -> List[List[MalmoObservationDict]]:
        """
        Gets the latest observations from Malmo, indexed first by timestamp, then by player.
        There may not be any Malmo observations
        """
        malmo_observations: List[List[MalmoObservationDict]] = [[]] * self.config[
            "num_players"
        ]
        with self.malmo_lock:
            for player_index in range(self.config["num_players"]):
                malmo_player_observations = self.malmo_client.get_observations(
                    player_index
                )

                # print(f"Player {player_index} obs:")
                # print(malmo_player_observations)

                if len(malmo_player_observations) > 0:
                    malmo_observations[player_index] = [
                        x for _, x in sorted(malmo_player_observations)
                    ]

        return [list(x) for x in list(zip(*malmo_observations))]

    def update_malmo_state(self, new_state: MalmoState):
        # TODO: Do a check to make sure that these values actually exist?
        with self.malmo_state_lock:
            blocks = new_state.get("blocks", None)
            if blocks:
                self.malmo_state["blocks"] = blocks

            self.malmo_state["player_directions"] = new_state.get(
                "player_directions", []
            )
            self.malmo_state["player_locations"] = new_state.get("player_locations", [])
            self.malmo_state["player_inventories"] = new_state.get(
                "player_inventories", []
            )

    def run_ai_actions(self):
        while True:
            player_index, ai_action = -1, None

            # print("Checking AI actions, ", self.ai_action_queue)
            with self.ai_action_lock:
                self.ai_action_lock.wait_for(
                    lambda: len(self.ai_action_queue) > 0 or self.done
                )
                if self.done:
                    return

                # print("Waited for action", self.ai_action_queue)
                player_index, ai_action = self.ai_action_queue.pop(0)

            if player_index == -1:
                logger.error("Did not find AI action in queue")
                time.sleep(0)
                continue

            assert not self.config["players"][player_index]["is_human"]
            # print("Processing Action", ai_action.action, ai_action.action.action_type)
            ai_expected_diffs = []
            if (
                ai_action.action.action_type == MbagAction.PLACE_BLOCK
                or ai_action.action.action_type == MbagAction.BREAK_BLOCK
            ):
                ai_expected_diffs = self.handle_place_break(player_index, ai_action)
            elif ai_action.action.action_type in MbagAction.MOVE_ACTION_TYPES:
                ai_expected_diffs = self.handle_move(player_index, ai_action)
            elif ai_action.action.action_type == MbagAction.GIVE_BLOCK:
                ai_expected_diffs = self.handle_give(player_index, ai_action)

            with self.ai_diff_queue_lock:
                self.ai_diff_queue.extend(ai_expected_diffs)
            print(f"AI Expected Diffs {ai_expected_diffs}")

            # Wait for AI expected diffs to come through the observations
            time.sleep(self.config["malmo"]["action_delay"])
            for ai_diff in self.ai_diff_queue:
                logger.warning(f"Expected AI diff not seen: {ai_diff}")

    def copy_palette_from_goal(self):
        # Sync with Malmo.
        with self.malmo_lock:
            width, height, depth = self.config["world_size"]
            goal_palette_x = self.palette_x + width + 1

            self.malmo_client.send_command(
                0,
                f"chat /clone {goal_palette_x} 0 0 "
                f"{goal_palette_x} {height - 1} {depth - 1} "
                f"{self.palette_x} 0 0",
            )
            time.sleep(0.3)


def generate_state_diffs(
    reference_state: MalmoState, updated_state: MalmoState
) -> List[MalmoStateDiff]:
    diffs: List[MalmoStateDiff] = []

    reference_blocks = reference_state["blocks"].blocks
    updated_blocks = updated_state["blocks"].blocks
    for location in cast(
        Sequence[BlockLocation],
        map(
            tuple,
            np.argwhere((reference_blocks != updated_blocks)),
        ),
    ):
        diffs.append(
            BlockDiff(
                location,
                reference_blocks[location],
                updated_blocks[location],
            )
        )
        # logger.info(
        #     f"BlockDiff at {location}: "
        #     "expected "
        #     f"{MinecraftBlocks.ID2NAME[reference_blocks[location]]} "
        #     f"in MalmoInterface but received "
        #     f"{MinecraftBlocks.ID2NAME[updated_blocks[location]]} "
        #     "from Malmo"
        # )

    # TODO: Somehow this is buggy in test_two_players_in_malmo
    for player_id, reference_location in enumerate(
        reference_state.get("player_locations", [])
    ):
        updated_location = updated_state.get("player_locations")[player_id]
        if any(
            abs(malmo_coord - stored_coord) > 1e-4
            for malmo_coord, stored_coord in zip(reference_location, updated_location)
        ):
            # logger.info(
            #     f"LocationDiff for player {player_id}: "
            #     f"expected {reference_location} in MalmoInterface but received "
            #     f"{updated_location} from Malmo"
            # )
            diffs.append(LocationDiff(player_id, reference_location, updated_location))

    for player_id, reference_inventory in enumerate(
        reference_state.get("player_inventories", [])
    ):
        reference_inventory_obs = get_inventory_obs(reference_inventory)
        updated_inventory_obs = get_inventory_obs(
            updated_state.get("player_inventories")[player_id]
        )

        for block_id in np.nonzero(reference_inventory_obs != updated_inventory_obs)[0]:
            # logger.info(
            #     f"InventoryDiff for player {player_id} in block {MinecraftBlocks.ID2NAME[block_id]}:"
            #     f"expected {reference_inventory_obs[block_id]}"
            #     f"but received {updated_inventory_obs[block_id]}"
            #     "from Malmo"
            # )

            diffs.append(
                InventoryDiff(
                    player_id,
                    block_id,
                    reference_inventory_obs[block_id],
                    updated_inventory_obs[block_id],
                )
            )

    return diffs


# Returns best effort malmo state given the malmo observation
# TODO: some of the types are weird, maybe we can just pass a fallback?
def convert_malmo_obs_to_state(
    obs: List[MalmoObservationDict], config
) -> Optional[MalmoState]:
    """

    Takes in a list of MalmoObservationDicts and returns a MalmoState object. Best
    effort is used when there is missing data.
    """

    assert len(obs) == len(config["players"])
    if not obs[0]:
        return None

    malmo_blocks = None
    global_obs = obs[0]

    if "world" in global_obs:
        malmo_blocks = MinecraftBlocks.from_malmo_grid(
            config["world_size"], global_obs["world"]
        )
    else:
        logger.warning("No block information from Malmo")

    malmo_inventories: List[MbagInventory] = [None for _ in obs]
    malmo_locations: List[WorldLocation] = [None for _ in obs]
    malmo_directions: List[FacingDirection] = [None for _ in obs]

    for player_index in range(config["num_players"]):
        player_obs = obs[player_index]

        if "InventorySlot_0_item" in player_obs:
            malmo_inventory: MbagInventory = np.zeros(
                (INVENTORY_NUM_SLOTS, 2), dtype=int
            )
            for slot in range(INVENTORY_NUM_SLOTS):
                item_name = player_obs[f"InventorySlot_{slot}_item"]  # type: ignore
                malmo_inventory[slot, 0] = MinecraftBlocks.NAME2ID.get(item_name, 0)
                malmo_inventory[slot, 1] = player_obs[f"InventorySlot_{slot}_size"]  # type: ignore

            malmo_inventories[player_index] = malmo_inventory
        else:
            logger.warning(
                "missing inventory information from Malmo observation "
                f"(keys = {player_obs.keys()})"
            )

        if not config["abilities"]["teleportation"]:
            # Make sure position is as expected.
            location = (
                player_obs.get("XPos", None),
                player_obs.get("YPos", None),
                player_obs.get("ZPos", None),
            )
            malmo_locations[player_index] = location

            direction = (player_obs.get("Pitch", None), player_obs.get("Yaw", None))
            malmo_directions[player_index] = direction

    malmo_state: MalmoState = {
        "blocks": malmo_blocks,
        "player_inventories": malmo_inventories,
        "player_locations": malmo_locations,
        "player_directions": malmo_directions,
        # TODO: Figure out how to get the following locations
        "last_interacted": np.zeros(config["world_size"]),
        "player_currently_breaking_placing": [
            False for _ in range(config["num_players"])
        ],
    }

    return malmo_state


def get_inventory_obs(player_inventory: MbagInventory) -> MbagInventoryObs:
    """
    Gets the array representation of the given player's inventory.
    """

    inventory_obs: MbagInventoryObs = np.zeros(
        MinecraftBlocks.NUM_BLOCKS, dtype=int
    )  # 10 total blocks
    for i in range(player_inventory.shape[0]):
        inventory_obs[player_inventory[i][0]] += player_inventory[i][1]

    return inventory_obs


# - in human action detection thread


# malmo_observation_dicts = ...
# for malmo_observation_dict in malmo_observation_dicts:
# 		new_state, state_diffs = update_malmo_state(self.malmo_state, malmo_observation_dict)
# 		for state_diff in state_diffs:
# 				if state_diff in self.expected_diffs:
# 						self.expected_diffs.remove(state_diff)
# 				else:
# 						human_actions_queue.add_all(get_human_actions(self.malmo_state, state_diff))
# 		self.malmo_state = new_state

# # - in AI actions thread

# with self.running_ai_actions_lock:
# 		self.running_ai_actions = True
# while not ai_action_queue.empty():
#     ai_action = ai_action_queue.pop()
# 		ai_expected_diffs = get_expected_diffs(ai_action)
# 		self.expected_diffs.extend(ai_expected_diffs)
# 	  run_ai_action(ai_action)  # Actually make the action happen in Malmo, can block while AI action runs (e.g. time.sleep)
# self.running_ai_actions = False
# for expected_diff in self.expected_diffs:
# 		# expected_diff should have been seen and cleared by now
# 		logger.warning("expected diff not seen")
# self.expected_diffs.clear()

# # In MbagEnv.step

# normal_step(actions)

# begin = time()
# while time() - begin < 0.2 and human_actions_queue.empty():
# 		sleep(0.001)

# if not malmo_interface.running_ai_actions and human_actions_queue.empty():
# 		self.update_malmo_state(malmo_interface.malmo_state)  # force-update MbagEnv state from MalmoState to make sure everything is in sync if there are no pending AI/human actions

# infos[human_index]["human_action"] = human_actions_queue.pop()

# # Don't put a queue in HumanAgent; instead, just have it play back whatever action is in info["human_action"]. By keeping the human action queue in MbagEnv, we know when there are still more human actions to be played.
