from typing import List, Tuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.human_actions import HumanActionDetector
from mbag.environment.types import (
    MalmoStateDiff,
    MbagAction,
    MalmoState,
    MbagInventory,
    WorldLocation,
)
from .malmo import MalmoClient, MalmoObservationDict
import numpy as np
import time
from threading import Thread, Lock
from .mbag_env import MalmoConfigDict


# TODO: This should be it's own type in MbagTypes at some point
NO_ONE = 0
CURRENT_PLAYER = 1
OTHER_PLAYER = 2
NO_INTERACTION = -1


class MalmoInterface:
    def __init__(self, config: MalmoConfigDict):
        self.ai_action_queue = []
        self.ai_action_lock = Lock()
        self.human_action_queue = []
        self.human_action_lock = Lock()
        self.malmo_client = MalmoClient()
        self.malmo_lock = Lock()
        self.human_action_detector = HumanActionDetector(self.config)
        self.config = config

    def get_malmo_client(self):
        """
        Dummy method to expose underlying malmo_client API
        Will get deleted after migration is complete
        """
        return self.malmo_client

    def done(self):
        # Wait for a second for the final block to place and then end mission.
        self.ai_thread.join()

        with self.malmo_lock:
            time.sleep(1)
            self.malmo_client.end_mission()

        # TODO: Assuming this should happen after the AI so that all the moves have time to get processed
        self.human_thread.join()

    def reset(
        self,
        current_blocks: MinecraftBlocks,
        goal_blocks: MinecraftBlocks,
        last_interacted: np.ndarray,
        player_locations: List[WorldLocation],
        player_directions: List[Tuple],
        player_inventories: List[MbagInventory],
    ):
        self.malmo_state: MalmoState = {
            "blocks": current_blocks,
            "player_inventories": player_inventories,
            "player_locations": player_locations,
            "player_directions": player_directions,
            "last_interacted": last_interacted,
            "player_currently_breaking_placing": [
                False for _ in range(self.config["num_players"])
            ],
        }

        self.ai_thread = Thread(target=self.run_ai_actions)
        self.ai_thread.run()

        # TODO: Placeholder for now
        self.human_thread = Thread(target=self.run_human_actions)
        self.human_thread.run()

        with self.malmo_lock:
            self.malmo_client.start_mission(config, current_blocks, goal_blocks)
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
                    "tp " + " ".join(map(str, self.player_locations[player_index])),
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

            # TODO: Do the palette here
            time.sleep(self.config["malmo"]["action_delay"])

    def update_malmo_state(
        previous_state: MalmoState, malmo_obs: MalmoObservationDict
    ) -> Tuple[MalmoState, List[MalmoStateDiff]]:
        pass

    def run_human_actions(self):
        malmo_observation_dicts = self.malmo_client.get_observations()
        for malmo_observation_dict in malmo_observation_dicts:
            new_state, state_diffs = self.update_malmo_state(
                self.malmo_state, malmo_observation_dict
            )

    # 		for state_diff in state_diffs:
    # 				if state_diff in self.expected_diffs:
    # 						self.expected_diffs.remove(state_diff)
    # 				else:
    # 						human_actions_queue.add_all(get_human_actions(self.malmo_state, state_diff))
    # 		self.malmo_state = new_state

    def get_human_actions(self):
        pass

    def handle_move(self, player_index, action):
        action_type = action[0]
        with self.malmo_lock:
            if MbagAction.MOVE_ACTION_MASK[action_type][1] != "tp":
                self.malmo_client.send_command(
                    player_index, MbagAction.MOVE_ACTION_MASK[action_type][1]
                )
            else:
                # TODO: Which player location goes here?
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, player_location)),
                )

    def handle_place_break(self, player_index, action):
        # TODO: Fix player location and inventory no states
        # TODO: Also figure out the click location thing (maybe some of that logic
        # should be pulled out here because it's malmo specific)
        with self.malmo_lock:
            if self.config["abilities"]["teleportation"]:
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, player_location)),
                )

            viewpoint = np.array(player_location)
            viewpoint[1] += 1.6
            delta = np.array(click_location) - viewpoint
            delta /= np.sqrt((delta**2).sum())
            yaw = np.rad2deg(np.arctan2(-delta[0], delta[2]))
            pitch = np.rad2deg(-np.arcsin(delta[1]))
            self.malmo_client.send_command(player_index, f"setYaw {yaw}")
            self.malmo_client.send_command(player_index, f"setPitch {pitch}")
            self.player_directions[player_index] = (yaw, pitch)

            if action.action_type == MbagAction.PLACE_BLOCK:
                if self.config["abilities"]["inf_blocks"]:
                    self.malmo_client.send_command(
                        player_index,
                        f"swapInventoryItems 0 {action.block_id}",
                    )
                    hotbar_slot = 0
                else:
                    player_inventory = self.player_inventories[player_index]
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
            else:
                time.sleep(0.1)  # Give time to teleport.
                self.malmo_client.send_command(player_index, "attack 1")

    def add_ai_action(self, player_index, action):
        with self.ai_action_lock:
            self.ai_actions_queue.push((player_index, action))

    def run_ai_actions(self):
        while True:
            player_index, action = -1, (-1, -1, -1)
            with self.ai_action_lock:
                if len(self.ai_actions.queue) > 0:
                    player_index, action = self.ai_action_queue.pop(0)

            if (
                action[0] == MbagAction.PLACE_BLOCK
                or action[0] == MbagAction.BREAK_BLOCK
            ):
                self.handle_place_break(player_index, action)
            elif action[0] in MbagAction.MOVE_ACTION_TYPES:
                self.handle_move(player_index, action)

            # TODO: I assume we want to let the thread just yield here?
            time.sleep(0)


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
