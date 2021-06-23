"""
Code to interface with Project Malmo.
"""

from typing import List, Optional, TypedDict, cast
import MalmoPython
import logging
import time
import uuid
import json
import numpy as np

from .types import BlockLocation
from .blocks import MinecraftBlocks
from .mbag_env import MbagConfigDict

logger = logging.getLogger(__name__)


class MalmoObservationDict(TypedDict, total=False):
    world: List[str]
    goal: List[str]


class MalmoClient(object):
    ACTION_DELAY = 0.3

    agent_hosts: List[MalmoPython.AgentHost]
    experiment_id: str

    def __init__(self):
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool_size = 0

    def _get_agent_section_xml(self, player_index: int, env_config: MbagConfigDict):
        width, height, depth = env_config["world_size"]

        inventory_item_tags: List[str] = []
        for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
            block_name = MinecraftBlocks.ID2NAME[block_id]
            inventory_item_tags.append(
                f"""
                <InventoryItem slot="{block_id}" type="{block_name}" />
                """
            )
        inventory_items_xml = "\n".join(inventory_item_tags)

        return f"""
        <AgentSection mode="Creative">
            <Name>player_{player_index}</Name>
            <AgentStart>
                <Placement x="{0.5 + player_index}" y="2" z="0.5" yaw="90"/>
                <Inventory>
                    {inventory_items_xml}
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromGrid>
                    <Grid name="world" absoluteCoords="true">
                        <min x="0" y="0" z="0" />
                        <max x="{width - 1}" y="{height - 1}" z="{depth - 1}" />
                    </Grid>
                    <Grid name="goal" absoluteCoords="true">
                        <min x="{width + 1}" y="0" z="0" />
                        <max x="{width * 2}" y="{height - 1}" z="{depth - 1}" />
                    </Grid>
                </ObservationFromGrid>
                <ObservationFromFullInventory />
                <AbsoluteMovementCommands />
                <DiscreteMovementCommands />
                <InventoryCommands />
                <MissionQuitCommands />
            </AgentHandlers>
        </AgentSection>
        """

    def _blocks_to_drawing_decorator_xml(
        self, blocks: MinecraftBlocks, offset: BlockLocation = (0, 0, 0)
    ) -> str:
        draw_tags: List[str] = []
        for (x, y, z), block_id in np.ndenumerate(blocks.blocks):
            block_name = MinecraftBlocks.ID2NAME[block_id]
            if (
                block_name == "air"
                or block_name == "bedrock"
                and y == 0
                or block_name == "dirt"
                and y == 1
            ):
                continue

            block_state = blocks.block_states[x, y, z]  # noqa, TODO: use this
            draw_tags.append(
                f"""
                <DrawBlock
                    type="{block_name}"
                    x="{x + offset[0]}"
                    y="{y + offset[1]}"
                    z="{z + offset[2]}"
                />
                """
            )
        return "\n".join(draw_tags)

    def _get_mission_spec_xml(
        self,
        env_config: MbagConfigDict,
        goal_blocks: MinecraftBlocks,
        force_reset: bool = True,
    ) -> str:
        width, height, depth = env_config["world_size"]
        force_reset_str = "true" if force_reset else "false"

        agent_sections_xml = "\n".join(
            self._get_agent_section_xml(player_index, env_config)
            for player_index in range(env_config["num_players"])
        )

        return f"""
        <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

            <About>
                <Summary>Minecraft Building Assistance Game</Summary>
            </About>

            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator
                        forceReset="{force_reset_str}"
                        generatorString="3;1*minecraft:bedrock,1*minecraft:grass;minecraft:plains;"
                        destroyAfterUse="true"
                    />
                    <DrawingDecorator>
                        {self._blocks_to_drawing_decorator_xml(goal_blocks, (width + 1, 0, 0))}
                    </DrawingDecorator>
                    <BuildBattleDecorator>
                        <PlayerStructureBounds>
                            <min x="0" y="0" z="0" />
                            <max x="{width - 1}" y="{height - 1}" z="{depth - 1}" />
                        </PlayerStructureBounds>
                        <GoalStructureBounds>
                            <min x="{width + 1}" y="0" z="0" />
                            <max x="{width * 2}" y="{height - 1}" z="{depth - 1}" />
                        </GoalStructureBounds>
                    </BuildBattleDecorator>
                </ServerHandlers>
            </ServerSection>

            {agent_sections_xml}
        </Mission>
        """

    def _expand_client_pool(self, num_clients):
        while self.client_pool_size < num_clients:
            self.client_pool.add(
                MalmoPython.ClientInfo("127.0.0.1", self.client_pool_size + 10000)
            )
            self.client_pool_size += 1

    # This method based on code from multi_agent_test.py in the Project Malmo examples.
    def _safe_start_mission(
        self,
        agent_host: MalmoPython.AgentHost,
        mission: MalmoPython.MissionSpec,
        mission_record: MalmoPython.MissionRecordSpec,
        player_index: int,
        max_attempts: int = 5,
    ):
        used_attempts = 0
        logger.info(f"starting Malmo mission for player {player_index}")
        while True:
            try:
                # Attempt start:
                agent_host.startMission(
                    mission,
                    self.client_pool,
                    mission_record,
                    player_index,
                    self.experiment_id,
                )
                break
            except MalmoPython.MissionException as error:
                error_code = error.details.errorCode
                if error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                    logger.info("server not quite ready yet, waiting...")
                    time.sleep(2)
                elif (
                    error_code
                    == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE
                ):
                    logger.warning("not enough available Minecraft instances running")
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        logger.info(
                            "will wait in case they are starting up; "
                            f"{max_attempts - used_attempts} attempts left",
                        )
                        time.sleep(2)
                elif (
                    error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND
                ):
                    logger.warning(
                        "server not found; has the mission with role 0 been started yet?"
                    )
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        logger.info(
                            "will wait and retry; "
                            f"{max_attempts - used_attempts} attempts left",
                        )
                        time.sleep(2)
                else:
                    raise error
                if used_attempts == max_attempts:
                    logger.error(
                        f"failed to start mission after {max_attempts} attempts"
                    )
                    raise error
        logger.info(f"Malmo mission successfully started for player {player_index}")

    # This method based on code from multi_agent_test.py in the Project Malmo examples.
    def _safe_wait_for_start(self, agent_hosts: List[MalmoPython.AgentHost]):
        logger.info("waiting for the mission to start")
        agent_hosts_started = [False for _ in agent_hosts]
        start_time = time.time()
        time_out = 120  # Allow a two minute timeout.
        while not all(agent_hosts_started) and time.time() - start_time < time_out:
            states = [agent_host.peekWorldState() for agent_host in agent_hosts]
            agent_hosts_started = [state.has_mission_begun for state in states]
            errors = [error for state in states for error in state.errors]
            if len(errors) > 0:
                logger.error("errors waiting for mission start:")
                for e in errors:
                    logger.error(e.text)
                raise errors[0]
            time.sleep(0.1)
        if time.time() - start_time >= time_out:
            logger.error("timed out while waiting for mission to start")
            raise RuntimeError("timed out while waiting for mission to start")

    def start_mission(self, env_config: MbagConfigDict, goal_blocks: MinecraftBlocks):
        self._expand_client_pool(env_config["num_players"])
        self.experiment_id = str(uuid.uuid4())

        self.agent_hosts = []
        for player_index in range(env_config["num_players"]):
            agent_host = MalmoPython.AgentHost()
            self.agent_hosts.append(agent_host)
            mission_spec_xml = self._get_mission_spec_xml(
                env_config, goal_blocks, force_reset=player_index == 0
            )
            self._safe_start_mission(
                agent_host,
                MalmoPython.MissionSpec(mission_spec_xml, True),
                MalmoPython.MissionRecordSpec(),
                player_index,
            )

        self._safe_wait_for_start(self.agent_hosts)

    def send_command(self, player_index: int, command: str):
        self.agent_hosts[player_index].sendCommand(command)

    def get_observation(self, player_index: int) -> Optional[MalmoObservationDict]:
        agent_host = self.agent_hosts[player_index]
        world_state = agent_host.getWorldState()
        if not world_state.is_mission_running:
            return None
        elif (
            world_state.is_mission_running
            and world_state.number_of_observations_since_last_state > 0
        ):
            return cast(
                MalmoObservationDict, json.loads(world_state.observations[-1].text)
            )
        else:
            return None
