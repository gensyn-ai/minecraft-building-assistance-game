"""
Various GoalTransforms which alter a goal.
"""

import logging
import math
import random
from typing import List, Optional, Tuple, TypedDict, cast

import cc3d
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.util import view_as_blocks
from typing_extensions import Literal

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_transform import GoalTransform

logger = logging.getLogger(__name__)


class RandomlyPlaceTransform(GoalTransform):
    """
    Given a structure of size smaller than the given size, randomly places the
    structure along the x and z axes (but always keeps it at the same height on
    the y axis).
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        structure = self.goal_generator.generate_goal(size)
        blocks = MinecraftBlocks(size)

        offset_x = random.randint(0, size[0] - structure.size[0])
        offset_z = random.randint(0, size[2] - structure.size[2])
        structure_slice = (
            slice(offset_x, offset_x + structure.size[0]),
            slice(0, structure.size[1]),
            slice(offset_z, offset_z + structure.size[2]),
        )
        blocks.blocks[structure_slice] = structure.blocks
        blocks.block_states[structure_slice] = structure.block_states

        return blocks


AddGrassMode = Literal["surround", "replace", "concatenate"]


class AddGrassTransformConfig(TypedDict):
    mode: AddGrassMode
    """
    Controls how to add grass:
        "surround" means replace all air blocks on the bottom layer with grass.
        "replace" means to replace the bottom layer with grass.
        "concatenate" means generate a 1-block shorter structure and then add a grass
        layer below it.
    """


class AddGrassTransform(GoalTransform):
    """
    Adds grass (dirt) to the bottom layer of a structure where there aren't yet
    blocks.
    """

    default_config: AddGrassTransformConfig = {"mode": "surround"}
    config: AddGrassTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        mode = self.config["mode"]
        if mode == "concatenate":
            width, height, depth = size
            smaller_goal = self.goal_generator.generate_goal((width, height - 1, depth))
            goal = MinecraftBlocks(
                (smaller_goal.size[0], smaller_goal.size[1] + 1, smaller_goal.size[2])
            )
            goal.blocks[:, 1:, :] = smaller_goal.blocks
            goal.block_states[:, 1:, :] = smaller_goal.block_states
            goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
        elif mode == "replace":
            goal = self.goal_generator.generate_goal(size)
            goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
        elif mode == "surround":
            goal = self.goal_generator.generate_goal(size)
            bottom_layer = goal.blocks[:, 0, :]
            bottom_layer[
                bottom_layer == MinecraftBlocks.NAME2ID["air"]
            ] = MinecraftBlocks.NAME2ID["grass"]

        return goal


class CropTransformConfig(TypedDict):
    density_threshold: float
    tethered_to_ground: bool
    wall: bool


class CropTransform(GoalTransform):
    """
    Crops large structures down to a smaller size. The crop is chosen such that the
    density difference between the original and cropped structures is no more than
    density_threshold as measured by percentage. For instance, if density_threshold is
    0.25, then the crop density will be 75-125% as dense as the original structure.
    If tethered_to_ground is True, then the crop will be taken from the bottom of the
    structure.
    """

    default_config: CropTransformConfig = {
        "density_threshold": 0.25,
        "tethered_to_ground": True,
        "wall": False,
    }
    config: CropTransformConfig

    def generate_goal(self, size: WorldSize, *, retries: int = 20) -> MinecraftBlocks:
        while True:
            # Generate a goal with effectively no size limits so we can crop it down.
            goal = self.goal_generator.generate_goal((100, 100, 100))
            struct_density = goal.density()
            if struct_density == 0:
                continue

            crop_size = (
                min(size[0], goal.size[0]),
                min(size[1], goal.size[1]),
                1 if self.config["wall"] else min(size[2], goal.size[2]),
            )

            x_range = goal.size[0] - crop_size[0]
            y_range = (
                0 if self.config["tethered_to_ground"] else goal.size[1] - crop_size[1]
            )
            z_range = goal.size[2] - crop_size[2]

            for _ in range(retries):
                crop = MinecraftBlocks(crop_size)
                crop.blocks[:] = MinecraftBlocks.AIR
                crop.block_states[:] = 0

                x, y, z = (
                    random.randint(0, x_range),
                    random.randint(0, y_range),
                    random.randint(0, z_range),
                )
                crop.fill_from_crop(goal, (x, y, z))

                if (
                    abs(crop.density() - struct_density) / struct_density
                    > self.config["density_threshold"]
                ):
                    continue

                return crop

            logger.info("CropTransform was unable to find a valid crop")


class AreaSampleTransformConfig(TypedDict):
    max_scaling_factor: float
    """Maximum factor by which goals should be scaled."""

    interpolate: bool
    """Use interpolation to scale by factors that are not a power of two."""

    interpolation_order: int
    """The spline order used to interpolate, i.e., 0 = nearest neighbor, 1 = bilinear,
    3 = bicubic, etc."""

    scale_y_independently: bool
    """If interpolate is True, then this allows the Y dimension to be scaled
    independently of the X and Z dimensions."""

    max_scaling_factor_ratio: float
    """If scale_y_independently is True, this controls the maximum ratio between the
    scale factors for the Y and X/Z dimensions."""


class AreaSampleTransform(GoalTransform):
    default_config: AreaSampleTransformConfig = {
        "max_scaling_factor": 4.0,
        "interpolate": True,
        "interpolation_order": 3,
        "scale_y_independently": True,
        "max_scaling_factor_ratio": 1.5,
    }
    config: AreaSampleTransformConfig

    DOOR = 254

    def generate_goal(self, size: WorldSize, *, retries: int = 20) -> MinecraftBlocks:
        structure: Optional[MinecraftBlocks] = None

        while structure is None:
            structure = self.goal_generator.generate_goal((100, 100, 100))
            max_scale_down_size = (
                structure.size[0] / self.config["max_scaling_factor"],
                structure.size[1] / self.config["max_scaling_factor"],
                structure.size[2] / self.config["max_scaling_factor"],
            )

            if (
                max_scale_down_size[0] > size[0]
                or max_scale_down_size[1] > size[1]
                or max_scale_down_size[2] > size[2]
            ):
                structure = None

        return self.scale_down_structure(structure, size)

    def scale_down_structure(
        self, structure: Optional[MinecraftBlocks], size: WorldSize
    ) -> MinecraftBlocks:
        assert structure is not None, "Must pass in a valid structure to scale down"

        logger.info(f"original structure size = {structure.size}")

        if all(structure.size[axis] <= size[axis] for axis in range(3)):
            return structure

        doors = AreaSampleTransform._find_doors(structure)
        structure.blocks[doors] = AreaSampleTransform.DOOR

        if self.config["interpolate"]:
            scaling_iterations = int(
                np.ceil(
                    np.log2(max(structure.size[axis] / size[axis] for axis in range(3)))
                )
            )
            assert scaling_iterations >= 1

            structure = self._interpolate_structure(
                structure,
                size,
                self._get_zoom(structure.size, size, scaling_iterations),
            )

        if False:
            # We directly zoom to get block types, but scale up and then scale down
            # to get the mask (where air should/shouldn't be).

            mask_zoom = self._get_zoom(structure.size, size, scaling_iterations)
            mask = structure.blocks != MinecraftBlocks.AIR
            mask = self._zoom_with_ids(mask.astype(np.uint8), mask_zoom) != 0
            print(f"{mask_zoom=} {mask.shape=}")
            while any(mask.shape[axis] > size[axis] for axis in range(3)):
                scaled_down_mask_size = tuple(
                    int(math.ceil(mask.shape[axis] / 2)) for axis in range(3)
                )
                padded_size = tuple(
                    scaled_down_mask_size[axis] * 2 for axis in range(3)
                )
                padded_mask = np.zeros_like(mask, shape=padded_size)
                padded_mask[: mask.shape[0], : mask.shape[1], : mask.shape[2]] = mask
                scaled_down_mask = (
                    view_as_blocks(padded_mask, (2, 2, 2))
                    .reshape((scaled_down_mask_size) + (8,))
                    .any(axis=-1)
                )
                mask = scaled_down_mask
                print(f"{mask.shape=}")

            block_zoom = (
                mask.shape[0] / structure.size[0],
                mask.shape[1] / structure.size[1],
                mask.shape[2] / structure.size[2],
            )
            interpolated_structure = self._interpolate_structure(
                structure, size, block_zoom, ignore_air=True
            )
            print(f"{block_zoom=} {interpolated_structure.size=}")

            interpolated_structure.blocks[~mask] = MinecraftBlocks.AIR
            return interpolated_structure

        if True:
            while (
                structure.size[0] > size[0]
                or structure.size[1] > size[1]
                or structure.size[2] > size[2]
            ):
                scaled_down_structure = MinecraftBlocks(
                    (
                        int(math.ceil(0.5 * structure.size[0])),
                        int(math.ceil(0.5 * structure.size[1])),
                        int(math.ceil(0.5 * structure.size[2])),
                    )
                )

                chunk_size = (2, 2, 2)
                logger.info("scaling down by 2x")

                idx = [
                    (i, j, k)
                    for i in range(scaled_down_structure.size[0])
                    for j in range(scaled_down_structure.size[1])
                    for k in range(scaled_down_structure.size[2])
                ]

                for i, chunk in enumerate(structure.get_chunks(chunk_size)):
                    index = idx[i]
                    scaled_down_structure.blocks[index] = self._most_common_block(
                        chunk, ignore_air=True
                    )

                structure = scaled_down_structure

            # Postprocess doors.
            for door_x, door_y, door_z in zip(
                *np.nonzero(structure.blocks == AreaSampleTransform.DOOR)
            ):
                structure.blocks[door_x, door_y, door_z] = MinecraftBlocks.AIR
                if door_y + 1 < structure.size[1]:
                    structure.blocks[door_x, door_y + 1, door_z] = MinecraftBlocks.AIR

            return structure

    def _most_common_block(self, array: np.ndarray, ignore_air=False) -> int:
        if np.any(array == AreaSampleTransform.DOOR):
            return AreaSampleTransform.DOOR

        mask = (array != 0) & (array != -1)
        if np.sum(mask) < array[(array != -1)].size / 2 and not ignore_air:
            return 0

        flat_arr = array.flatten()
        filtered_arr = flat_arr[(flat_arr != -1) & (flat_arr != 0)]
        counts = np.bincount(filtered_arr)

        try:
            ties = np.where(counts == counts[np.argmax(counts)])[0]
        except Exception:
            return 0

        return int(max(ties))

    def _zoom_with_ids(
        self,
        input: np.ndarray,
        zoom: Tuple[float, float, float],
        ignore_air=False,
        **kwargs,
    ) -> np.ndarray:
        all_ids = np.sort(np.unique(input))
        assert all_ids[0] == MinecraftBlocks.AIR
        zoomed_per_id_list: List[np.ndarray] = []
        for id_index, id in enumerate(all_ids):
            zoomed_per_id_list.append(
                ndimage.zoom((input == id).astype(float), zoom, **kwargs)
            )
        zoomed_per_id = np.stack(zoomed_per_id_list, axis=0)
        if ignore_air:
            zoomed_per_id[0] = -1
        return cast(np.ndarray, all_ids[zoomed_per_id.argmax(axis=0)])

    def _interpolate_structure(
        self,
        structure: MinecraftBlocks,
        size: WorldSize,
        zoom: Tuple[float, float, float],
        ignore_air: bool = False,
    ) -> MinecraftBlocks:
        blocks = self._zoom_with_ids(
            structure.blocks,
            zoom,
            order=self.config["interpolation_order"],
            ignore_air=ignore_air,
        )
        block_states = self._zoom_with_ids(
            structure.block_states, zoom, order=self.config["interpolation_order"]
        )

        interpolated_structure = MinecraftBlocks(cast(WorldSize, blocks.shape))
        interpolated_structure.blocks[...] = blocks
        interpolated_structure.block_states[...] = block_states
        return interpolated_structure

    def _get_zoom(
        self,
        structure_size: WorldSize,
        target_size: WorldSize,
        scaling_iterations: int,
    ) -> Tuple[float, float, float]:
        total_scaling = 2.0**scaling_iterations

        raw_zoom = [np.nan] * 3
        for axis in range(3):
            raw_zoom[axis] = (
                min(1, target_size[axis] / structure_size[axis]) * total_scaling
            )

        if self.config["scale_y_independently"]:
            xz_zoom = min(raw_zoom[0], raw_zoom[2])
            y_zoom = raw_zoom[1]

            y_zoom = min(y_zoom, xz_zoom * self.config["max_scaling_factor_ratio"])
            xz_zoom = min(xz_zoom, y_zoom * self.config["max_scaling_factor_ratio"])
            return xz_zoom, y_zoom, xz_zoom
        else:
            xyz_zoom = min(raw_zoom)
            return xyz_zoom, xyz_zoom, xyz_zoom

    @staticmethod
    def _find_doors(structure: MinecraftBlocks):
        width, height, depth = structure.size
        world = MinecraftBlocks((width + 2, height + 2, depth + 2))
        world.blocks[1:-1, :-2, 1:-1] = structure.blocks
        bottom_layer = world.blocks[:, 0, :]
        bottom_layer[bottom_layer != MinecraftBlocks] = 1
        can_stand = np.concatenate(
            [
                np.zeros_like(structure.blocks[:, :1, :], dtype=bool),
                (structure.blocks != MinecraftBlocks.AIR)[:, :-1, :],
            ],
            axis=1,
        )
        can_stand = np.zeros_like(world.blocks, dtype=bool)
        can_stand[:, 1:-1, :] = (
            (world.blocks != MinecraftBlocks.AIR)[:, :-2, :]
            & (world.blocks == MinecraftBlocks.AIR)[:, 1:-1]
            & (world.blocks == MinecraftBlocks.AIR)[:, 2:, :]
        )
        can_stand[:, 2:-1, :] |= (
            (world.blocks != MinecraftBlocks.AIR)[:, :-3, :]
            & (world.blocks == MinecraftBlocks.AIR)[:, 1:-2]
            & (world.blocks == MinecraftBlocks.AIR)[:, 2:-1]
            & (world.blocks == MinecraftBlocks.AIR)[:, 3:, :]
        )

        component_ids = cc3d.connected_components(can_stand, connectivity=6)
        ground_component_id = component_ids[0, 1, 0]
        ground_component = (component_ids == ground_component_id)[:, :-2, :]

        xx, yy, zz = np.meshgrid(
            np.arange(width + 2), np.arange(height), np.arange(depth + 2), indexing="ij"
        )
        numel = (width + 2) * height * (depth + 2)
        xx1, yy1, zz1 = xx.reshape(numel, 1), yy.reshape(numel, 1), zz.reshape(numel, 1)
        xx2, yy2, zz2 = xx.reshape(1, numel), yy.reshape(1, numel), zz.reshape(1, numel)
        adj_matrix = np.zeros((numel, numel), dtype=bool)
        adj_matrix |= (np.abs(xx1 - xx2) == 1) & (yy1 == yy2) & (zz1 == zz2)
        adj_matrix |= (xx1 == xx2) & (np.abs(yy1 - yy2) == 1) & (zz1 == zz2)
        adj_matrix |= (xx1 == xx2) & (yy1 == yy2) & (np.abs(zz1 - zz2) == 1)
        adj_matrix[~ground_component.reshape(numel), :] = False
        adj_matrix[:, ~ground_component.reshape(numel)] = False

        walkable_graph = nx.Graph(
            adj_matrix,
        )
        walkable_graph.remove_nodes_from(
            [
                node
                for node, degree in dict(walkable_graph.degree()).items()
                if degree == 0
            ]
        )

        centrality_dict = nx.betweenness_centrality(walkable_graph)
        centrality = np.zeros((width + 2, height, depth + 2))
        for node_id, node_centrality in centrality_dict.items():
            centrality.flat[node_id] = node_centrality

        centrality[0, :, :] = 0
        centrality[-1, :, :] = 0
        centrality[:, :, 0] = 0
        centrality[:, :, -1] = 0

        local_max_id = np.arange(numel)
        local_max_val = np.zeros(numel)
        local_max_val[:] = centrality.flat
        prev_local_max_id = np.zeros_like(local_max_id)
        while np.any(local_max_id != prev_local_max_id):
            prev_local_max_id = local_max_id
            possible_max_vals = np.zeros((numel, numel))
            possible_max_vals[...] = centrality.reshape(numel)[None, :]
            possible_max_vals[~(adj_matrix | np.eye(numel, dtype=bool))] = 0
            new_max_val_indices = possible_max_vals.argmax(axis=1)
            local_max_id = local_max_id[new_max_val_indices]
            local_max_val = local_max_val[new_max_val_indices]

        local_max_val = local_max_val.reshape((width + 2, height, depth + 2))
        is_local_max = (local_max_id == np.arange(numel)).reshape(
            (width + 2, height, depth + 2)
        ) & ground_component

        doors = np.zeros_like(centrality, dtype=bool)
        for node_id in np.nonzero((is_local_max & (centrality >= 0.2)).flat)[0]:
            node_id = int(node_id)
            local_graph = nx.generators.ego_graph(walkable_graph, node_id, radius=2)
            local_node_ids = [
                local_node_id
                for local_node_id in local_graph.nodes.keys()
                if not doors.flat[local_node_id]
            ]
            local_node_ids.sort(
                key=lambda local_node_id: centrality_dict[local_node_id], reverse=True
            )
            for node_id in local_node_ids[:3]:
                doors.flat[node_id] = True

        return doors[1:-1, :, 1:-1]


class SeamCarvingTransformConfig(TypedDict):
    position_coefficient: float
    density_coefficient: float
    max_original_size: WorldSize


class SeamCarvingTransform(GoalTransform):
    """
    Turns larger goals into smaller ones by removing slices strategically so as
    to maintain the structure of the goal while making it smaller.
    """

    default_config: SeamCarvingTransformConfig = {
        "position_coefficient": 1,
        "density_coefficient": 1,
        "max_original_size": (100, 100, 100),
    }
    config: SeamCarvingTransformConfig

    def _get_relative_positions(self, size: WorldSize) -> np.ndarray:
        return np.stack(
            np.meshgrid(*[np.linspace(0, 1, size[axis]) for axis in range(3)]),
            axis=-1,
        ).transpose((1, 0, 2, 3))

    def _slice(self, axis: int, index: int, arr: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.delete(arr, [index], axis=axis))

    def _slice_cost(
        self,
        axis: int,
        index: int,
        goal: MinecraftBlocks,
        original_positions: np.ndarray,
    ) -> float:
        density = float(
            np.take(goal.blocks != MinecraftBlocks.AIR, np.array([index]), axis=axis)
            .astype(float)
            .mean()
        )

        position_mse_before = float(
            np.sqrt(
                np.mean(
                    (original_positions - self._get_relative_positions(goal.size)) ** 2
                )
            )
        )
        original_positions_sliced = self._slice(axis, index, original_positions)
        position_mse_after = float(
            np.sqrt(
                np.mean(
                    (
                        original_positions_sliced
                        - self._get_relative_positions(
                            cast(WorldSize, original_positions_sliced.shape)
                        )
                    )
                    ** 2
                )
            )
        )
        position_mse_delta = position_mse_after - position_mse_before

        cost = (
            self.config["density_coefficient"] * density
            + self.config["position_coefficient"] * position_mse_delta
        )
        logger.debug(
            f"axis {axis} slice {index:02d}: "
            f"cost={cost:.2f}\t"
            f"density={density:.2f} position_mse_delta={position_mse_delta:.2f}"
        )
        return cost

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        # Generate a goal with effectively no size limits so we can slice it down.
        goal = self.goal_generator.generate_goal(self.config["max_original_size"])
        original_positions = self._get_relative_positions(goal.size)

        while True:
            too_big_axes = [axis for axis in range(3) if goal.size[axis] > size[axis]]
            if len(too_big_axes) == 0:
                break

            best_slice: Tuple[int, int] = (0, 0)
            best_cost: float = np.inf
            for axis in too_big_axes:
                for index in range(goal.size[axis]):
                    cost = self._slice_cost(axis, index, goal, original_positions)
                    if cost < best_cost:
                        best_cost = cost
                        best_slice = (axis, index)
            best_axis, best_index = best_slice
            goal.blocks = self._slice(best_axis, best_index, goal.blocks)
            goal.block_states = self._slice(best_axis, best_index, goal.block_states)
            goal.size = cast(WorldSize, goal.blocks.shape)
            original_positions = self._slice(best_axis, best_index, original_positions)

        return goal


class UniformBlockTypeTransformConfig(TypedDict):
    block_type: int


class UniformBlockTypeTransform(GoalTransform):
    """
    Modify structure so that every non-air block is the same specified block type.
    """

    default_config: UniformBlockTypeTransformConfig = {
        "block_type": MinecraftBlocks.NAME2ID["grass"],
    }
    config: UniformBlockTypeTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        goal.blocks[goal.blocks != MinecraftBlocks.AIR] = self.config["block_type"]
        return goal


class MirrorTransform(GoalTransform):
    """
    Mirrors a structure so it's symmetric along the X axis.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        goal.mirror_x_axis()
        return goal
