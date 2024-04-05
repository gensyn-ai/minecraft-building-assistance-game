
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "action_distributions.h"
#include "blocks.h"
#include "constants.h"


static bool _is_valid_position_to_move_to(int x, int y_feet, int z, int width, int height, int depth, PyArrayObject *world_obs_array) {
    if (x >= 0 && x < width && y_feet >= 0 && y_feet < height && z >= 0 && z < depth) {
        int y_head = fmin(y_feet + 1, height - 1);
        // New space needs to both not have a block and not have a different player.
        return (
            *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y_feet, z) == AIR
            && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y_head, z) == AIR
            && (
                *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y_feet, z) == NO_ONE
                || *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y_feet, z) == CURRENT_PLAYER
            )
            && (
                *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y_head, z) == NO_ONE
                || *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y_head, z) == CURRENT_PLAYER
            )
        );
    } else {
        return false;
    }
}


npy_uint8 get_block_from_world_obs(void* world_obs_array, int x, int y, int z) {
    return *(npy_uint8*)PyArray_GETPTR4((PyArrayObject*)world_obs_array, CURRENT_BLOCKS, x, y, z);
}

bool is_player_from_world_obs(void* world_obs_array, int x, int y, int z) {
    npy_uint8 player_marker = *(npy_uint8*)PyArray_GETPTR4((PyArrayObject*)world_obs_array, PLAYER_LOCATIONS, x, y, z);
    return !(player_marker == NO_ONE || player_marker == CURRENT_PLAYER);
}


// get_action_distribution_mask(world_obs, inventory_obs, timestep, teleportation, inf_blocks)
PyObject* _mbag_get_action_distribution_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
    import_array();

    // Arguments
    PyArrayObject *world_obs_array;
    PyArrayObject *inventory_obs_array;
    int timestep;
    int teleportation, inf_blocks, line_of_sight_masking = false;

    // Other variables
    int i, x, y, z, block_id;
    bool have_block, in_reach;
    int num_viewpoint_click_candidates;
    double *viewpoint_click_candidates;

    static char *kwlist[] = {
        "world_obs",
        "inventory_obs",
        "timestep",
        "teleportation",
        "inf_blocks",
        "line_of_sight_masking",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OOipp|p",
        kwlist,
        &world_obs_array,
        &inventory_obs_array,
        &timestep,
        &teleportation,
        &inf_blocks,
        &line_of_sight_masking
    ))
        return NULL;

    if (!PyArray_Check(world_obs_array)) {
        PyErr_SetString(PyExc_TypeError, "world_obs must be an array");
        return NULL;
    } else if (PyArray_NDIM(world_obs_array) != 4) {
        PyErr_SetString(PyExc_TypeError, "world_obs must be a 4d array");
        return NULL;
    } else if (PyArray_TYPE(world_obs_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "world_obs must be an array of dtype uint8");
        return NULL;
    }

    if (!PyArray_Check(inventory_obs_array)) {
        PyErr_SetString(PyExc_TypeError, "inventory_obs must be an array");
        return NULL;
    } else if (PyArray_NDIM(inventory_obs_array) != 2) {
        PyErr_SetString(PyExc_TypeError, "inventory_obs must be a 2d array");
        return NULL;
    } else if (PyArray_TYPE(inventory_obs_array) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "inventory_obs must be an array of dtype int32");
        return NULL;
    }

    const int width = PyArray_DIMS(world_obs_array)[1];
    const int height = PyArray_DIMS(world_obs_array)[2];
    const int depth = PyArray_DIMS(world_obs_array)[3];

    npy_intp dims[] = {NUM_CHANNELS, width, height, depth};
    PyArrayObject* valid_array = (PyArrayObject*) PyArray_SimpleNew(4, dims, NPY_BOOL);
    PyArray_FILLWBYTE(valid_array, 0);

    // Find player location
    int player_x = -1, feet_y = -1, player_z = -1;
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            for (z = 0; z < depth; z++) {
                if (*(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y, z) == CURRENT_PLAYER) {
                    player_x = x;
                    if (feet_y == -1 || y < feet_y) {
                        // Get the minimum y to find the player's feet.
                        feet_y = y;
                    }
                    player_z = z;
                }
            }
        }
    }
    if (player_x == -1 && !teleportation) {
        PyErr_SetString(PyExc_RuntimeError, "No player location found");
        return NULL;
    }
    int head_y = feet_y + 1;

    /* NOOP and move actions */
    bool valid_pos_x, valid_neg_x, valid_pos_y, valid_neg_y, valid_pos_z, valid_neg_z;
    if (teleportation) {
        valid_pos_x = false;
        valid_neg_x = false;
        valid_pos_y = false;
        valid_neg_y = false;
        valid_pos_z = false;
        valid_neg_z = false;
    } else {
        valid_pos_x = _is_valid_position_to_move_to(player_x + 1, feet_y, player_z, width, height, depth, world_obs_array);
        valid_neg_x = _is_valid_position_to_move_to(player_x - 1, feet_y, player_z, width, height, depth, world_obs_array);
        valid_pos_y = _is_valid_position_to_move_to(player_x, feet_y + 1, player_z, width, height, depth, world_obs_array);
        valid_neg_y = _is_valid_position_to_move_to(player_x, feet_y - 1, player_z, width, height, depth, world_obs_array);
        valid_pos_z = _is_valid_position_to_move_to(player_x, feet_y, player_z + 1, width, height, depth, world_obs_array);
        valid_neg_z = _is_valid_position_to_move_to(player_x, feet_y, player_z - 1, width, height, depth, world_obs_array);
    }
    int channels[] = {NOOP_CHANNEL, MOVE_POS_X_CHANNEL, MOVE_NEG_X_CHANNEL, MOVE_POS_Y_CHANNEL, MOVE_NEG_Y_CHANNEL, MOVE_POS_Z_CHANNEL, MOVE_NEG_Z_CHANNEL};
    bool valids[] = {true, valid_pos_x, valid_neg_x, valid_pos_y, valid_neg_y, valid_pos_z, valid_neg_z};
    for (i = 0; i < (int) (sizeof(channels) / sizeof(channels[0])); i++) {
        int channel = channels[i];
        bool valid = valids[i];
        for (x = 0; x < width; x++) {
            for (y = 0; y < height; y++) {
                for (z = 0; z < depth; z++) {
                    *(npy_bool*)PyArray_GETPTR4(valid_array, channel, x, y, z) = valid;
                }
            }
        }
    }

    /* PLACE_BLOCK actions */
    int min_place_x, max_place_x, min_place_y, max_place_y, min_place_z, max_place_z;
    if (teleportation) {
        min_place_x = 0;
        max_place_x = width - 1;
        min_place_y = 0;
        max_place_y = height - 1;
        min_place_z = 0;
        max_place_z = depth - 1;
    } else {
        min_place_x = fmax(0, floor(player_x - MAX_PLAYER_REACH));
        max_place_x = fmin(width - 1, ceil(player_x + MAX_PLAYER_REACH));
        min_place_y = fmax(0, floor(head_y - MAX_PLAYER_REACH));
        max_place_y = fmin(height - 1, ceil(head_y + MAX_PLAYER_REACH));
        min_place_z = fmax(0, floor(player_z - MAX_PLAYER_REACH));
        max_place_z = fmin(depth - 1, ceil(player_z + MAX_PLAYER_REACH));
    }

    for (block_id = 0; block_id < NUM_BLOCKS; block_id++) {
        if (inf_blocks) {
            have_block = block_id != AIR && block_id != BEDROCK;
        } else {
            have_block = *(npy_int32*)PyArray_GETPTR2(inventory_obs_array, 0, block_id) > 0;
        }
        if (!have_block) continue;
        for (x = min_place_x; x <= max_place_x; x++) {
            for (y = min_place_y; y <= max_place_y; y++) {
                for (z = min_place_z; z <= max_place_z; z++) {
                    // To be placeable, the block needs to be in the player's reach,
                    // the space needs to be empty, and there needs to be an
                    // adjacent solid block to place against.
                    bool in_reach = teleportation || (
                        (x - player_x) * (x - player_x)
                        + (y - head_y) * (y - head_y)
                        + (z - player_z) * (z - player_z)
                        <= MAX_PLAYER_REACH * MAX_PLAYER_REACH
                    );
                    bool empty_space = (
                        *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y, z) == AIR
                        && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y, z) == NO_ONE
                    );
                    bool adjacent_solid = (
                        (x > 0 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x - 1, y, z) != AIR)
                        || (x < width - 1 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x + 1, y, z) != AIR)
                        || (y > 0 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y - 1, z) != AIR)
                        || (y < height - 1 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y + 1, z) != AIR)
                        || (z > 0 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y, z - 1) != AIR)
                        || (z < depth - 1 && *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y, z + 1) != AIR)
                    );

                    if (line_of_sight_masking && in_reach && empty_space && adjacent_solid) {
                        viewpoint_click_candidates = get_viewpoint_click_candidates(
                            PLACE_BLOCK,
                            width,
                            height,
                            depth,
                            x,
                            y,
                            z,
                            teleportation ? NAN : player_x + 0.5,
                            teleportation ? NAN : feet_y,
                            teleportation ? NAN : player_z + 0.5,
                            world_obs_array,
                            get_block_from_world_obs,
                            world_obs_array,
                            is_player_from_world_obs,
                            &num_viewpoint_click_candidates
                        );
                        if (viewpoint_click_candidates == NULL) {
                            PyErr_SetString(PyExc_RuntimeError, "Error getting viewpoint click candidates");
                            return NULL;
                        } else {
                            free(viewpoint_click_candidates);
                        }
                        in_reach = in_reach && (num_viewpoint_click_candidates > 0);
                    }

                    *(npy_bool*)PyArray_GETPTR4(valid_array, PLACE_BLOCK_CHANNEL + block_id, x, y, z) = in_reach && empty_space && adjacent_solid;
                }
            }
        }
    }

    /* BREAK_BLOCK actions */
    for (x = min_place_x; x <= max_place_x; x++) {
        for (y = min_place_y; y <= max_place_y; y++) {
            for (z = min_place_z; z <= max_place_z; z++) {
                npy_uint8 block = *(npy_uint8*)PyArray_GETPTR4(world_obs_array, CURRENT_BLOCKS, x, y, z);
                if (block == AIR || block == BEDROCK) {
                    *(npy_bool*)PyArray_GETPTR4(valid_array, BREAK_BLOCK_CHANNEL, x, y, z) = false;
                } else {
                    in_reach = teleportation || (
                        (x - player_x) * (x - player_x)
                        + (y - head_y) * (y - head_y)
                        + (z - player_z) * (z - player_z)
                        <= MAX_PLAYER_REACH * MAX_PLAYER_REACH
                    );

                    if (line_of_sight_masking && in_reach) {
                        viewpoint_click_candidates = get_viewpoint_click_candidates(
                            BREAK_BLOCK,
                            width,
                            height,
                            depth,
                            x,
                            y,
                            z,
                            teleportation ? NAN : player_x + 0.5,
                            teleportation ? NAN : feet_y,
                            teleportation ? NAN : player_z + 0.5,
                            world_obs_array,
                            get_block_from_world_obs,
                            world_obs_array,
                            is_player_from_world_obs,
                            &num_viewpoint_click_candidates
                        );
                        if (viewpoint_click_candidates == NULL) {
                            PyErr_SetString(PyExc_RuntimeError, "Error getting viewpoint click candidates");
                            return NULL;
                        } else {
                            free(viewpoint_click_candidates);
                        }
                        in_reach = in_reach && (num_viewpoint_click_candidates > 0);
                    }

                    *(npy_bool*)PyArray_GETPTR4(valid_array, BREAK_BLOCK_CHANNEL, x, y, z) = in_reach;
                }
            }
        }
    }

    /* GIVE_BLOCK actions */
    if (!inf_blocks) {
        int min_give_x, max_give_x, min_give_y, max_give_y, min_give_z, max_give_z;
        if (teleportation) {
            min_give_x = 0;
            max_give_x = width - 1;
            min_give_y = 0;
            max_give_y = height - 1;
            min_give_z = 0;
            max_give_z = depth - 1;
        } else {
            min_give_x = fmax(0, floor(player_x - 1));
            max_give_x = fmin(width - 1, ceil(player_x + 1));
            min_give_y = fmax(0, floor(head_y - 1));
            max_give_y = fmin(height - 1, ceil(head_y + 1));
            min_give_z = fmax(0, floor(player_z - 1));
            max_give_z = fmin(depth - 1, ceil(player_z + 1));
        }
        for (block_id = 0; block_id < NUM_BLOCKS; block_id++) {
            have_block = *(npy_int32*)PyArray_GETPTR2(inventory_obs_array, 0, block_id) > 0;
            if (!have_block) continue;
            for (x = min_give_x; x <= max_give_x; x++) {
                for (y = min_give_y; y <= max_give_y; y++) {
                    for (z = min_give_z; z <= max_give_z; z++) {
                        // Check if there is a player there.
                        npy_uint8 player_marker = *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y, z);
                        bool is_player;
                        if (player_marker == NO_ONE || player_marker == CURRENT_PLAYER) {
                            is_player = false;
                        } else {
                            // The give location should be the feet of the other player,
                            // which means that the block below should not have the
                            // same player marker (otherwise we've found the head).
                            is_player = (
                                y - 1 < 0
                                || *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y - 1, z) != player_marker
                            );
                        }
                        *(npy_bool*)PyArray_GETPTR4(valid_array, GIVE_BLOCK_CHANNEL + block_id, x, y, z) = is_player;
                    }
                }
            }
        }
    }

    return PyArray_Return(valid_array);
}
