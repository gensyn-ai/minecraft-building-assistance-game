
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/*
    A bunch of constants. These are checked against the corresponding Python ones when
    the C module is loaded.
*/
#define NUM_ACTION_TYPES 10
#define NUM_BLOCKS 10
#define NUM_CHANNELS (NUM_ACTION_TYPES + 2 * (NUM_BLOCKS - 1))
#define NOOP_CHANNEL 0
#define PLACE_BLOCK_CHANNEL 1
#define BREAK_BLOCK_CHANNEL (1 + NUM_BLOCKS)
#define MOVE_POS_X_CHANNEL (2 + NUM_BLOCKS)
#define MOVE_NEG_X_CHANNEL (3 + NUM_BLOCKS)
#define MOVE_POS_Y_CHANNEL (4 + NUM_BLOCKS)
#define MOVE_NEG_Y_CHANNEL (5 + NUM_BLOCKS)
#define MOVE_POS_Z_CHANNEL (6 + NUM_BLOCKS)
#define MOVE_NEG_Z_CHANNEL (7 + NUM_BLOCKS)
#define GIVE_BLOCK_CHANNEL (8 + NUM_BLOCKS)
#define NOOP 0
#define PLACE_BLOCK 1
#define BREAK_BLOCK 2
#define MOVE_POS_X 3
#define MOVE_NEG_X 4
#define MOVE_POS_Y 5
#define MOVE_NEG_Y 6
#define MOVE_POS_Z 7
#define MOVE_NEG_Z 8
#define GIVE_BLOCK 9
#define CURRENT_BLOCKS 0
#define PLAYER_LOCATIONS 4
#define NO_ONE 0
#define CURRENT_PLAYER 1
#define AIR 0
#define BEDROCK 1
#define MAX_PLAYER_REACH 4.5


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


// get_mask(world_obs, inventory_obs, timestep, teleportation, inf_blocks)
static PyObject *
_mbag_action_distributions_get_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // Arguments
    PyArrayObject *world_obs_array;
    PyArrayObject *inventory_obs_array;
    int timestep;
    bool teleportation, inf_blocks;

    // Other variables
    int i, x, y, z, block_id;
    bool have_block, in_reach;

    static char *kwlist[] = {
        "world_obs", "inventory_obs", "timestep", "teleportation", "inf_blocks", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OOipp",
        kwlist,
        &world_obs_array,
        &inventory_obs_array,
        &timestep,
        &teleportation,
        &inf_blocks
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
    } else if (PyArray_NDIM(inventory_obs_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "inventory_obs must be a 1d array");
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
            have_block = *(npy_int32*)PyArray_GETPTR1(inventory_obs_array, block_id) > 0;
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
                    *(npy_bool*)PyArray_GETPTR4(valid_array, BREAK_BLOCK_CHANNEL, x, y, z) = in_reach;
                }
            }
        }
    }

    /* GIVE_BLOCK actions */
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
        if (inf_blocks) {
            // Can't give blocks when there are infinite blocks.
            have_block = false;
        } else {
            have_block = *(npy_int32*)PyArray_GETPTR1(inventory_obs_array, block_id) > 0;
        }
        if (!have_block) continue;
        for (x = min_give_x; x <= max_give_x; x++) {
            for (y = min_give_y; y <= max_give_y; y++) {
                for (z = min_give_z; z <= max_give_z; z++) {
                    // Check if there is a player there.
                    bool is_player = (
                        *(npy_uint8*)PyArray_GETPTR4(world_obs_array, PLAYER_LOCATIONS, x, y, z) != NO_ONE
                    );
                    *(npy_bool*)PyArray_GETPTR4(valid_array, GIVE_BLOCK_CHANNEL + block_id, x, y, z) = is_player;
                }
            }
        }
    }

    return valid_array;
}

static PyMethodDef ActionDistributionsMethods[] = {
    {"get_mask", (PyCFunction) _mbag_action_distributions_get_mask,
     METH_VARARGS | METH_KEYWORDS, "Get the action mask given an MBAG observation."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef _mbag_action_distributionsmodule = {
    PyModuleDef_HEAD_INIT,
    "_mbag_action_distributions",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    ActionDistributionsMethods
};

PyMODINIT_FUNC
PyInit__mbag_action_distributions(void)
{
    import_array();

    PyObject *blocks_module = PyImport_ImportModule("mbag.environment.blocks");
    double max_player_reach = PyFloat_AsDouble(PyObject_GetAttrString(blocks_module, "MAX_PLAYER_REACH"));
    if (abs(max_player_reach - MAX_PLAYER_REACH) > 1e-6) {
        PyErr_SetString(PyExc_RuntimeError, "MAX_PLAYER_REACH does not match the expected value");
        return NULL;
    }
    PyObject *MinecraftBlocks = PyObject_GetAttrString(blocks_module, "MinecraftBlocks");
    long num_blocks = PyLong_AsLong(PyObject_GetAttrString(MinecraftBlocks, "NUM_BLOCKS"));
    if (num_blocks != NUM_BLOCKS) {
        PyErr_SetString(PyExc_RuntimeError, "NUM_BLOCKS does not match the expected value");
        return NULL;
    }
    long air = PyLong_AsLong(PyObject_GetAttrString(MinecraftBlocks, "AIR"));
    if (air != AIR) {
        PyErr_SetString(PyExc_RuntimeError, "AIR does not match the expected value");
        return NULL;
    }
    long bedrock = PyLong_AsLong(PyObject_GetAttrString(MinecraftBlocks, "BEDROCK"));
    if (bedrock != BEDROCK) {
        PyErr_SetString(PyExc_RuntimeError, "BEDROCK does not match the expected value");
        return NULL;
    }

    PyObject *types_module = PyImport_ImportModule("mbag.environment.types");
    PyObject *MbagAction = PyObject_GetAttrString(types_module, "MbagAction");
    int num_action_types = PyLong_AsLong(PyObject_GetAttrString(MbagAction, "NUM_ACTION_TYPES"));
    if (num_action_types != NUM_ACTION_TYPES) {
        PyErr_SetString(PyExc_RuntimeError, "NUM_ACTION_TYPES does not match the expected value");
        return NULL;
    }
    int current_blocks = PyLong_AsLong(PyObject_GetAttrString(types_module, "CURRENT_BLOCKS"));
    if (current_blocks != CURRENT_BLOCKS) {
        PyErr_SetString(PyExc_RuntimeError, "CURRENT_BLOCKS does not match the expected value");
        return NULL;
    }
    int player_locations = PyLong_AsLong(PyObject_GetAttrString(types_module, "PLAYER_LOCATIONS"));
    if (player_locations != PLAYER_LOCATIONS) {
        PyErr_SetString(PyExc_RuntimeError, "PLAYER_LOCATIONS does not match the expected value");
        return NULL;
    }

    PyObject *mbag_env_module = PyImport_ImportModule("mbag.environment.mbag_env");
    int no_one = PyLong_AsLong(PyObject_GetAttrString(mbag_env_module, "NO_ONE"));
    if (no_one != NO_ONE) {
        PyErr_SetString(PyExc_RuntimeError, "NO_ONE does not match the expected value");
        return NULL;
    }
    int current_player = PyLong_AsLong(PyObject_GetAttrString(mbag_env_module, "CURRENT_PLAYER"));
    if (current_player != CURRENT_PLAYER) {
        PyErr_SetString(PyExc_RuntimeError, "CURRENT_PLAYER does not match the expected value");
        return NULL;
    }

    PyObject *MbagActionDistribution = PyObject_GetAttrString(
        PyImport_ImportModule("mbag.agents.action_distributions"),
        "MbagActionDistribution"
    );
    int num_channels = PyLong_AsLong(PyObject_GetAttrString(MbagActionDistribution, "NUM_CHANNELS"));
    if (num_channels != NUM_CHANNELS) {
        PyErr_SetString(PyExc_RuntimeError, "NUM_CHANNELS does not match the expected value");
        return NULL;
    }

    return PyModule_Create(&_mbag_action_distributionsmodule);
}
