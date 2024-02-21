
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "blocks.h"
#include "constants.h"


double* get_viewpoint_click_candidates(
    int action_type,
    int width,
    int height,
    int depth,
    int block_x,
    int block_y,
    int block_z,
    double player_x,
    double player_y,
    double player_z,
    void* blocks_data,
    npy_uint8 (*get_block)(void*, int, int, int),
    void* other_player_data,
    bool (*is_player)(void*, int, int, int),
    int *num_viewpoint_click_candidates
) {
    int i, j, k, x, y, z;
    int collision;
    double *viewpoint_click_candidates = NULL;

    // Find possible locations to click on.
    double click_locations[3 * 2 * 3 * 3][3];
    double shift = action_type == BREAK_BLOCK ? 1e-4 : -1e-4;
    i = 0;
    for (int face_dim = 0; face_dim < 3; face_dim++) {
        // for face in [0 - shift, 1 + shift]
        for (double face = 0 - shift; face < 1.1; face += 1 + 2 * shift) {
            if (action_type == PLACE_BLOCK) {
                int against_block_location[3] = { block_x, block_y, block_z };
                against_block_location[face_dim] += face > 0.5 ? 1 : -1;
                // Needs to be in bounds and solid (not air)
                if (
                    against_block_location[0] < 0 || against_block_location[0] >= width
                    || against_block_location[1] < 0 || against_block_location[1] >= height
                    || against_block_location[2] < 0 || against_block_location[2] >= depth
                    || get_block(blocks_data, against_block_location[0], against_block_location[1], against_block_location[2]) == AIR
                ) {
                    continue;
                }
            }

            for (double u = 0.1; u < 1; u += 0.4) {
                for (double v = 0.1; v < 1; v += 0.4) {
                    click_locations[i][0] = block_x;
                    click_locations[i][1] = block_y;
                    click_locations[i][2] = block_z;
                    click_locations[i][face_dim] += face;
                    click_locations[i][(face_dim + 1) % 3] += u;
                    click_locations[i][(face_dim + 2) % 3] += v;

                    // Make sure the click location is within the world.
                    if (
                        click_locations[i][0] < 0
                        || click_locations[i][0] >= width
                        || click_locations[i][1] < 0
                        || click_locations[i][1] >= height
                        || click_locations[i][2] < 0
                        || click_locations[i][2] >= depth
                    ) continue;

                    i++;
                }
            }
        }
    }
    int num_click_locations = i;

    // Find possible locations that the player can stand.
    double player_locations[9 * 9 * 9][3];
    i = 0;
    if (!isnan(player_x)) {
        player_locations[i][0] = player_x;
        player_locations[i][1] = player_y;
        player_locations[i][2] = player_z;
        i++;
    } else {
        for (int dx = -4; dx <= 4; dx++) {
            for (int dy = -5; dy <= 3; dy++) {
                for (int dz = -4; dz <= 4; dz++) {
                    // Remove deltas which would put the player inside the block
                    // being placed/broken.
                    if (
                        dx == 0
                        && dz == 0
                        && dy >= -1
                        && dy <= (action_type == PLACE_BLOCK ? 1 : 0)
                    ) continue;

                    // Make sure the player would be within the world.
                    if (
                        block_x + dx < 0
                        || block_x + dx >= width
                        || block_y + dy < 0
                        || block_y + dy >= height
                        || block_z + dz < 0
                        || block_z + dz >= depth
                    ) continue;

                    // Make sure the player would not be inside a block.
                    if (
                        get_block(blocks_data, block_x + dx, block_y + dy, block_z + dz) != AIR
                        || (
                            block_y + dy + 1 < height
                            && get_block(blocks_data, block_x + dx, block_y + dy + 1, block_z + dz) != AIR
                        )
                    ) continue;

                    // Make sure the player would not be overlapping with another
                    // player.
                    if (
                        is_player(other_player_data, block_x + dx, block_y + dy, block_z + dz)
                        || is_player(other_player_data, block_x + dx, block_y + dy + 1, block_z + dz)
                    ) {
                        continue;
                    }

                    player_locations[i][0] = block_x + 0.5 + dx;
                    player_locations[i][1] = block_y + dy;
                    player_locations[i][2] = block_z + 0.5 + dz;
                    i++;
                }
            }
        }
    }
    int num_player_locations = i;

    viewpoint_click_candidates = malloc(num_click_locations * num_player_locations * 6 * sizeof(double));
    i = 0;
    for (j = 0; j < num_player_locations; j++) {
        for (k = 0; k < num_click_locations; k++) {
            double viewpoint_x = player_locations[j][0];
            double viewpoint_y = player_locations[j][1] + 1.6;
            double viewpoint_z = player_locations[j][2];
            double click_x = click_locations[k][0];
            double click_y = click_locations[k][1];
            double click_z = click_locations[k][2];
            double delta_x = click_x - viewpoint_x;
            double delta_y = click_y - viewpoint_y;
            double delta_z = click_z - viewpoint_z;

            // Check if the player can reach the click location.
            if (
                delta_x * delta_x
                + delta_y * delta_y
                + delta_z * delta_z
                > MAX_PLAYER_REACH * MAX_PLAYER_REACH
            ) continue;

            // Voxel traversal to make sure there are no blocks between the viewpoint
            // and the click location.
            // Based on http://www.cse.yorku.ca/~amana/research/grid.pdf
            int step_x = (int) copysign(1, delta_x);
            int step_y = (int) copysign(1, delta_y);
            int step_z = (int) copysign(1, delta_z);
            double t_max_x = delta_x == 0 ? 1 : fabs(((-step_x * viewpoint_x) - floor(-step_x * viewpoint_x)) / delta_x);
            double t_max_y = delta_y == 0 ? 1 : fabs(((-step_y * viewpoint_y) - floor(-step_y * viewpoint_y)) / delta_y);
            double t_max_z = delta_z == 0 ? 1 : fabs(((-step_z * viewpoint_z) - floor(-step_z * viewpoint_z)) / delta_z);
            double t_delta_x = delta_x == 0 ? 1 : fabs(1.0 / delta_x);
            double t_delta_y = delta_y == 0 ? 1 : fabs(1.0 / delta_y);
            double t_delta_z = delta_z == 0 ? 1 : fabs(1.0 / delta_z);
            x = (int) floor(viewpoint_x);
            y = (int) floor(viewpoint_y);
            z = (int) floor(viewpoint_z);
            collision = false;
            while (t_max_x < 1 || t_max_y < 1 || t_max_z < 1) {
                if (t_max_x <= t_max_y) {
                    if (t_max_x <= t_max_z) {
                        x += step_x;
                        t_max_x += t_delta_x;
                    } else {
                        z += step_z;
                        t_max_z += t_delta_z;
                    }
                } else {
                    if (t_max_y <= t_max_z) {
                        y += step_y;
                        t_max_y += t_delta_y;
                    } else {
                        z += step_z;
                        t_max_z += t_delta_z;
                    }
                }
                if (
                    x < 0 || x >= width || y < 0 || z < 0 || z >= depth
                ) {
                    // Don't test for an upper bound on y because the player's
                    // head can be above the world.
                    PyErr_SetString(PyExc_RuntimeError, "voxel traversal went out of bounds");
                    goto get_viewpoint_click_candidates_error;
                } else {
                    // Make sure the block is empty. This means it must be air and
                    // not contain a player.
                    if (
                        y < height
                        && get_block(blocks_data, x, y, z) != AIR
                    ) {
                        collision = true;
                        break;
                    }
                    if (is_player(other_player_data, x, y, z)) {
                        collision = true;
                        break;
                    }
                }
            }
            if(!collision) {
                viewpoint_click_candidates[i * 6] = viewpoint_x;
                viewpoint_click_candidates[i * 6 + 1] = viewpoint_y;
                viewpoint_click_candidates[i * 6 + 2] = viewpoint_z;
                viewpoint_click_candidates[i * 6 + 3] = click_x;
                viewpoint_click_candidates[i * 6 + 4] = click_y;
                viewpoint_click_candidates[i * 6 + 5] = click_z;
                i++;
            }
        }
    }
    *num_viewpoint_click_candidates = i;

    return viewpoint_click_candidates;

    get_viewpoint_click_candidates_error:
    if (viewpoint_click_candidates != NULL) {
        free(viewpoint_click_candidates);
        viewpoint_click_candidates = NULL;
    }
    return viewpoint_click_candidates;
}


typedef struct {
    double *other_player_locations;
    int num_other_players;
} other_player_locations_vector_t;

bool is_player_in_other_player_locations_vector(void *other_players_locations_ptr, int x, int y, int z) {
    int i;
    other_player_locations_vector_t *other_player_locations_vector = (other_player_locations_vector_t*) other_players_locations_ptr;
    double *other_player_locations = other_player_locations_vector->other_player_locations;
    int num_other_players = other_player_locations_vector->num_other_players;
    for (i = 0; i < num_other_players; i++) {
        int other_x = (int) other_player_locations[i * 3];
        int other_y = (int) other_player_locations[i * 3 + 1];
        int other_z = (int) other_player_locations[i * 3 + 2];
        if (
            x == other_x
            && (y == other_y || y == other_y + 1)
            && z == other_z
        ) return true;
    }
    return false;
}

npy_uint8 get_block_from_blocks_array(void *blocks_array, int x, int y, int z) {
    return *(npy_uint8*)PyArray_GETPTR3((PyArrayObject*) blocks_array, x, y, z);
}


// get_viewpoint_click_candidates(blocks, action_type, block_location, player_location, other_player_locations)
PyObject* _mbag_get_viewpoint_click_candidates(PyObject *self, PyObject *args, PyObject *kwargs)
{
    import_array();

    // Arguments
    PyArrayObject *blocks_array;
    int action_type;
    PyObject *block_location_tuple, *player_location_optional_tuple, *other_player_locations_list;

    // Other variables
    int i;
    double *other_player_locations = NULL, *viewpoint_click_candidates = NULL;
    PyObject *ret = NULL;

    static char *kwlist[] = {
        "blocks",
        "action_type",
        "block_location",
        "player_location",
        "other_player_locations",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OiOOO",
        kwlist,
        &blocks_array,
        &action_type,
        &block_location_tuple,
        &player_location_optional_tuple,
        &other_player_locations_list
    ))
        goto _mbag_get_viewpoint_click_candidates_exit;

    if (!PyArray_Check(blocks_array)) {
        PyErr_SetString(PyExc_TypeError, "blocks must be an array");
        goto _mbag_get_viewpoint_click_candidates_exit;
    } else if (PyArray_NDIM(blocks_array) != 3) {
        PyErr_SetString(PyExc_TypeError, "blocks must be a 3d array");
        goto _mbag_get_viewpoint_click_candidates_exit;
    } else if (PyArray_TYPE(blocks_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "blocks must be an array of dtype uint8");
        goto _mbag_get_viewpoint_click_candidates_exit;
    }
    const int width = PyArray_DIMS(blocks_array)[0];
    const int height = PyArray_DIMS(blocks_array)[1];
    const int depth = PyArray_DIMS(blocks_array)[2];

    int block_x, block_y, block_z;
    if (!PyTuple_Check(block_location_tuple) || PyTuple_Size(block_location_tuple) != 3) {
        PyErr_SetString(PyExc_TypeError, "block_location must be a tuple of length 3");
        goto _mbag_get_viewpoint_click_candidates_exit;
    }
    block_x = PyLong_AsLong(PyTuple_GetItem(block_location_tuple, 0));
    block_y = PyLong_AsLong(PyTuple_GetItem(block_location_tuple, 1));
    block_z = PyLong_AsLong(PyTuple_GetItem(block_location_tuple, 2));

    double player_x, player_y, player_z;
    if (player_location_optional_tuple == Py_None) {
        player_x = NAN;
        player_y = NAN;
        player_z = NAN;
    } else if (!PyTuple_Check(player_location_optional_tuple) || PyTuple_Size(player_location_optional_tuple) != 3) {
        PyErr_SetString(PyExc_TypeError, "player_location must be a tuple of length 3");
        goto _mbag_get_viewpoint_click_candidates_exit;
    } else {
        player_x = PyFloat_AsDouble(PyTuple_GetItem(player_location_optional_tuple, 0));
        player_y = PyFloat_AsDouble(PyTuple_GetItem(player_location_optional_tuple, 1));
        player_z = PyFloat_AsDouble(PyTuple_GetItem(player_location_optional_tuple, 2));
    }

    if (!PyList_Check(other_player_locations_list)) {
        PyErr_SetString(PyExc_TypeError, "other_player_locations must be a list");
        goto _mbag_get_viewpoint_click_candidates_exit;
    }
    int num_other_players = PyList_Size(other_player_locations_list);
    other_player_locations = (double*) malloc(num_other_players * 3 * sizeof(double));
    for (i = 0; i < num_other_players; i++) {
        PyObject *player_location_tuple = PyList_GetItem(other_player_locations_list, i);
        if (!PyTuple_Check(player_location_tuple) || PyTuple_Size(player_location_tuple) != 3) {
            PyErr_SetString(PyExc_TypeError, "other_player_locations must be a list of tuples of length 3");
            goto _mbag_get_viewpoint_click_candidates_exit;
        }
        other_player_locations[i * 3] = PyFloat_AsDouble(PyTuple_GetItem(player_location_tuple, 0));
        other_player_locations[i * 3 + 1] = PyFloat_AsDouble(PyTuple_GetItem(player_location_tuple, 1));
        other_player_locations[i * 3 + 2] = PyFloat_AsDouble(PyTuple_GetItem(player_location_tuple, 2));
    }

    int num_viewpoint_click_candidates;
    other_player_locations_vector_t other_player_locations_vector = {other_player_locations, num_other_players};

    viewpoint_click_candidates = get_viewpoint_click_candidates(
        action_type,
        width,
        height,
        depth,
        block_x,
        block_y,
        block_z,
        player_x,
        player_y,
        player_z,
        blocks_array,
        get_block_from_blocks_array,
        &other_player_locations_vector,
        is_player_in_other_player_locations_vector,
        &num_viewpoint_click_candidates
    );
    if (viewpoint_click_candidates == NULL) {
        goto _mbag_get_viewpoint_click_candidates_exit;
    }

    // Return the viewpoint click candidates.
    npy_intp dims[] = {num_viewpoint_click_candidates, 2, 3};
    PyArrayObject *viewpoint_click_candidates_array = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    for (i = 0; i < num_viewpoint_click_candidates; i++) {
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 0, 0) = (npy_double) viewpoint_click_candidates[i * 6 + 0];
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 0, 1) = (npy_double) viewpoint_click_candidates[i * 6 + 1];
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 0, 2) = (npy_double) viewpoint_click_candidates[i * 6 + 2];
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 1, 0) = (npy_double) viewpoint_click_candidates[i * 6 + 3];
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 1, 1) = (npy_double) viewpoint_click_candidates[i * 6 + 4];
        *(npy_double*)PyArray_GETPTR3(viewpoint_click_candidates_array, i, 1, 2) = (npy_double) viewpoint_click_candidates[i * 6 + 5];
    }
    ret = PyArray_Return(viewpoint_click_candidates_array);

    _mbag_get_viewpoint_click_candidates_exit:
    if (other_player_locations != NULL) free(other_player_locations);
    if (viewpoint_click_candidates != NULL) free(viewpoint_click_candidates);
    return ret;
}
