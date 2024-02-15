
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "constants.h"
#include "action_distributions.h"
#include "blocks.h"
#include "mcts.h"

static PyMethodDef MbagMethods[] = {
    {"get_action_distribution_mask", (PyCFunction) _mbag_get_action_distribution_mask, METH_VARARGS | METH_KEYWORDS, "Get the action mask given an MBAG observation."},
    {"get_viewpoint_click_candidates", (PyCFunction) _mbag_get_viewpoint_click_candidates, METH_VARARGS | METH_KEYWORDS, "Get viewpoint-click candidates for placing or breaking a block."},
    {"mcts_best_action", (PyCFunction) _mbag_mcts_best_action, METH_VARARGS | METH_KEYWORDS, "Select the best action to expand with MCTS."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef _mbagmodule = {
    PyModuleDef_HEAD_INIT,
    "_mbag",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MbagMethods
};

PyMODINIT_FUNC
PyInit__mbag(void)
{
    import_array();

    PyObject *blocks_module = PyImport_ImportModule("mbag.environment.blocks");
    double max_player_reach = PyFloat_AsDouble(PyObject_GetAttrString(blocks_module, "MAX_PLAYER_REACH"));
    if (fabs(max_player_reach - MAX_PLAYER_REACH) > 1e-6) {
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

    PyObject *actions_module = PyImport_ImportModule("mbag.environment.actions");
    PyObject *MbagAction = PyObject_GetAttrString(actions_module, "MbagAction");
    int num_action_types = PyLong_AsLong(PyObject_GetAttrString(MbagAction, "NUM_ACTION_TYPES"));
    if (num_action_types != NUM_ACTION_TYPES) {
        PyErr_SetString(PyExc_RuntimeError, "NUM_ACTION_TYPES does not match the expected value");
        return NULL;
    }

    PyObject *types_module = PyImport_ImportModule("mbag.environment.types");
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
    int no_one = PyLong_AsLong(PyObject_GetAttrString(types_module, "NO_ONE"));
    if (no_one != NO_ONE) {
        PyErr_SetString(PyExc_RuntimeError, "NO_ONE does not match the expected value");
        return NULL;
    }
    int current_player = PyLong_AsLong(PyObject_GetAttrString(types_module, "CURRENT_PLAYER"));
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

    return PyModule_Create(&_mbagmodule);
}

