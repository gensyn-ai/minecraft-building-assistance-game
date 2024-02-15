
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "mcts.h"

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})


// mcts_best_action(child_total_values, child_number_visits, priors, number_visits, c_puct, init_q_value, max_value, min_value, valid_action_indices, prior_scale)
PyObject* _mbag_mcts_best_action(PyObject *self, PyObject *args, PyObject *kwargs)
{
    import_array();

    // Arguments
    PyArrayObject *child_total_values_array, *child_number_visits_array, *priors_array, *valid_action_indices_array = NULL;
    int number_visits;
    float c_puct, init_q_value, max_value, min_value, prior_scale = 1.0;

    // Other variables
    PyObject *ret = NULL;

    static char *kwlist[] = {
        "child_total_values",
        "child_number_visits",
        "priors",
        "number_visits",
        "c_puct",
        "init_q_value",
        "max_value",
        "min_value",
        "valid_action_indices",
        "prior_scale",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OOOiffff|Of",
        kwlist,
        &child_total_values_array,
        &child_number_visits_array,
        &priors_array,
        &number_visits,
        &c_puct,
        &init_q_value,
        &max_value,
        &min_value,
        &valid_action_indices_array,
        &prior_scale
    ))
        goto mcts_best_action_exit;

    // Check child_total_values
    if (!PyArray_Check(child_total_values_array)) {
        PyErr_SetString(PyExc_TypeError, "child_total_values must be an array");
        goto mcts_best_action_exit;
    } else if (PyArray_NDIM(child_total_values_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "child_total_values must be a 1d array");
        goto mcts_best_action_exit;
    } else if (PyArray_TYPE(child_total_values_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "child_total_values must be an array of dtype float");
        goto mcts_best_action_exit;
    }
    const int num_actions = PyArray_DIM(child_total_values_array, 0);

    // Check child_number_visits
    if (!PyArray_Check(child_number_visits_array)) {
        PyErr_SetString(PyExc_TypeError, "child_number_visits must be an array");
        goto mcts_best_action_exit;
    } else if (PyArray_NDIM(child_number_visits_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "child_number_visits must be a 1d array");
        goto mcts_best_action_exit;
    } else if (PyArray_TYPE(child_number_visits_array) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "child_number_visits must be an array of dtype int64");
        goto mcts_best_action_exit;
    } else if (PyArray_DIM(child_number_visits_array, 0) != num_actions) {
        PyErr_SetString(PyExc_TypeError, "child_number_visits must have the same length as child_total_values");
        goto mcts_best_action_exit;
    }

    // Check priors
    if (!PyArray_Check(priors_array)) {
        PyErr_SetString(PyExc_TypeError, "priors must be an array");
        goto mcts_best_action_exit;
    } else if (PyArray_NDIM(priors_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "priors must be a 1d array");
        goto mcts_best_action_exit;
    } else if (PyArray_TYPE(priors_array) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "priors must be an array of dtype float");
        goto mcts_best_action_exit;
    } else if (PyArray_DIM(priors_array, 0) != num_actions) {
        PyErr_SetString(PyExc_TypeError, "priors must have the same length as child_total_values");
        goto mcts_best_action_exit;
    }

    // Check valid_action_indices
    if (valid_action_indices_array != NULL) {
        if (!PyArray_Check(valid_action_indices_array)) {
            PyErr_SetString(PyExc_TypeError, "valid_action_indices must be an array");
            goto mcts_best_action_exit;
        } else if (PyArray_NDIM(valid_action_indices_array) != 1) {
            PyErr_SetString(PyExc_TypeError, "valid_action_indices must be a 1d array");
            goto mcts_best_action_exit;
        } else if (PyArray_TYPE(valid_action_indices_array) != NPY_INT64) {
            PyErr_SetString(PyExc_TypeError, "valid_action_indices must be an array of dtype int");
            goto mcts_best_action_exit;
        }
    }

    int num_valid_actions = num_actions;
    if (valid_action_indices_array != NULL) {
        num_valid_actions = PyArray_DIM(valid_action_indices_array, 0);
    }

    float best_action_score = -INFINITY;
    int best_action = -1;
    for (int i = 0; i < num_valid_actions; i++) {
        int action = i;
        if (valid_action_indices_array != NULL) {
            action = *(npy_int64*)PyArray_GETPTR1(valid_action_indices_array, i);
            if (action < 0 || action >= num_actions) {
                PyErr_SetString(PyExc_ValueError, "valid_action_indices contains an invalid action");
                goto mcts_best_action_exit;
            }
        }
        float child_total_value = *(npy_float*)PyArray_GETPTR1(child_total_values_array, action);
        int child_number_visits = *(npy_int64*)PyArray_GETPTR1(child_number_visits_array, action);
        float prior = *(npy_float*)PyArray_GETPTR1(priors_array, action);
        prior *= prior_scale;

        float q;
        if (child_number_visits == 0) {
            q = init_q_value;
        } else {
            q = child_total_value / ((float) child_number_visits);
        }
        q = (q - min_value) / max(max_value - min_value, 0.01);

        float u = prior * sqrt((float) number_visits) / (1 + (float) child_number_visits);
        float score = q + c_puct * u;

        if (score > best_action_score) {
            best_action_score = score;
            best_action = action;
        }
    }
    ret = Py_BuildValue("i", best_action);

    mcts_best_action_exit:
    return ret;
}
