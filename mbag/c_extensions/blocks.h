#ifndef BLOCKS_H
#define BLOCKS_H

#include <stdbool.h>
#include <numpy/arrayobject.h>

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
);
PyObject* _mbag_get_viewpoint_click_candidates(PyObject *self, PyObject *args, PyObject *kwargs);

#endif
