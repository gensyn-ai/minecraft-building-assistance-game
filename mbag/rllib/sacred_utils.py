import copy

from sacred.config.custom_containers import DogmaticDict, DogmaticList

# Fix for sacred issue that DogmaticDicts and DogmaticLists don't deepcopy correctly.
# This is a hacky fix, but it works for now.


def _dogmatic_dict_copy(self):
    return DogmaticDict(dict(self))


def _dogmatic_dict_deepcopy(self, memo):
    return DogmaticDict(copy.deepcopy(dict(self), memo=memo))


def _dogmatic_list_copy(self):
    return DogmaticList(list(self))


def _dogmatic_list_deepcopy(self, memo):
    return DogmaticList(copy.deepcopy(list(self), memo=memo))


DogmaticDict.__copy__ = _dogmatic_dict_copy  # type: ignore
DogmaticDict.__deepcopy__ = _dogmatic_dict_deepcopy  # type: ignore
DogmaticList.__copy__ = _dogmatic_list_copy  # type: ignore
DogmaticList.__deepcopy__ = _dogmatic_list_deepcopy  # type: ignore
