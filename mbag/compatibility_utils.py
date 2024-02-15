import pickle


class OldHumanDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "mbag.environment.types" and name in [
            "MbagActionTuple",
            "MbagAction",
            "MbagActionType",
        ]:
            module = "mbag.environment.actions"
        return super().find_class(module, name)
