"""VLA-TTT: a RoboTTT-inspired, stateful VLA implementation."""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "VLATTTPolicy":
        from .models.policy import VLATTTPolicy

        return VLATTTPolicy
    raise AttributeError(name)

