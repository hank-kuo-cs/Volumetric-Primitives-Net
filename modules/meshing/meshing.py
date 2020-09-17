import torch
from .cuboid import cuboid_meshing
from .sphere import sphere_meshing


class Meshing:
    def __init__(self):
        pass

    @classmethod
    def cuboid_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        return cuboid_meshing(v, q, t)

    @classmethod
    def sphere_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        return sphere_meshing(v, q, t)

    @staticmethod
    def check_parameters(v: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
        assert v.size(0) == q.size(0) == t.size(0)
        B = v.size(0)

        assert v.size() == (B, 3)
        assert q.size() == (B, 4)
        assert t.size() == (B, 3)
