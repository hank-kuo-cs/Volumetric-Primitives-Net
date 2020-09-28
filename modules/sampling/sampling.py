import torch
from .cuboid import cuboid_sampling
from .sphere import sphere_sampling
from ..transform import transform_points


class Sampling:
    def __init__(self):
        pass

    @classmethod
    def cuboid_sampling(cls,
                        v: torch.Tensor,
                        q: torch.Tensor,
                        t: torch.Tensor,
                        num_points: int = 1000):

        cls.check_parameters(v, q, t)

        canonical_points = cuboid_sampling(v, num_points)
        sample_points = transform_points(canonical_points, q, t)

        return sample_points

    @classmethod
    def sphere_sampling(cls,
                        v: torch.Tensor,
                        q: torch.Tensor,
                        t: torch.Tensor,
                        num_points: int = 1000):

        cls.check_parameters(v, q, t)

        canonical_points = sphere_sampling(v, num_points)
        sample_points = transform_points(canonical_points, q, t)

        return sample_points

    @classmethod
    def cone_sampling(cls,
                      v: torch.Tensor,
                      q: torch.Tensor,
                      t: torch.Tensor,
                      num_points: int = 1000):
        pass

    @staticmethod
    def check_parameters(v, q, t):
        B = v.size(0)
        assert v.size() == (B, 3)
        assert q.size() == (B, 4)
        assert t.size() == (B, 3)
