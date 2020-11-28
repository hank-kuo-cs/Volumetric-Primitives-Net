import torch
from kaolin.rep import TriangleMesh
from .cuboid import cuboid_meshing
from .sphere import sphere_meshing
from config import DEVICE


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

    @classmethod
    def cone_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        pass

    @staticmethod
    def compose_meshes(meshes: list) -> TriangleMesh:
        vertices = []
        faces = []

        for i, mesh in enumerate(meshes):
            last_vertices_num = meshes[i - 1].vertices.size(0) if i != 0 else 0
            vertices_now = mesh.vertices.clone()
            faces_now = mesh.faces.clone()

            vertices.append(vertices_now)
            faces.append(faces_now + last_vertices_num)

        result_mesh = TriangleMesh.from_tensors(vertices=torch.cat(vertices), faces=torch.cat(faces))
        result_mesh.to(DEVICE)

        return result_mesh

    @staticmethod
    def check_parameters(v: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
        assert v.size(0) == q.size(0) == t.size(0)
        B = v.size(0)

        assert v.size() == (B, 3)
        assert q.size() == (B, 4)
        assert t.size() == (B, 3)
