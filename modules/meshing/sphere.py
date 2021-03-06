import os
import torch
from ..transform import transform_points
from config import DEVICE
from kaolin.rep import TriangleMesh


def sphere_meshing(v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
    meshes = []
    batch_vertices = []

    B = v.size(0)

    for b in range(B):
        mesh = load_cuboid()
        vertices = (mesh.vertices * v[b]).unsqueeze(0)

        batch_vertices.append(vertices)
        meshes.append(mesh)

    batch_vertices = torch.cat(batch_vertices)
    batch_vertices = transform_points(batch_vertices, q, t)

    for b in range(B):
        meshes[b].vertices = batch_vertices[b]

    return meshes


def load_cuboid():
    obj_path = os.path.dirname(__file__) + '/objects/sphere.obj'
    mesh = TriangleMesh.from_obj(obj_path)
    mesh.vertices -= torch.mean(mesh.vertices, 0)  # zero center
    mesh.vertices /= torch.mean(torch.norm(mesh.vertices, dim=1))  # normalize length
    mesh.to(DEVICE)
    return mesh
