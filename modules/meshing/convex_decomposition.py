import torch
import trimesh
from kaolin.rep import TriangleMesh
from config import DEVICE


def approximate_convex_decomposition(mesh: TriangleMesh) -> TriangleMesh:
    mesh = get_trimesh_from_kaolinmesh(mesh)
    convex_list = mesh.convex_decomposition(8, beta=0.8, alpha=0.8, concavity=0.01)
    convex_list = [get_kaolinmesh_from_trimesh(convex) for convex in convex_list]

    mesh, colors = merge_meshes(convex_list)
    return mesh, colors


def get_trimesh_from_kaolinmesh(kao_mesh: TriangleMesh):
    m = trimesh.Trimesh()
    m.vertices = kao_mesh.vertices.cpu()
    m.faces = kao_mesh.faces.cpu()
    return m


def get_kaolinmesh_from_trimesh(tri_mesh: trimesh.Trimesh):
    vertices = torch.tensor(tri_mesh.vertices, dtype=torch.float)
    faces = torch.tensor(tri_mesh.faces, dtype=torch.long)
    return TriangleMesh.from_tensors(vertices, faces)


def merge_meshes(meshes: list) -> (TriangleMesh, torch.Tensor):
    vertex_num = 0
    vertices, faces, colors = [], [], []

    for i, mesh in enumerate(meshes):
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_num)
        colors.append(get_random_colors(mesh.vertices.size(0)))

        vertex_num += mesh.vertices.size(0)

    vertices = torch.cat(vertices)
    faces = torch.cat(faces)
    colors = torch.cat(colors).to(DEVICE)

    merged_mesh = TriangleMesh.from_tensors(vertices, faces)
    merged_mesh.to(DEVICE)

    return merged_mesh, colors


def get_random_colors(n: int):
    c = torch.rand(3)
    return torch.cat([torch.full((n, 1), c[i].item()) for i in range(3)], 1)
