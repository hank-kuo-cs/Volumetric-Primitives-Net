import torch
import trimesh
from kaolin.rep import TriangleMesh
from config import DEVICE, DECOMPOSE_CONVEX_NUM


def approximate_convex_decomposition(mesh: TriangleMesh) -> (TriangleMesh, torch.Tensor, torch.Tensor):
    mesh = get_trimesh_from_kaolinmesh(mesh)
    convex_list = mesh.convex_decomposition(DECOMPOSE_CONVEX_NUM, beta=0.8, alpha=0.8, concavity=0.01)
    convex_list = [get_kaolinmesh_from_trimesh(convex) for convex in convex_list]

    mesh, uv, texture = merge_meshes(convex_list)
    return mesh, uv, texture


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
    vertices, faces, texture, uv = [], [], [], []

    for i, mesh in enumerate(meshes):
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_num)
        uv.append(torch.full((mesh.vertices.size(0), 2), i/len(meshes) + 0.01))
        texture.append(get_random_colors())

        vertex_num += mesh.vertices.size(0)

    vertices = torch.cat(vertices)
    faces = torch.cat(faces)

    uv = torch.cat(uv)[None].to(DEVICE)
    texture = torch.cat(texture, 2)[None].to(DEVICE)

    merged_mesh = TriangleMesh.from_tensors(vertices, faces)
    merged_mesh.to(DEVICE)

    return merged_mesh, uv, texture


def get_random_colors():
    c = torch.rand(3)
    return torch.cat([torch.full((1, 1, 1), c[i].item()) for i in range(3)], 0)
