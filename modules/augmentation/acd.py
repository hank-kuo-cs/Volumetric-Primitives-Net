import torch
import random
import trimesh
from ..transform import rotate_points
from kaolin.rep import TriangleMesh


def rotate_points_forward_vec(points: torch.Tensor, angle: float, vec: list):
    assert points.ndimension() == 2

    if not points.is_cuda:
        points = points.cuda()

    v = torch.tensor([vec], dtype=torch.float).cuda()

    angles = torch.tensor([[angle]], dtype=torch.float).cuda() / 360
    q = torch.cat([v, angles], dim=1).cuda()

    points = rotate_points(points[None], q)

    return points[0].detach().cpu()


def acd(mesh: trimesh.Trimesh, hull_num: int = 8):
    convex_hulls = mesh.convex_decomposition(hull_num)
    if not isinstance(convex_hulls, list):
        convex_hulls = [convex_hulls]
    return convex_hulls


def is_center(mesh: TriangleMesh) -> bool:
    return ((torch.any(mesh.vertices[:, 2] > 0) and
             torch.any(mesh.vertices[:, 2] < 0)) or
            mesh.vertices[:, 2].abs().min() < 0.05).item()


def random_scale(convex_hulls: list) -> list:
    scale = 0.8 + torch.rand(1).item() * 0.4  # s~[0.8, 1.2]

    for i, convex_hull in enumerate(convex_hulls):
        convex_hulls[i].vertices *= scale

    return convex_hulls


def random_translate(convex_hulls: list) -> list:
    t = (torch.rand(1).item() - 0.5) / 5  # t~[-0.1, 0.1]

    for i, convex_hull in enumerate(convex_hulls):
        convex_hulls[i].vertices[:, 1] += t

    return convex_hulls


def random_rotate_forward_axis(convex_hulls: list) -> list:
    y, z = [0, 1, 0], [0, 0, 1]
    #     axis = random.choice([y, z])
    axis = z
    angle = random.choice([90.0, -90.0, 0.0, 180.0, -180.0])

    for i, convex_hull in enumerate(convex_hulls):
        convex_hulls[i].vertices = rotate_points_forward_vec(convex_hull.vertices, angle, axis)

    return convex_hulls


def random_cutout(convex_hulls: list) -> list:
    center_indices = [i for i, convex_hull in enumerate(convex_hulls) if is_center(convex_hull)]

    if random.choice([True, False]) and len(center_indices) > 1:
        num = random.randint(1, len(center_indices) - 1)
        cutout_indices = random.sample(center_indices, num)
        convex_hulls = [convex_hulls[i] for i in range(len(convex_hulls)) if i not in cutout_indices]
    else:
        convex_hulls = [convex_hulls[i] for i in range(len(convex_hulls)) if i in center_indices]

    return convex_hulls


def get_trimesh_from_kaolinmesh(kao_mesh, is_list=False):
    def one(kao_mesh):
        m = trimesh.Trimesh()
        m.vertices = kao_mesh.vertices.cpu()
        m.faces = kao_mesh.faces.cpu()
        return m
    if not is_list:
        return one(kao_mesh)
    else:
        return [one(k) for k in kao_mesh]


def get_kaolinmesh_from_trimesh(tri_mesh, is_list=False):
    def one(tri_mesh):
        vertices = torch.tensor(tri_mesh.vertices, dtype=torch.float)
        faces = torch.tensor(tri_mesh.faces, dtype=torch.long)
        return TriangleMesh.from_tensors(vertices, faces)
    if not is_list:
        return one(tri_mesh)
    else:
        return [one(t) for t in tri_mesh]


def merge_meshes(convex_hulls: list) -> TriangleMesh:
    # convex_hulls: [trimesh.TriMesh, ...]
    if isinstance(convex_hulls[0], TriangleMesh):
        convex_hulls = get_trimesh_from_kaolinmesh(convex_hulls, is_list=True)
    mesh = trimesh.boolean.union(convex_hulls)
    mesh = get_kaolinmesh_from_trimesh(mesh)
    return mesh


def augment(convex_hulls: list) -> list:
    convex_hulls = random_cutout(convex_hulls)
    convex_hulls = random_scale(convex_hulls)
    convex_hulls = random_rotate_forward_axis(convex_hulls)
    convex_hulls = random_translate(convex_hulls)
    return convex_hulls
