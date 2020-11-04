import torch
import numpy as np
import open3d as o3d
from kaolin.rep import TriangleMesh


def ball_pivot_surface_reconstruction(points: torch.Tensor) -> TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1 * avg_dist

    recon_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2, radius * 4, radius * 8]))

    vertices = torch.tensor(recon_mesh.vertices, dtype=torch.float)
    faces = torch.tensor(recon_mesh.triangles, dtype=torch.long)

    faces_ex = faces.clone()
    faces_ex[..., 1] = faces[..., 2]
    faces_ex[..., 2] = faces[..., 1]
    faces = torch.cat([faces, faces_ex], 0)

    recon_mesh = TriangleMesh.from_tensors(vertices, faces)

    return recon_mesh
