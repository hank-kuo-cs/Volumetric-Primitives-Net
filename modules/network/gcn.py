import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from torch_geometric.nn import GCNConv, TAGConv, GraphUNet, BatchNorm


class GCNModel(nn.Module):
    def __init__(self, n_dim=3):
        super().__init__()
        conv = GCNConv
        self.relu = nn.ReLU()
        self.conv1 = conv(n_dim, 512)
        self.conv2 = conv(512, 512)

        self.conv3 = conv(512, 512)
        self.conv4 = conv(512, 512)

        self.conv5 = conv(512, 64)
        self.conv6 = conv(64, 3)

    def forward(self, meshes: list, img_features: torch.Tensor):
        # meshes: [TriangleMesh, ...]
        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)
        edge_indices = self.get_edge_indices(meshes[0])  # (2, K)

        x = torch.cat([batch_vertices, img_features[:, None, :].repeat(1, batch_vertices.size(1), 1)], 2)

        x = self.conv1(x, edge_indices, None)
        x = self.conv2(x, edge_indices, None)
        x = self.relu(x)

        x = self.conv3(x, edge_indices, None)
        x = self.conv4(x, edge_indices, None)
        x = self.relu(x)

        x = self.conv5(x, edge_indices, None)
        x = self.conv6(x, edge_indices, None)

        return x

    @staticmethod
    def get_edge_indices(mesh: TriangleMesh):
        edges = TriangleMesh.compute_adjacency_info(mesh.vertices, mesh.faces)[1]
        edge_indices = torch.cat([edges, edges.flip(1)]).view(2, -1)
        return edge_indices

    @staticmethod
    def get_batch_vertices(meshes: list):
        return torch.cat([mesh.vertices[None] for mesh in meshes])
