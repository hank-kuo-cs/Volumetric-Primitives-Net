import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from torch_geometric.nn import GCNConv, TAGConv, GraphUNet, BatchNorm


class GCNModel(nn.Module):
    def __init__(self, n_dim=3, img_feature_dim=512, v_num=2048, use_position_encoding=False):
        super().__init__()
        conv = GCNConv
        self.use_position_encoding = use_position_encoding
        self.relu = nn.ReLU()

        n_dim = n_dim + n_dim * 12 if use_position_encoding else n_dim

        self.conv1 = conv(n_dim + img_feature_dim, 512)
        self.conv2 = conv(512, 512)

        self.conv3 = conv(512, 512)
        self.conv4 = conv(512, 512)

        self.conv5 = conv(512, 64)
        self.conv6 = conv(64, 3)

        self.fc = nn.Sequential(
            nn.Linear(v_num * 3, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, v_num * 3),
            nn.Tanh()
        )

    def forward(self, meshes: list, img_features: torch.Tensor):
        # meshes: [TriangleMesh, ...]
        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)
        edge_indices = self.get_edge_indices(meshes[0])  # (2, K)

        if self.use_position_encoding:
            batch_vertices = self.positional_encoding(batch_vertices)

        x = torch.cat([batch_vertices, img_features[:, None, :].repeat(1, batch_vertices.size(1), 1)], 2)

        x = self.conv1(x, edge_indices, None)
        x = self.conv2(x, edge_indices, None)
        x = self.relu(x)

        x = self.conv3(x, edge_indices, None)
        x = self.conv4(x, edge_indices, None)
        x = self.relu(x)

        x = self.conv5(x, edge_indices, None)
        x = self.conv6(x, edge_indices, None)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        deformations = self.fc(x).view(x.size(0), -1, 3) * 0.1
        predict_vertices = batch_vertices + deformations

        return predict_vertices

    @staticmethod
    def get_edge_indices(mesh: TriangleMesh):
        edges = TriangleMesh.compute_adjacency_info(mesh.vertices, mesh.faces)[1]
        edge_indices = torch.cat([edges, edges.flip(1)]).view(2, -1)
        return edge_indices

    @staticmethod
    def get_batch_vertices(meshes: list):
        return torch.cat([mesh.vertices[None] for mesh in meshes])

    @staticmethod
    def positional_encoding(x: torch.Tensor):
        encoding = [x]
        frequency_bands = 2.0 ** torch.linspace(0, 5, 6, dtype=torch.float, device=x.device)

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))

        return torch.cat(encoding, dim=-1)
