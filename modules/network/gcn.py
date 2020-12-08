import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from torch_geometric.nn import GCNConv, TAGConv, GraphUNet, BatchNorm


class GCNModel(nn.Module):
    def __init__(self, n_dim=3, img_feature_dim=960, v_num=2048, use_position_encoding=True):
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

    def forward(self, meshes: list, rgbs: torch.Tensor, perceptual_features: list, global_features: torch.Tensor=None):
        # meshes: [TriangleMesh, ...]
        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)
        edge_indices = self.get_edge_indices(meshes[0])  # (2, K)

        x = self.positional_encoding(batch_vertices) if self.use_position_encoding else batch_vertices

        global_features = global_features[:, None, :].repeat(1, x.size(1), 1)
        local_features = self.get_local_features(batch_vertices, rgbs, perceptual_features)

        x = torch.cat([x, local_features, global_features], 2)

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

    @classmethod
    def get_local_features(cls, vertices: torch.Tensor, rgbs: torch.Tensor, perceptual_features: list):
        bounds = cls.get_bound_of_images(rgbs)
        local_features = cls.perceptual_feature_pooling(perceptual_features, vertices, bounds)
        return local_features

    @staticmethod
    def get_bound_of_images(imgs: torch.Tensor):
        assert imgs.ndimension() == 4  # (B, C, H, W)
        h, w = imgs.size(2), imgs.size(3)
        bounds = torch.zeros((imgs.size(0), 4)).cuda()
        bounds[:, 1] = w
        bounds[:, 3] = h

        for b in range(imgs.size(0)):
            img = imgs[b]
            mask = img.sum(0)

            x_any = (mask > 0.03).any(0)  # set  0.03 to prevent mask has some low noise value to affect the result
            y_any = (mask > 0.03).any(1)

            for i in range(w):
                j = w - i - 1

                if x_any[i] and bounds[b, 0] == 0:
                    bounds[b, 0] = i

                if x_any[j] and bounds[b, 1] == w:
                    bounds[b, 1] = j

                if bounds[b, 0] > 0 and bounds[b, 1] < w:
                    break

            for i in range(h):
                j = h - i - 1

                if y_any[i] and bounds[b, 2] == 0:
                    bounds[b, 2] = i

                if y_any[j] and bounds[b, 3] == h:
                    bounds[b, 3] = j

                if bounds[b, 2] > 0 and bounds[b, 3] < h:
                    break

        # normalize bounds value to [-1, 1]
        bounds[:, :2] = bounds[:, :2] / w * 2 - 1
        bounds[:, 2: 4] = bounds[:, 2: 4] / h * 2 - 1

        return bounds  # (B, 4)

    @staticmethod
    def perceptual_feature_pooling(perceptual_features: list, points: torch.Tensor, bounds: torch.Tensor):
        assert points.ndimension() == 3.  # (B, N, 3)
        assert bounds.ndimension() == 2.  # (B, 4)

        bounds = bounds[:, None, :].repeat(1, points.size(1), 1)  # (B, N, 4)

        grids = torch.zeros((points.size(0), points.size(1), 2), dtype=torch.float, device=points.device)
        max_zs = torch.zeros((points.size(0), points.size(1), 1), dtype=torch.float, device=points.device)
        min_zs, max_ys, min_ys = max_zs.clone(), max_zs.clone(), max_zs.clone()

        for b in range(points.size(0)):
            instance_points = points[b]
            max_pos, min_pos = instance_points.max(0)[0], instance_points.min(0)[0]
            max_zs[b, :, 0], min_zs[b, :, 0] = max_pos[2], min_pos[2]
            max_ys[b, :, 0], min_ys[b, :, 0] = max_pos[1], min_pos[1]

        grids[..., 0] = bounds[..., 0] + (1 - (points[..., 2] - min_zs[..., 0]) / (max_zs[..., 0] - min_zs[..., 0])) * (bounds[..., 1] - bounds[..., 0])
        grids[..., 1] = bounds[..., 2] + (1 - (points[..., 1] - min_ys[..., 0]) / (max_ys[..., 0] - min_ys[..., 0])) * (bounds[..., 3] - bounds[..., 2])

        grids = grids[:, None, :, :]

        pooling_features = []

        for layer_features in perceptual_features:
            layer_pooling_features = nn.functional.grid_sample(layer_features, grids, align_corners=True)
            pooling_features.append(layer_pooling_features)

        pooling_features = torch.cat(pooling_features, 1).view(points.size(0), -1, points.size(1)).permute(0, 2, 1)
        return pooling_features  # (B, N, C)
