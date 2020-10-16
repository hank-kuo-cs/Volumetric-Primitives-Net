import torch


class ShapeNetData:
    def __init__(self,
                 img_path: str,
                 canonical_obj_path: str,
                 view_center_obj_path: str,
                 dist: float,
                 elev: float,
                 azim: float):
        self.img_path = img_path
        self.canonical_obj_path = canonical_obj_path
        self.view_center_obj_path = view_center_obj_path
        self.dist = dist
        self.elev = elev
        self.azim = azim

        self.check_parameters()

    def check_parameters(self):
        assert type(self.img_path) == str and len(self.img_path) > 0
        assert type(self.canonical_obj_path) == str and len(self.canonical_obj_path) > 0
        assert type(self.view_center_obj_path) == str and len(self.view_center_obj_path) > 0
        assert type(self.dist) == float and self.dist > 0
        assert type(self.elev) == float and 90.0 >= self.elev >= -90.0
        assert type(self.azim) == float and 360.0 >= self.azim >= -360.0
