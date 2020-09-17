import torch


class ShapeNetData:
    def __init__(self,
                 img_path: str,
                 obj_path: str,
                 dist: float,
                 elev: float,
                 azim: float):
        self.img_path = img_path
        self.obj_path = obj_path
        self.dist = dist
        self.elev = elev
        self.azim = azim

        self.check_parameters()

    def check_parameters(self):
        assert type(self.img_path) == str and len(self.img_path) > 0
        assert type(self.obj_path) == str and len(self.obj_path) > 0
        assert type(self.dist) == float and self.dist > 0
        assert type(self.elev) == float and 90.0 >= self.elev >= -90.0
        assert type(self.azim) == float and 360.0 >= self.azim >= -360.0
