class_dict = {'airplane': '02691156', 'rifle': '04090263', 'display': '03211117', 'table': '04379243',
              'telephone': '04401088', 'car': '02958343', 'chair': '03001627', 'bench': '02828884', 'lamp': '03636649',
              'cabinet': '02933112', 'loudspeaker': '03691459', 'sofa': '04256520', 'watercraft': '04530566'}
class_ids = list(class_dict.values())
class_names = list(class_dict.keys())


class Classes:
    @staticmethod
    def get_id_by_name(class_name):
        return class_ids[class_names.index(class_name)]

    @staticmethod
    def get_name_by_id(class_id):
        return class_names[class_ids.index(class_id)]

    @staticmethod
    def get_class_ids():
        return class_ids.copy()

    @staticmethod
    def get_class_names():
        return class_names.copy()

    @staticmethod
    def get_class_index_by_name(class_name):
        return class_names.index(class_name)

    @staticmethod
    def get_class_index_by_id(class_id):
        return class_ids.index(class_id)
