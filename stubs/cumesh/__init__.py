"""Minimal stub so Mesh imports succeed without a compiled CuMesh build."""


class CuMesh:
    def __init__(self):
        self.num_boundaries = 0
        self.num_boundary_loops = 0
        self._vertices = None
        self._faces = None

    def init(self, vertices, faces):
        self._vertices = vertices
        self._faces = faces

    def get_edges(self):
        return None

    def get_boundary_info(self):
        self.num_boundaries = 0

    def get_vertex_edge_adjacency(self):
        return None

    def get_vertex_boundary_adjacency(self):
        return None

    def get_manifold_boundary_adjacency(self):
        return None

    def read_manifold_boundary_adjacency(self):
        return None

    def get_boundary_connected_components(self):
        return None

    def get_boundary_loops(self):
        self.num_boundary_loops = 0

    def fill_holes(self, max_hole_perimeter=None):
        return None

    def remove_faces(self, face_mask):
        return None

    def simplify(self, target, verbose=False, options=None):
        return None

    def read(self):
        return self._vertices, self._faces
