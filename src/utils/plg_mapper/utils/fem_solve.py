import numpy as np
import triangle as tr
from skfem import MeshTri, InteriorBasis, ElementTriP1, asm, condense, solve
from skfem.models.poisson import laplace

from .get_coefficients import get_coefficients


def get_segment(n):
    ids = np.arange(n, dtype='int64')
    return np.stack((ids, np.roll(ids, -1)), axis=1)


def polygon_area(vertices):
    x, y = vertices.T
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def separate_boundary_ids(mesh_segments, mesh_segment_markers):
    mesh_segment_markers = mesh_segment_markers[:, 0]
    boundary_ids_separated = []
    for idx in range(mesh_segment_markers.max()):
        boundary_ids_separated.append(np.unique(mesh_segments[mesh_segment_markers == (idx + 1)]))
    return boundary_ids_separated


def lin2d(pts, coefficients):
    x, y = pts.T
    a, b, c = coefficients
    return a * x + b * y + c


def fem_solve(vertices, corner_idxs):
    # Create mesh
    mesh = tr.triangulate(tri=dict(vertices=vertices,
                                   segments=get_segment(len(vertices)),
                                   segment_markers=np.arange(len(vertices), dtype='int64') + 1),
                          opts='qpa{:.10f}'.format(max(polygon_area(vertices) / 1e4, 1e-5)))
    mesh_vertices, mesh_triangles = mesh['vertices'], mesh['triangles']
    mesh_segments, mesh_segment_markers = mesh['segments'], mesh['segment_markers']

    # Get boundary info
    boundary_ids = np.unique(mesh_segments)
    boundary_ids_separated = separate_boundary_ids(mesh_segments, mesh_segment_markers)
    coefficients_x, coefficients_y = get_coefficients(vertices, corner_idxs)

    # Solver settings
    basis = InteriorBasis(mesh=MeshTri(mesh_vertices.T.copy(), mesh_triangles.T.copy()), elem=ElementTriP1())
    system_matrix = asm(laplace, basis)
    interior_ids = basis.complement_dofs(boundary_ids)

    # Solve x
    u = np.zeros(basis.N)
    for idx, tmp in enumerate(boundary_ids_separated):
        u[tmp] = lin2d(mesh_vertices[tmp], coefficients_x[idx])
    u_x = solve(*condense(system_matrix, np.zeros_like(u), u, interior_ids))

    # Solve y
    u = np.zeros(basis.N)
    for idx, tmp in enumerate(boundary_ids_separated):
        u[tmp] = lin2d(mesh_vertices[tmp], coefficients_y[idx])
    u_y = solve(*condense(system_matrix, np.zeros_like(u), u, interior_ids))

    mesh_vertices_mapped = np.stack((u_x, u_y), axis=1)
    return mesh_vertices, mesh_vertices_mapped
