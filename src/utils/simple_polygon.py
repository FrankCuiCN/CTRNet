import pyclipper

from utils.exceptions import InvalidVerticesError


class SimplePolygon:

    def __init__(self, path):
        """
        Create a "cleaned" and "strictly-simple" (see clipper documentation) outer (holes filled) polygon/multipolygon
        constructed from the given path. Self-intersecting path is accepted.
        Please note that InvalidVerticesError will be returned if the input path follows any of the following:
            1. It composes of less than 3 vertices.
            2. The path can degrade into a curve, i.e., it encircles no area.
            3. (Additional to the original document.) The path becomes empty after processing, due to the
            CleanPolygons function.

        Parameters:
            path: an array like object (float) with shape (-1, 2).
        """
        # Scale up and round down to int.
        # noinspection PyArgumentList
        path_scaled = pyclipper.scale_to_clipper(path)  # Won't mutate the input. Returns a new list.

        # Initialize pyclipper and enable "StrictlySimple" (see clipper documentation).
        pc = pyclipper.Pyclipper()
        pc.StrictlySimple = True

        # Load path. Raise InvalidVerticesError if it's invalid.
        try:
            pc.AddPath(path=path_scaled, poly_type=pyclipper.PT_CLIP, closed=True)
            pc.AddPath(path=path_scaled, poly_type=pyclipper.PT_SUBJECT, closed=True)
        except pyclipper.ClipperException:
            raise InvalidVerticesError('The given vertices are invalid, check the docstring for details.')

        # Calculate the hierarchical structure of the path using non-zero rule.
        poly_tree = pc.Execute2(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Obtain only the exterior rings of the polygon (holes ignored) and "clean" (see clipper documentation) them.
        self.paths_scaled = pyclipper.CleanPolygons([child.Contour for child in poly_tree.Childs])
        self.paths_scaled = [x for x in self.paths_scaled if len(x) != 0]

        # Raise InvalidVerticesError if self.paths_scaled is empty.
        if len(self.paths_scaled) == 0:
            raise InvalidVerticesError('The given vertices are invalid, check the docstring for details.')

    def __len__(self):
        return len(self.paths_scaled)

    def area(self):
        # Calculate the area encircled by paths_scaled
        area_scaled = sum([pyclipper.Area(path) for path in self.paths_scaled])

        # Scale back and return the area.
        return pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(area_scaled))

    def contours(self):
        """
        Return:
            A list of lists (float) with shape (-1, 2).
        """
        # Scale back and return the contours (i.e. paths).
        return pyclipper.scale_from_clipper(self.paths_scaled)

    def iou_with(self, plg):
        """
        Parameters:
            plg: a SimplePolygon object.

        Return:
            The iou (float) with the given polygon.
        """

        # Initialize pyclipper.
        pc = pyclipper.Pyclipper()

        # Load the paths of the two polygons.
        pc.AddPaths(self.paths_scaled, pyclipper.PT_CLIP, True)
        pc.AddPaths(plg.paths_scaled, pyclipper.PT_SUBJECT, True)

        # Calculate the intersection of the two polygons.
        intersection = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Return the iou (= 0.) if there's no intersection.
        if len(intersection) == 0:
            return 0.

        # Otherwise calculate the union of the two polygons and thereafter return the iou.
        # No need to scale back, for the scale factors cancel out through division.
        union = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        return sum([pyclipper.Area(path) for path in intersection]) / sum([pyclipper.Area(path) for path in union])
