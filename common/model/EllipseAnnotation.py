from .Annotation import Annotation
from .BoxAnnotation import BoxAnnotation
import math

from loguru import logger


class EllipseAnnotation(BoxAnnotation):
    CV2_MODE = 0
    BBOX_MODE = 1

    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], degree=0, mode=BBOX_MODE, is_default_value=False):
        if mode == self.CV2_MODE:
            # Convert from OpenCV ellipse format (center, axes) to bounding box format
            center, sizes = ptStart, ptEnd
            ptStart = [center[0] - sizes[0] / 2, center[1] - sizes[1] / 2]
            ptEnd = [center[0] + sizes[0] / 2, center[1] + sizes[1] / 2]

        # source skip: default-mutable-arg
        super(EllipseAnnotation, self).__init__(ptStart, ptEnd, degree, is_default_value=is_default_value)

        # Angle in radians
        self.angle = Annotation.DEGREE_TO_RAD(degree)

        self.active_endpoint_idx = -1

    @classmethod
    def from_json(cls, annotations, offset=(0, 0)):
        anno = EllipseAnnotation()
        anno._from_json(annotations, offset)
        if 'angle' in annotations:
            anno.angle = annotations['angle']
        return anno

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'ellipse'
        anno['angle'] = self.angle
        return anno

    def major_axis_length(self):
        a = abs(self.ptEnd[0] - self.ptStart[0])
        b = abs(self.ptEnd[1] - self.ptStart[1])

        return a if a > b else b

    def minor_axis_length(self):
        a = abs(self.ptEnd[0] - self.ptStart[0])
        b = abs(self.ptEnd[1] - self.ptStart[1])

        return a if a < b else b

    def minor_radius_points(self):
        return self._major_or_minor_radius_points(False)

    def major_radius_points(self):
        return self._major_or_minor_radius_points(True)

    def major_and_minor_points(self):
        w, h = self.half_size()

        if w > h:
            major_dir = [w, 0]
            minor_dir = [0, h]
        else:
            major_dir = [0, h]
            minor_dir = [w, 0]

        radian = self.radian()
        if radian != 0:
            c = math.cos(radian)
            s = math.sin(radian)

            # Rotate major direction vector
            x = major_dir[0] * c - major_dir[1] * s
            y = major_dir[0] * s + major_dir[1] * c
            major_dir = [x, y]

            # Rotate minor direction vector
            x = minor_dir[0] * c - minor_dir[1] * s
            y = minor_dir[0] * s + minor_dir[1] * c
            minor_dir = [x, y]

        pt_center = self.center_point()
        major_points = [[pt_center[0] - major_dir[0], pt_center[1] - major_dir[1]],
                        [pt_center[0] + major_dir[0], pt_center[1] + major_dir[1]]]
        minor_points = [[pt_center[0] - minor_dir[0], pt_center[1] - minor_dir[1]],
                        [pt_center[0] + minor_dir[0], pt_center[1] + minor_dir[1]]]

        return major_points, minor_points

    def _major_or_minor_radius_points(self, major=True):

        w, h = self.half_size()
        if major:
            dir = [w, 0] if w > h else [0, h]
        else:
            dir = [0, h] if w > h else [w, 0]

        radian = self.radian()
        if radian != 0:
            c = math.cos(radian)
            s = math.sin(radian)

            # Rotate direction vector
            x = dir[0] * c - dir[1] * s
            y = dir[0] * s + dir[1] * c
            dir = [x, y]

        pt_center = self.center_point()
        return [[pt_center[0] - dir[0], pt_center[1] - dir[1]],
                [pt_center[0] + dir[0], pt_center[1] + dir[1]]]

    def is_point_on(self, pos, tol=8):
        # Unrotate point to local ellipse coordinate system
        local_pos = self.convert_to_local(pos)
        a = abs(self.ptEnd[0] - self.ptStart[0]) / 2
        b = abs(self.ptEnd[1] - self.ptStart[1]) / 2

        x = local_pos[0] / a
        y = local_pos[1] / b

        val = x * x + y * y - 1

        tol2 = tol * tol
        base_tol = tol2 / (a * a) + tol2 / (b * b)
        off_tol = tol * abs(local_pos[0]) / (a * a) + tol * abs(local_pos[1]) / (b * b)
        off_tol *= 2

        if base_tol - off_tol < abs(val) < base_tol + off_tol:
            return True

        return False

    def circumference(self):
        a = abs(self.ptEnd[0] - self.ptStart[0]) / 2
        b = abs(self.ptEnd[1] - self.ptStart[1]) / 2
        # Circle case
        if abs(a - b) < 1.0e-7:
            return math.pi * (a + b)

        # Ensure a is major axis, b is minor axis
        if a < b:
            a, b = b, a
        if b < 1.0e-7:
            return 4 * a

        # Series coefficients for ellipse circumference calculation
        coeff_list = [0.25, 0.046875, 0.01953125, 1.0681152343E-02 + 7.5E-13, 6.7291259765625E-03,
                      4.62627410888672E-03, 3.37529182434082E-03, 2.57102306932E-03 + 2.11E-15, 2.02349037863314E-03,
                      1.63396848074626E-03, 1.3470112E-03 + 0.623504E-10, 1.12952502189501E-03, 9.60764626611876E-04,
                      8.27188932350786E-04, 7.19654371145184E-04, 6.31805937E-04 + 1.67501E-13, 5.59115461697537E-04,
                      4.98285770262851E-04, 4.46869856295286E-04, 4.03020751646311001E-04, 3.65323232359666E-04,
                      3.32678129468022E-04, 3.04221257334888E-04, 2.79265607319136E-04, 2.57259477462388E-04,
                      2.37755707906253E-04, 2.20388778625035E-04, 2.04857554110962E-04, 1.90912137972017E-04,
                      1.78343755555526E-04, 1.66976892883542E-04, 1.56663134607288E-04, 1.47276293897348E-04,
                      1.3870853372036099E-04, 1.30867255385557E-04, 1.23672585673967E-04, 1.1705533446825599E-04,
                      1.10955324829242E-04, 1.05320019869966E-04, 1.00103387635780991E-04]

        E = 1 - math.pow(b / a, 2)
        F = 1
        ePower = E

        for coeff in coeff_list:
            F -= coeff * ePower
            ePower *= E

        return 2 * math.pi * a * F

    def area(self):
        a = abs(self.ptEnd[0] - self.ptStart[0])
        b = abs(self.ptEnd[1] - self.ptStart[1])
        return math.pi * a * b / 4

    def contain_point(self, pos):
        local_pos = self.convert_to_local(pos)

        a = abs(self.ptEnd[0] - self.ptStart[0]) / 2
        b = abs(self.ptEnd[1] - self.ptStart[1]) / 2

        x = local_pos[0] / a
        y = local_pos[1] / b

        return x * x + y * y <= 1 + 1.0e-7

    def snap_edit_points(self, pos, tol=8):
        local_pos = self.convert_to_local(pos)

        a = abs(self.ptEnd[0] - self.ptStart[0]) / 2
        b = abs(self.ptEnd[1] - self.ptStart[1]) / 2

        edit_points = [(-a, 0), (0, -b), (a, 0), (0, b)]

        for i, pnt in enumerate(edit_points):
            dist = self.square_dist(local_pos, pnt)
            if dist < tol * tol:
                return i

        return -1

    # Translate the endpoint of the ellipse
    def translate_endpoint(self, idx, offset):

        # Get rectangle corner coordinates
        left_top_point = self.start_point()
        right_bottom_point = self.end_point()
        mid_point = self.center_point()

        # Rotation angle in radians
        angle = self.angle

        # Distance between endpoints and length of the non-endpoint axis
        endpoint_length = right_bottom_point[0] - left_top_point[0]
        endpoint_width = right_bottom_point[1] - left_top_point[1]

        # Calculate coordinates of endpoint 0
        x_offset = round(0.5 * endpoint_length * math.cos(angle), 4)
        y_offset = round(0.5 * endpoint_length * math.sin(angle), 4)
        endpoint_0 = [mid_point[0] - x_offset, mid_point[1] - y_offset]

        # Calculate coordinates of endpoint 1
        endpoint_1 = [mid_point[0] + x_offset, mid_point[1] + y_offset]

        # Store the two ellipse endpoints (floating point values)
        endpoint_set = [endpoint_0, endpoint_1]

        # Get the inactive endpoint index
        idx_other = 0
        if idx == 0:
            idx_other = 1

        # ************************************ Start moving **************************************

        # Get new position of the moved endpoint
        new_endpoint = [endpoint_set[idx][0] + offset[0], endpoint_set[idx][1] + offset[1]]

        # Calculate new rotation angle (radians) - this is the most complex step
        new_angle = math.atan(
            (new_endpoint[1] - endpoint_set[idx_other][1]) / (new_endpoint[0] - endpoint_set[idx_other][0]))

        if new_angle < 0:
            logger.info('end!')

        # Calculate new distance between endpoints (distance formula)
        new_endpoint_length = (((new_endpoint[0] - endpoint_set[idx_other][0]) ** 2) +
                               ((new_endpoint[1] - endpoint_set[idx_other][1]) ** 2)) ** 0.5

        # Calculate new center point
        new_mid_point = [(new_endpoint[0] + endpoint_set[idx_other][0]) / 2,
                         (new_endpoint[1] + endpoint_set[idx_other][1]) / 2]

        # Calculate new top-left corner of bounding box
        new_left_top_point = [new_mid_point[0] - 0.5 * new_endpoint_length, new_mid_point[1] - 0.5 * endpoint_width]

        # Calculate new bottom-right corner of bounding box
        new_right_bottom_point = [new_mid_point[0] + 0.5 * new_endpoint_length, new_mid_point[1] + 0.5 * endpoint_width]

        # Update all ellipse annotation properties
        self.ptStart = new_left_top_point
        self.ptEnd = new_right_bottom_point
        self.angle = new_angle

        # Switch active endpoint if angle becomes negative
        if (new_endpoint[0] - endpoint_set[idx_other][0]) / (endpoint_set[idx][0] - endpoint_set[idx_other][0]) < 0:
            self.switch_active_endpoint()

    # Scale the non-endpoint axis of the ellipse
    def scale_ellipse_axis(self, direction, step_size=1):
        """
            direction: scaling direction; "long": increase length, "short": decrease length
            step_size: scaling step size
        """
        if direction == 'long':
            self.ptStart[1] -= step_size
            self.ptEnd[1] += step_size
        elif direction == 'short':
            self.ptStart[1] += step_size
            self.ptEnd[1] -= step_size