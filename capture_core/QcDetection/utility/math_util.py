import math
import numpy as np


def square_legnth(vector):
    """
    square length of vector
    """
    return vector[0] * vector[0] + vector[1] * vector[1]


def length(vector):
    """
    length of vector
    """
    return math.sqrt(square_legnth(vector))


def normalize(vector):
    """
    normalize vector
    """
    vec_len = length(vector)
    if vec_len < 1.0e-8:
        return vector

    return (vector[0] / vec_len, vector[1] / vec_len)


def square_dist_between(point1, point2):
    """
    square distance between point1 and point2
    """
    vec = (point2[0] - point1[0], point2[1] - point1[1])
    return square_legnth(vec)


def distance_between(point1, point2):
    """
    distance between point1 and point2
    """
    return math.sqrt(square_dist_between(point1, point2))


def point_pair_with_max_dist(points, is_closed=False):
    """
    pair of points with largest distance
    is_closed: whether the points list is closed polygon
    return: [pt0, pt1]
    """
    idx_pair = point_pair_idx_with_max_dist(points, is_closed)

    if not idx_pair:
        return None

    return [points[idx_pair[0]], points[idx_pair[1]]]


def _points_idx_with_max_dist(points, is_closed=False):
    """
    is_closed: whether the points list is closed polygon
    """
    idx_pair = []
    max_dist = 0
    if is_closed:
        start_idx = int(len(points) * 0.3)
        end_idx = int(len(points) * 0.7) + 1
        for i in range(end_idx):
            end = min(len(points), end_idx + i)
            for j in range(start_idx + i, end):
                dist = square_dist_between(points[i], points[j])
                if dist > max_dist:
                    max_dist = dist
                    idx_pair = [i, j]
    else:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = square_dist_between(points[i], points[j])
                if dist > max_dist:
                    max_dist = dist
                    idx_pair = [i, j]

    # print(idx_pair)
    return idx_pair


def point_pair_idx_with_max_dist(points, is_closed=False):
    """
    pair of points with largest distance
    is_closed: whether the points list is closed polygon
    return: [pt0, pt1]
    """
    if len(points) == 0:
        return None
    if len(points) == 1:
        return [0, 0]
    if len(points) == 2:
        return [0, 1]

    # len(points) < 20
    if len(points) < 20:
        return _points_idx_with_max_dist(points, is_closed)
    else:
        # divide into two parts
        pt_min, pt_max = boundingbox(points)
        dx = pt_max[0] - pt_min[0]
        dy = pt_max[1] - pt_min[1]

        # dx, dy区别不大的情况下，无法确定最远的两个点分别位于哪一端
        if dx < dy * 1.1 and dy < dx * 1.1:
            return _points_idx_with_max_dist(points, is_closed)

        max_dist = 0
        idx_pair = []

        left_points_idx = []
        right_points_idx = []
        if dx > dy:
            delta = min(dx / 5, 100)
            left_x = pt_min[0] + delta
            right_x = pt_max[0] - delta
            for i, pt in enumerate(points):
                if pt[0] < left_x:
                    left_points_idx.append(i)
                elif pt[0] > right_x:
                    right_points_idx.append(i)
        else:
            delta = min(dy / 5, 100)
            upper_y = pt_min[1] + delta
            bottom_y = pt_max[1] - delta
            for i, pt in enumerate(points):
                if pt[1] < upper_y:
                    left_points_idx.append(i)
                elif pt[1] > bottom_y:
                    right_points_idx.append(i)

        # find points with largest distance
        for i in left_points_idx:
            for j in right_points_idx:
                dist = square_dist_between(points[i], points[j])
                if dist > max_dist:
                    max_dist = dist
                    idx_pair = [i, j]

    # print(idx_pair)
    return idx_pair


def point_pair_with_min_dist(points, line=None):
    """
    if line is specified, return point pair with minimum distance along the line
    line: [point, dir]
    """
    if len(points) <= 2:
        return points

    if line:
        p0, dir = line
        pos_t = []
        neg_t = []
        for pt in points:
            if not pt:
                continue
            vec = vec_subtract(pt, p0)
            t = dot_product(vec, dir)

            if t < 0:
                neg_t.append((t, pt))
            else:
                pos_t.append((t, pt))

        if len(neg_t) == 0:
            pos_t.sort(key=lambda x: x[0])
            return [pos_t[0][1], pos_t[1][1]]
        elif len(pos_t) == 0:
            neg_t.sort(key=lambda x: x[0], reverse=True)
            return [neg_t[0][1], neg_t[1][1]]

        t0 = min(pos_t, key=lambda x: x[0])
        t1 = max(neg_t, key=lambda x: x[0])

        return [t0[1], t1[1]]
    else:
        idx_pair = point_pair_idx_with_min_dist(points)
        return [points[idx_pair[0]], points[idx_pair[1]]]


def point_pair_idx_with_min_dist(points):
    if len(points) == 0:
        return None
    if len(points) == 1:
        return [0, 0]
    if len(points) == 2:
        return [0, 1]

    min_dist = 1.0e10
    idx_pair = [0, 1]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = square_dist_between(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                idx_pair = [i, j]

    return idx_pair


def minor_axis(major_axis, contour, minor_dir=None):
    """
    major_axis: [point0, point1], two end points
    contour: list of points
    if minor_dir is None, set dir to be normal of major_axis
    """
    # orthogonal
    dir = vec_subtract(major_axis[1], major_axis[0])
    total_len = length(dir)
    # normalize
    dir = (dir[0] / total_len, dir[1] / total_len)

    if not minor_dir:
        # orthogonal to major axis
        minor_dir = (-dir[1], dir[0])

    max_dist = 0
    minor_axis = None
    # coarse: every 10 pixel
    if total_len > 20:
        # find largest one
        cur_len = 10

        start_len = 0
        while cur_len < total_len:
            pt_center = [major_axis[0][0] + cur_len * dir[0], major_axis[0][1] + cur_len * dir[1]]
            inter_pts = line_intersect_with_polygon([pt_center, minor_dir], contour, True)
            if inter_pts:
                dist = square_dist_between(inter_pts[0], inter_pts[1])
                if dist > max_dist:
                    max_dist = dist
                    start_len = cur_len - 10
                    minor_axis = inter_pts
            cur_len += 10

        start_pt = [major_axis[0][0] + start_len * dir[0], major_axis[0][1] + start_len * dir[1]]
        refine_len = min(20, total_len - start_len)
    else:
        start_pt = major_axis[0]
        refine_len = total_len

    # refine: every 2 pixel
    cur_len = 1
    while cur_len < refine_len:
        pt_center = (start_pt[0] + cur_len * dir[0], start_pt[1] + cur_len * dir[1])
        inter_pts = line_intersect_with_polygon([pt_center, minor_dir], contour, True)
        if inter_pts:
            dist = square_dist_between(inter_pts[0], inter_pts[1])
            if dist > max_dist:
                max_dist = dist
                minor_axis = inter_pts
        cur_len += 2

    return minor_axis


def point_with_max_dist(pt, points):
    """
    point of points with max distance with pt
    points: list of point
    return: pt, dist
    """
    dists = [square_dist_between(pt, point) for point in points]
    idx = np.argmax(dists)
    return points[idx], math.sqrt(dists[idx])


def point_with_min_dist(pt, points, unsorted_or_polyine_or_polygon=0):
    """
    unsorted_or_polyine_or_polygon: 0-unsorted, 1-polyline, 2-polygon
    """
    dists = [square_dist_between(pt, point) for point in points]
    idx = np.argmin(dists)

    if unsorted_or_polyine_or_polygon == 0:
        return points[idx], math.sqrt(dists[idx])

    if idx == 0:
        if unsorted_or_polyine_or_polygon == 2:
            point, dist = min(point_dist_of_point_to_lineseg(pt, [points[0], points[1]]),
                              point_dist_of_point_to_lineseg(pt, [points[0], points[-1]]))
        else:
            point, dist = point_dist_of_point_to_lineseg(pt, [points[0], points[1]])
    elif idx == len(points) - 1:
        if unsorted_or_polyine_or_polygon == 2:
            point, dist = min(point_dist_of_point_to_lineseg(pt, [points[idx], points[0]]),
                              point_dist_of_point_to_lineseg(pt, [points[idx], points[idx - 1]]))
        else:
            point, dist = point_dist_of_point_to_lineseg(pt, [points[idx], points[idx - 1]])
    else:
        point, dist = min(point_dist_of_point_to_lineseg(pt, [points[idx], points[idx + 1]]),
                          point_dist_of_point_to_lineseg(pt, [points[idx], points[idx - 1]]))

    return point, dist


def avg_point(point_list):
    """
    barycenter of point list
    """
    num_points = len(point_list)
    if num_points == 1:
        return point_list[0]

    sx = 0
    sy = 0
    for pt in point_list:
        sx += pt[0]
        sy += pt[1]

    return [sx / num_points, sy / num_points]


def avg_point3(pt0, pt1, pt2):
    """
    optimize for 3 points
    """
    sx = pt0[0] + pt1[0] + pt2[0]
    sy = pt0[1] + pt1[1] + pt2[1]

    return [sx / 3, sy / 3]


def mid_point(point1, point2):
    """
    middle point between point1 and point2
    """
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]


def insert_points(point_a, point_b):
    # 计算需要插入的两个点的坐标
    x1, y1 = point_a
    x2, y2 = point_b
    x_mid1 = (2 * x1 + x2) / 3
    y_mid1 = (2 * y1 + y2) / 3
    x_mid2 = (x1 + 2 * x2) / 3
    y_mid2 = (y1 + 2 * y2) / 3

    # 插入两个点
    insert_point1 = [x_mid1, y_mid1]
    insert_point2 = [x_mid2, y_mid2]

    return insert_point1, insert_point2


def insert_three_points(point_a, point_b):
    # 计算需要插入的三个点的坐标
    x1, y1 = point_a
    x2, y2 = point_b
    x_mid1 = (2 * x1 + x2) / 3
    y_mid1 = (2 * y1 + y2) / 3
    x_mid2 = (x1 + x2) / 2
    y_mid2 = (y1 + y2) / 2
    x_mid3 = (x1 + 2 * x2) / 3
    y_mid3 = (y1 + 2 * y2) / 3

    # 插入三个点
    insert_point1 = [x_mid1, y_mid1]
    insert_point2 = [x_mid2, y_mid2]
    insert_point3 = [x_mid3, y_mid3]

    return insert_point1, insert_point2, insert_point3


def ortho_vec(start, end, normalization=False):
    """
    vector which is orthogonal to (start->end)
    """
    dx, dy = vec_subtract(end, start)
    vec = [-dy, dx]
    if normalization:
        vec = normalize(vec)
    return vec


def normalized_ortho_vec(start, end):
    """
    vector which is orthogonal to (start->end)
    """
    return ortho_vec(start, end, normalization=True)


def dot_product(vec1, vec2):
    """
    """
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    return dot


def cross_product(vec1, vec2):
    """
    norm of the cross product between vec1 and vec2
    which is equivalient to |vec1| * |vec2| * sin()
    """
    val = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return val


def vec_add(vec1, vec2):
    """
    vec1 + vec2
    """
    return [vec1[0] + vec2[0], vec1[1] + vec2[1]]


def vec_subtract(vec1, vec2):
    """
    vec1 - vec2
    """
    return [vec1[0] - vec2[0], vec1[1] - vec2[1]]


def rotate_points(points, center, angle, in_degree=True):
    """
    rotate points around center with degree
    """
    radian = angle
    if in_degree:
        radian = math.radians(angle)
    if abs(angle) < 0.001:
        return points

    c = math.cos(radian)
    s = math.sin(radian)

    results = []
    for pt in points:
        vec = vec_subtract(pt, center)
        x = vec[0] * c - vec[1] * s + center[0]
        y = vec[0] * s + vec[1] * c + center[1]

        results.append([x, y])

    return results


def rotate_point(pt, center, angle, in_degree=True):
    """
    rotate points around center with degree
    """
    vec = vec_subtract(pt, center)
    vec = rotate_vec(vec, angle, in_degree)
    return vec_add(vec, center)


def rotate_vec(vec, angle, in_degree=True):
    """
    rotate points around center with degree
    """
    radian = angle
    if in_degree:
        radian = math.radians(angle)

    c = math.cos(radian)
    s = math.sin(radian)

    x = vec[0] * c - vec[1] * s
    y = vec[0] * s + vec[1] * c

    return [x, y]


def translate_point(point, offset):
    return vec_add(point, offset)


def translate_and_scale_point(point, offset, scale):
    x = (point[0] - offset[0]) * scale
    y = (point[1] - offset[1]) * scale
    return [x, y]


def mirror_along_point(point, center):
    """
    mirror along center: 2 * center - point
    """
    return [center[0] * 2 - point[0], center[1] * 2 - point[1]]


def mirror_along_line(point, line):
    """
    mirror along line
    line: (point, dir)
    """
    pt, dir = line

    dir = normalize(dir)
    vec = vec_subtract(point, pt)

    dist = cross_product(vec, dir) * 2

    normal = (-dir[1] * dist, dir[0] * dist)
    point = vec_add(point, normal)

    return point


def angle_between(vec1, vec2, ignore_direction=False, in_degree=False):
    """
    angle between vec1 and vec2 in counter-clockwise if direction is not ignored
    in_degree: return angle in degree or radian
    return value: [0, pi] if ignore_direction and not in_degree else [-pi, pi]
    """
    cos = dot_product(vec1, vec2)
    sin = cross_product(vec1, vec2)

    angle = math.atan2(sin, cos)
    if ignore_direction:
        angle = abs(angle)

    if in_degree:
        angle = math.degrees(angle)

    return angle


def angle_between_points(point1, point2, point3, ignore_direction=False, in_degree=False):
    """
    angle between (point2 -> point1) and (point2 -> point3) in counter-clockwise if direction is not ignored
    in_degree: return angle in degree or radian
    return value: [0, pi] if ignore_direction and not in_degree else [-pi, pi]
    """
    vec1 = vec_subtract(point1, point2)
    vec2 = vec_subtract(point3, point2)

    return angle_between(vec1, vec2, ignore_direction, in_degree)


def degree_to_rad(degree):
    return math.radians(degree)


def rad_to_degree(rad):
    return math.degrees(rad)


def is_obtuse(angle, in_degree=True):
    """
    whether is obtuse
    """
    return abs(angle) > 90 if in_degree else abs(angle) > math.pi / 2


def kb_to_point_dir(k, b):
    """
    convert line from y = kx + b to p(t) = p0 + t * dir
    return: (p0, dir)
    """
    if k == float('inf'):
        p0 = [b, 0]
        dir = [0, 1]
    else:
        p0 = [0, b]
        dir = [1, k]

    return (p0, dir)


def lineseg_to_line(point1, point2):
    """
    convert line from lineseg = (p0, p1) to p(t) = p0 + t * dir
    return: (p0, dir)
    """
    dir = [point2[0] - point1[0], point2[1] - point1[1]]
    return (point1, dir)


def normal_line(line):
    """
    line: can be either (k, b) or (p0, dir)
    return: normal line: [p, dir]
    """
    p0, dir = line
    if isinstance(p0, (tuple, list, np.ndarray)):
        return [p0, [-dir[1], dir[0]]]

    k, b = line
    p0 = [0, b]
    dir = [k, -1]

    return [p0, dir]


def dist_of_point_to_line(point, line):
    """
    line: [point, dir]
    """
    dist = signed_dist_of_point_to_line(point, line)
    return abs(dist)


def signed_dist_of_point_to_line(point, line, is_normalized=False):
    """
    line: [point, dir]
    is_normalzied: dir of line is normalized
    sign of dist can be used to judge point is above or below the line
    dist > 0: below the line
    dist < 0: above the line
    """
    start, dir = line
    if not is_normalized:
        dir = normalize(dir)
    vec = vec_subtract(point, start)

    dist = cross_product(vec, dir)
    return dist


def dist_of_point_to_lineseg(point, lineseg):
    """
    lineseg: [p1, p2]
    """
    pt_start, pt_end = lineseg

    # normalize
    dir = vec_subtract(pt_end, pt_start)
    dir = normalize(dir)

    vec = vec_subtract(point, pt_start)

    # whether is between line segment
    proj = dot_product(vec, dir)
    # start point is closest
    if proj < 0:
        return distance_between(point, pt_start)

    # end point is closest
    if proj > distance_between(pt_start, pt_end):
        return distance_between(point, pt_end)

    # distance using cross product
    dist = cross_product(vec, dir)
    return abs(dist)


def point_dist_of_point_to_lineseg(point, lineseg):
    """
    lineseg: [p1, p2]
    return both dist and projection point
    """
    pt_start, pt_end = lineseg

    # normalize
    dir = vec_subtract(pt_end, pt_start)
    dir = normalize(dir)

    vec = vec_subtract(point, pt_start)

    # whether is between line segment
    proj = dot_product(vec, dir)
    # start point is closest
    if proj < 0:
        return pt_start, distance_between(point, pt_start)

    # end point is closest
    if proj > distance_between(pt_start, pt_end):
        return pt_end, distance_between(point, pt_end)

    # distance using cross product
    dist = cross_product(vec, dir)

    x = pt_start[0] + dir[0] * proj
    y = pt_start[1] + dir[1] * proj

    return [x, y], abs(dist)


def is_polygon_intersect_approximate(polygon0, polygon1):
    bbox0 = boundingbox(polygon0)
    bbox1 = boundingbox(polygon1)

    if not is_box_intersect(bbox0, bbox1):
        return False

    # point in another polygon
    for pt in polygon0:
        if point_is_contained_by(pt, bbox1):
            return True

    #
    return False


def line_intersect_with_polyline(line, polyline):
    """
    intersection points between line and polyline
    line: (point, dir)
    return: intersection points if exist
    """
    point, dir = line

    vec0 = vec_subtract(polyline[0], point)
    pre_cross = cross_product(vec0, dir)

    inter_points = []
    for i in range(1, len(polyline)):
        vec0 = vec_subtract(polyline[i], point)
        cross = cross_product(vec0, dir)

        # exist one intersection point
        # donot use multiply, since multiply result canbe overlow
        if pre_cross <= 0 and cross >= 0 or pre_cross >= 0 and cross <= 0:
            temp_line = (polyline[i], vec_subtract(polyline[i], polyline[i - 1]))
            inter_pt = line_intersect_with_line(line, temp_line)
            if inter_pt:
                inter_points.append(inter_pt)

        pre_cross = cross

    return inter_points


def line_intersect_with_polygon(line, polygon, keep_two_max=False, keep_two_min=False):
    """
    intersection points between line and polygon
    line: (point, dir)
    keep_two_max: return two intersection points with max distance
    keep_two_min: return two intersection points with min distance center with point
    """
    point, dir = line
    polygon = np.squeeze(polygon)

    pre_pt = polygon[-1]
    vec0 = vec_subtract(pre_pt, point)
    pre_cross = cross_product(vec0, dir)

    inter_points = []
    for pt in polygon:
        vec0 = vec_subtract(pt, point)
        cross = cross_product(vec0, dir)

        # exist one intersection point
        # donot use multiply, since multiply result canbe overlow
        if pre_cross <= 0 and cross >= 0 or pre_cross >= 0 and cross <= 0:
            temp_line = (pre_pt, vec_subtract(pt, pre_pt))
            inter_pt = line_intersect_with_line(line, temp_line)
            if inter_pt:
                inter_points.append(inter_pt)

        pre_pt = pt
        pre_cross = cross

    # return all intersection points
    if not keep_two_max and not keep_two_min:
        return inter_points

    if len(inter_points) < 2:
        return []
    if len(inter_points) == 2:
        return inter_points

    print(f'{len(inter_points)} intersection points')

    if keep_two_max:
        # choose 2 point with max distance
        max_dist = 0
        keep_points = []
        for i in range(len(inter_points)):
            for j in range(i + 1, len(inter_points)):
                dist = square_dist_between(inter_points[i], inter_points[j])
                if dist > max_dist:
                    max_dist = dist
                    keep_points = [inter_points[i], inter_points[j]]
        return keep_points

    if keep_two_min:
        # choose 2 point with min distance center with point
        return point_pair_with_min_dist(inter_points, line)

    return inter_points


def line_intersect_with_box(line, box):
    """
    line: format with (point, dir)
    box: format with xyxy(x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    return line_intersect_with_polygon(line, polygon)


def line_intersect_with_ellipse(line, ellipse):
    """
    line: format with (point, dir)
    ellipse: format with (center, axis, angle_in_degree)

    return: None or two intersection points
    """
    # rotate to align with axis
    rad = degree_to_rad(ellipse[2])

    pt_on_line = rotate_point(line[0], ellipse[0], -rad, in_degree=False)
    dir = rotate_vec(line[1], -rad, in_degree=False)

    return pt_on_line, dir


def split_convex_polygon_with_line(polygon, line, keep_two_part=False):
    """
    line: (point, dir)
    polygon: list of polygon points
    return: two polygon after splition
    """

    point, dir = line
    polygon = np.squeeze(polygon)

    pre_pt = polygon[-1]
    vec0 = vec_subtract(pre_pt, point)
    pre_cross = cross_product(vec0, dir)

    new_polygon = []
    another_polygon = []

    first_idx = -1
    for i, pt in enumerate(polygon):
        vec0 = vec_subtract(pt, point)
        cross = cross_product(vec0, dir)

        # exist one intersection point
        # donot use multiply, since multiply result canbe overlow
        if pre_cross <= 0 and cross >= 0 or pre_cross >= 0 and cross <= 0:
            temp_line = (pre_pt, vec_subtract(pt, pre_pt))
            inter_pt = line_intersect_with_line(line, temp_line)
            if inter_pt:
                new_polygon.append(inter_pt)
                if first_idx < 0:
                    new_polygon.append(pt)
                    first_idx = i

            # two intersection points at most
            if len(new_polygon) > 2:
                another_polygon.append(new_polygon[-1])
                for j in range(i, len(polygon)):
                    another_polygon.append(polygon[j])
                for j in range(first_idx):
                    another_polygon.append(polygon[j])
                another_polygon.append(new_polygon[0])
                break

        elif len(new_polygon) > 0:
            new_polygon.append(pt)

        pre_pt = pt
        pre_cross = cross

    return new_polygon, another_polygon


def split_box_with_line(box, line):
    """
    box: format with xyxy (x1, y1, x2, y2)
    line: format with (point, dir)
    """
    x1, y1, x2, y2 = box
    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    return split_convex_polygon_with_line(polygon, line)


def line_intersect_with_lineseg(line, lineseg):
    """
    line: (point, dir)
    lineseg: (start, end)
    """
    # whether has intersection point
    point, dir = line
    start, end = lineseg
    cross0 = cross_product(vec_subtract(start, point), dir)
    cross1 = cross_product(vec_subtract(end, point), dir)

    if cross0 <= 0 and cross1 >= 0 or cross0 >= 0 and cross1 <= 0:
        line1 = (start, vec_subtract(end, start))
        return line_intersect_with_line(line, line1)


def line_intersect_with_line(line0, line1):
    """
    line0: (point, dir)
    line1: (point, dir)
    """
    p0, d0 = line0
    p1, d1 = line1

    d = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(d) < 1.0e-6:
        return

    px, py = (p1[0] - p0[0], p1[1] - p0[1])
    t = (d1[1] * px - d1[0] * py) / d

    return [int(round(p0[0] + t * d0[0])), int(round(p0[1] + t * d0[1]))]


def lineseg_intersect_with_lineseg(lineseg0, lineseg1):
    """
    lineseg0: [start_point, end_point]
    lineseg1: [start_point, end_point]
    """
    # whether has intersection point between lingseg1 and line along lineseg0
    dir0 = vec_subtract(lineseg0[1], lineseg0[0])
    cross0 = cross_product(vec_subtract(lineseg1[0], lineseg0[0]), dir0)
    cross1 = cross_product(vec_subtract(lineseg1[1], lineseg0[0]), dir0)

    if cross0 < 0 and cross1 < 0 or cross0 > 0 and cross1 > 0:
        return None

    # whether has intersection point between lingseg0 and line along lineseg1
    dir1 = vec_subtract(lineseg1[1], lineseg1[0])

    cross0 = cross_product(vec_subtract(lineseg0[0], lineseg1[0]), dir1)
    cross1 = cross_product(vec_subtract(lineseg0[1], lineseg1[0]), dir1)

    if cross0 < 0 and cross1 < 0 or cross0 > 0 and cross1 > 0:
        return None

    return line_intersect_with_line([lineseg0[0], dir0], [lineseg1[0], dir1])


# /////////////////// bbox //////////////////////////


def is_contained_by(inner_box, outer_box):
    """
    inner_box, outer_box: [x1, y1, x2, y2]
    """
    if inner_box[0] >= outer_box[0] and inner_box[2] <= outer_box[2] and \
            inner_box[1] >= outer_box[1] and inner_box[3] <= outer_box[3]:
        return True

    return False


def point_is_contained_by(point, bbox):
    """
    whether point is contained by bbox
    bbox: [x1, y1, x2, y2]
    """
    if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
        return True

    return False


def boundingbox(points):
    """
    return: bbox with format xyxy
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    return [(min_x, min_y), (max_x, max_y)]


def is_box_intersect(box1, box2):
    """
    whether box1 is intersected with box2
    box1, box2: [x1, y1, x2, y2]
    """
    if box1[0] > box2[2] or box1[2] < box2[0] or \
            box1[1] > box2[3] or box1[3] < box2[1]:
        return False

    return True


def is_intersect(box1, box2):
    """
    same as is_box_intersect()
    """
    return is_box_intersect(box1, box2)


def box_area(bbox):
    """
    area of bounding box with form [x1, y1, x2, y2]
    """
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return abs(area)


def box_center(bbox):
    """
    center of bounding box [x1, y1, x2, y2]
    """
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    return (cx, cy)


def box_iou(box1, box2):
    """
    iou of box1 and box2: (x1, y1, x2, y2)
    """
    # when two boxes don't intersect
    # if not is_box_intersect(box1, box2):
    #     return 0

    x0 = max(box1[0], box2[0])
    x1 = min(box1[2], box2[2])
    y0 = max(box1[1], box2[1])
    y1 = min(box1[3], box2[3])

    # no intersection
    if x0 > x1 or y0 > y1:
        return 0

    intersect_area = box_area((x0, y0, x1, y1))
    united_area = box_area(box1) + box_area(box2) - intersect_area

    if united_area < 1.0e-8:
        return 0

    return intersect_area / united_area


def box_union(box1, box2):
    """
    union of box: (x1, y1, x2, y2)
    """
    x0 = min(box1[0], box2[0])
    x1 = max(box1[2], box2[2])
    y0 = min(box1[1], box2[1])
    y1 = max(box1[3], box2[3])

    return (x0, y0, x1, y1)


def is_contained_approximate(box1, box2, ratio_thresh=0.8):
    """
    小物体是否包含于大物体(近似)
    """
    # when two boxes don't intersect
    if not is_box_intersect(box1, box2):
        return False

    x0 = max(box1[0], box2[0])
    x1 = min(box1[2], box2[2])
    y0 = max(box1[1], box2[1])
    y1 = min(box1[3], box2[3])

    intersect_area = box_area((x0, y0, x1, y1))
    min_box_area = min(box_area(box1), box_area(box2))

    contain_ratio = intersect_area / (min_box_area + 1.0e-8)
    if contain_ratio > ratio_thresh:
        return True
    else:
        return False


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


# 使用Shoelace公式计算多边形面积
def polygon_area(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


if __name__ == '__main__':
    vec1 = (0, 10)
    vec2 = (-3.0, 293.5)
    angle = angle_between(vec1, vec2, in_degree=True)
    print(angle)

    line = [[0, 0], [2, 0]]
    point = [2, -2]

    point = mirror_along_line(point, line)
    print(point)

    line0 = ([0, 0], [1, 0])
    line1 = ([20, 0], [0, 1])
    print(line_intersect_with_line(line0, line1))
    print(line_intersect_with_lineseg(line0, ([20, 10], [20, -10])))
    print(lineseg_intersect_with_lineseg(([0, 0], [10, 0]), ([20, 10], [20, -10])))

    # vec1 = (2, 0)
    # vec2 = (0, 2)
    #
    # print(normalize(vec1))
    #
    # dist = distance_between([2, 3], [0, 1])
    # print(dist)
    #
    # angle = angle_between(vec1, vec2, in_degree=True)
    # print(angle)
    #
    # angle = angle_between(vec2, vec1, ignore_direction=True)
    # print(angle)
    #
    # angle = angle_between(vec2, vec1)
    # print(angle)

    bbox2 = [10, 10, 20, 20]
    bbox1 = [15, 15, 25, 25]

    # if is_contained_by(bbox1, bbox2):
    #     print("contained by")

    if is_box_intersect(bbox1, bbox2):
        print("is intersect")

    # iou = box_iou(bbox1, bbox2)
    # print(iou)

    split_box_with_line(bbox2, ((15, 15), vec1))
