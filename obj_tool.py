from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct

import numpy as np
import data_utils as du

# out_dir = du.getConf("obj_config")['data']
out_dir = ""


def load_ply(path):
    """
    Loads 3D mesh model from a PLY file.

    :param path: A path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3  # Only triangular faces are supported
    face_n_texcoords = 6  # no use, only for loading data
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r')  # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split(' ')[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split(' ')[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'):  # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split(' ')[-1], line.split(' ')[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split(' ')
            if elems[4] == 'vertex_indices':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            elif elems[4] == 'texcoord':
                # (name of the property, data type)
                face_props.append(('n_tex', elems[2]))
                for i in range(face_n_texcoords):
                    face_props.append(('tex_' + str(i), elems[3]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.int)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    formats = {  # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(val))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(int(elems[prop_id])))
                        exit(-1)
                if prop[0] == 'n_tex':
                    if int(elems[prop_id]) != face_n_texcoords:
                        print('Error: Only 6 texcoord are supported.')
                        print('Number of face texcoord: ' + str(int(elems[prop_id])))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])
        # print(face_id, model['faces'][face_id])

    f.close()

    return model


"""
Loads 3D mesh model from a OBJ file.

:param path: A path to a PLY file.
:return: The loaded model given by a dictionary with items:
'pts' (nx3 ndarray), 'faces' (mx3 ndarray) - the latter three are optional.
"""


def load_obj(path):
    print("Load Obj File:", path)

    obj_f = open(path, 'r')
    obj_lines = obj_f.readlines()

    n_pts = 0
    n_faces = 0

    pts = []
    faces = []

    # face_n_corners = 3int triangular faces are supported now.
    # is_binary = False Only Non binary are supported now.

    n_sub_faces = 0
    # Read Obj File
    for line in obj_lines:

        if line[0] != '#':

            # pts
            line_s = line.strip().split(' ')
            if line_s[0] == 'v':
                pts.append([float(line_s[1]), float(line_s[2]), float(line_s[3])])

            # face
            if line_s[0] == 'f':
                line_s = line.strip().split(' ')
                if '/' in line_s[2]:
                    faces.append([int(line_s[2].split('/')[0]) - 1,
                                  int(line_s[3].split('/')[0]) - 1,
                                  int(line_s[4].split('/')[0]) - 1])
                    n_sub_faces += 1
                else:
                    faces.append([int(line_s[1]) - 1, int(line_s[2]) - 1, int(line_s[3]) - 1])

        else:

            if 'vertex positions' in line:
                line_s = line.strip().split(' ')
                n_pts += int(line_s[1])
            if 'faces' in line:
                print("len_faces += %d" % n_sub_faces)
                n_sub_faces = 0
                line_s = line.strip().split(' ')
                n_faces += int(line_s[4])
                print("n_faces += %d" % int(line_s[4]))

    # check read
    if n_pts == len(pts) and n_faces == len(faces):
        print("Check Read: Read Success(pts:" + str(n_pts) + ",faces:" + str(n_faces) + ")")
        n_faces = len(faces)
    else:
        print("Check Read: Read Error. n_pts = %d, len_pts = %d, n_faces = %d, len_faces = %d" %
              (n_pts, len(pts), n_faces, len(faces)))

    # Prepare data structures
    model = {}

    model['pts'] = np.array(pts)
    model['faces'] = np.array(faces)

    obj_f.close()

    return model


def load_labeled_obj(path, label, color):
    print("Load Obj File:", path)

    obj_f = open(path, 'r')
    obj_lines = obj_f.readlines()

    n_pts = 0
    n_faces = 0

    pts = []
    faces = []
    colors = []
    labels = []

    # face_n_corners = 3int triangular faces are supported now.
    # is_binary = False Only Non binary are supported now.

    # Read Obj File
    for line in obj_lines:

        if line[0] != '#':

            # pts
            if line[0] == 'v':
                line_s = line.strip().split(' ')
                pts.append([float(line_s[1]), float(line_s[2]), float(line_s[3])])

                colors.append(color)
                labels.append(label)

            # face
            if line[0] == 'f':
                line_s = line.strip().split(' ')
                faces.append([int(line_s[1]) - 1, int(line_s[2]) - 1, int(line_s[3]) - 1])

        else:

            if line.startswith('# Vertices'):
                line_s = line.strip().split(' ')
                n_pts = int(line_s[2])

            if line.startswith('# Face'):
                line_s = line.strip().split(' ')
                n_faces = int(line_s[2])

    # check read
    if n_pts == len(pts) and n_faces == len(faces):
        print("Check Read: Read Success(pts:" + str(n_pts) + ",faces:" + str(n_faces) + ")")
    else:
        print("Check Read: Read Error")

    # Prepare data structures
    model = {}

    model['pts'] = np.array(pts)
    model['faces'] = np.array(faces)
    model['colors'] = np.array(colors)

    obj_f.close()

    return model, labels


def format_obj(vertexs, faces, label, color):
    colors = []
    labels = []
    model = {}

    for v in vertexs:
        colors.append(color)
        labels.append(label)

    # Prepare data structures
    model['pts'] = np.array(vertexs)
    model['faces'] = np.array(faces.astype(np.int32))
    model['colors'] = np.array(colors)

    return model, labels


def save_obj(model, path):
    print("Save Obj File:", path)

    obj_f = open(path, 'w')

    pts = model['pts']
    faces = model['faces']

    n_pts = len(pts)
    n_faces = len(faces)

    # write head
    obj_f.writelines("####\n#\n")
    obj_f.writelines("# OBJ File Generated by mcsun obj tool\n")
    obj_f.writelines("#\n####\n")

    obj_f.writelines("#\n")
    obj_f.writelines("# Vertices: " + str(n_pts) + "\n")
    obj_f.writelines("# Faces: " + str(n_faces) + "\n")
    obj_f.writelines("#\n####\n")

    # write v
    for pt in pts:
        obj_f.writelines("v " + str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n")

    obj_f.writelines("# " + str(n_pts) + " vertices, 0 vertices normals\n\n")

    # write f
    for face in faces:
        obj_f.writelines("f " + str(face[0] + 1) + " " + str(face[1] + 1) + " " + str(face[2] + 1) + "\n")

    obj_f.writelines("# " + str(n_faces) + " faces, 0 coords texture\n\n")

    obj_f.writelines("# End of File\n")

    obj_f.close()


def obj_getbbox(pc):
    x = []
    y = []
    z = []

    for pts in pc:
        x.append(pts[0])
        y.append(pts[1])
        z.append(pts[2])

    boundary = [min(x), max(x), min(y), max(y), min(z), max(z)]

    return boundary


def obj_trans(model, trans, isprint=True):
    if isprint:
        print("obj trans:", trans)

    for pt in model['pts']:
        pt[0] = pt[0] + trans[0]
        pt[1] = pt[1] + trans[1]
        pt[2] = pt[2] + trans[2]

    return model


# some bugs
def obj_rotY(pc, boundary, rotY):
    cen_x = (boundary[0] + boundary[1]) / 2
    cen_y = (boundary[2] + boundary[3]) / 2
    cen_z = (boundary[4] + boundary[5]) / 2

    rot = math.pi * rotY / 180.0

    print("pc rotate:", rotY, rot, "center:", [cen_x, cen_y, cen_z])

    if rotY - 0.0 < 0.01:

        return pc

    else:

        for pts in pc:
            ptrx = pts[0] - cen_x
            ptry = pts[1] - cen_y
            ptrz = pts[2] - cen_z

            pts[0] = ptrx * math.cos(rot) - ptrz * math.sin(rot) + cen_x
            pts[1] = ptry + cen_y
            pts[2] = ptrx * math.sin(rot) + ptrz * math.sin(rot) + cen_z

        return pc


def obj_retrans(model_list, trans_list, rotate_list):
    print("\nReTrans Obj")
    print("Trans:", trans_list)
    print("RotateY:", rotate_list, "\n")

    Cabin_b = []
    Cabin_r = []

    # item[info,model,label]
    for k, item in enumerate(model_list):

        trans = trans_list[k]
        rotY = rotate_list[k]

        print("ReTrans Obj:", k)

        if k == 0:
            print("Trans Cabin:", item[0][0], trans)
            model = item[1]

            boundary = obj_getbbox(model['pts'])
            trans = [-boundary[0] + trans[0], -boundary[2] + trans[1], -boundary[4] + trans[2]]
            rotate = rotY

        else:
            print("Trans Item:", item[0][0], trans)
            model = item[1]
            boundary = obj_getbbox(model['pts'])
            trans = [-boundary[0] + trans[0], -boundary[2] + trans[1], -boundary[4] + trans[2]]
            rotate = rotY

        print("Item_Size:[" + str(round(boundary[1] - boundary[0], 3)) + "," + str(
            round(boundary[3] - boundary[2], 3)) + "," + str(round(boundary[5] - boundary[4], 3)) + "]\n", )

        # item[1] = obj_rotY(pc,boundary,rotate)
        item[1] = obj_trans(model, trans)

    return model_list


def obj_merge(model_list, out_name, is_save_item=False, is_save_obj=False):
    print("\nObj Merge")

    out_dir_obj = out_dir + "/scene_obj/"

    du.checkmkdir(out_dir_obj)

    pts_m = []
    face_m = []
    colors_m = []
    label_m = []

    for k, item in enumerate(model_list):

        # print ("Merge obj:",k,item[0])

        model = item[1]
        label = item[2]

        pts = model['pts'].tolist()
        colors = model['colors'].tolist()

        faces = model['faces'].tolist()
        faces_n = []

        for face in faces:
            faces_n.append([face[0] + len(pts_m) - 1, face[1] + len(pts_m) - 1, face[2] + len(pts_m) - 1])

        face_m = face_m + faces_n

        pts_m = pts_m + pts
        colors_m = colors_m + colors
        label_m = label_m + label

        if is_save_item:
            save_obj(model, out_dir_obj + out_name + "_item_" + str(k) + ".obj")

    model_m = {}

    model_m['pts'] = np.array(pts_m)
    model_m['faces'] = np.array(face_m)
    model_m['colors'] = np.array(colors_m)
    model_m['labels'] = np.array(label_m)

    # For Test
    if is_save_obj:
        save_obj(model_m, out_dir_obj + out_name + ".obj")

    return model_m
