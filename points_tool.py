# -*- coding: utf-8 -*-

import numpy as np
import struct
import plyfile
import os,shutil,math
import subprocess
import time
import data_utils as du

#out_dir = du.getConf("obj_config")['data']


def load_pc_bin(pcbin_files,label,color,type="tuple"):

#    print "\nread pointsfile:",pcbin_files

    point_num = 0
    point_list = []
    normal_list = []
    color_list = []
    label_list = []

    #scan = np.fromfile(pcbin_files, dtype=np.float32)
    #scan_list.append(scan.reshape((-1, 4)))
    with open(pcbin_files,'rb') as f:

    	c = f.read(4)

    	point_num = struct.unpack('i',c)[0]

    	#print "points_num:",point_num

    	#print "read pts"

    	for i in range(point_num):

    		#print "read_point ",i,"/",point_num

    		c = f.read(4)

    		x = struct.unpack('f',c)[0]

    		c = f.read(4)

    		y = struct.unpack('f',c)[0]

    		c = f.read(4)

    		z = struct.unpack('f',c)[0]

    		#print [x,y,z]

    		if type == "tuple":

    		    point_list.append((x,y,z))

    		elif type == "list":

    			point_list.append([x,y,z])

    		label_list.append(label)
    		color_list.append(color) 
            

    	#print "read norms"

    	for i in range(point_num):

    		#print "read_normals ",i,"/",point_num

    		c = f.read(4)

    		nx = struct.unpack('f',c)[0]

    		c = f.read(4)

    		ny = struct.unpack('f',c)[0]

    		c = f.read(4)

    		nz = struct.unpack('f',c)[0]

    		#Print [nx,ny,nz]
    		if type == "tuple":
    			normal_list.append((nx,ny,nz))
    		elif type == "list":
    			normal_list.append([nx,ny,nz])
    		
    return point_num,point_list,color_list,normal_list,label_list

def save_ply(points, colors, filename):

    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_color = np.array([tuple(c) for c in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, dtype=vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write(filename)

    #print "save ply to",filename

def save_pts(pts,out_file):

    with open(out_file,"w") as f:

        for p in pts:

            f.writelines(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")


def save_obj(pts, out_file):

    with open(out_file, "w") as f:

        for p in pts:

            f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")

        
def save_seg(seg,out_file,offset):

    with open(out_file,"w") as f:

        for s in seg:

            f.writelines(str(s + offset) + "\n")
    
def pc_merge(pc_list,out_name,save_item=False,samplenum = 4096):

    du.checkmkdir(out_dir)

    pts_m = []
    seg_m = []
    label_m = []

    for k,pc in enumerate(pc_list):

        pts = pc[1]
        seg = pc[2]
        label = pc[3]

        pts_m = pts_m + pts
        seg_m = seg_m + seg
        label_m = label_m + label

        if save_item:

            save_ply(pts,seg,out_dir + out_name + "_item_" + str(k) + ".ply")

    #sample pts
    pts_s,index = du.list_random_sample(pts_m,samplenum)
    seg_s = du.list_sample(seg_m,index)
    label_s = du.list_sample(label_m,index)

    #show ply result
    save_ply(pts_s,seg_s,out_dir + "/plyshow/" + out_name + ".ply")

    #save pts file
    save_pts(pts_s,out_dir + "/pts/" + out_name + ".pts")

    #save seg file
    save_seg(label_s,out_dir + "/seg/" + out_name + ".seg")

def scan_pc_merge(pc_list,samplenum = 0):

    pts_m = []
    seg_m = []
    label_m = []

    for k,pc in enumerate(pc_list):

        pts = pc[0]
        seg = pc[1]
        label = pc[2]

        pts_m = pts_m + pts
        seg_m = seg_m + seg
        label_m = label_m + label
        
    #sample pts
    if samplenum != 0:
        pts_s,index = du.list_random_sample(pts_m,samplenum)
        seg_s = du.list_sample(seg_m,index)
        label_s = du.list_sample(label_m,index)

        return pts_s,seg_s,label_s

    else:

        return pts_m,seg_m,label_m


def pc_getbbox(pc):

    x = []
    y = []
    z = []

    for pts in pc:

        x.append(pts[0])
        y.append(pts[1])
        z.append(pts[2])

    boundary = [min(x),max(x),min(y),max(y),min(z),max(z)]

    return boundary

def pc_trans(pc,trans,isprint = True):

    if isprint:
        print ("pc trans:",trans)

    for pts in pc:

        pts[0] = pts[0] + trans[0]
        pts[1] = pts[1] + trans[1]
        pts[2] = pts[2] + trans[2]

    return pc

def pc_rotY(pc,boundary,rotY):

    cen_x = (boundary[0] + boundary[1])/2
    cen_y = (boundary[2] + boundary[3])/2
    cen_z = (boundary[4] + boundary[5])/2

    rot = math.pi*rotY/180.0

    print ("pc rotate:",rotY,rot,"center:",[cen_x,cen_y,cen_z])

    if rotY - 0.0 < 0.01:

        return pc

    else:
    
        for pts in pc:

            ptrx = pts[0] - cen_x
            ptry = pts[1] - cen_y
            ptrz = pts[2] - cen_z
    
            pts[0] = ptrx*math.cos(rot) - ptrz*math.sin(rot) + cen_x
            pts[1] = ptry + cen_y
            pts[2] = ptrx*math.sin(rot) +ptrz*math.sin(rot) + cen_z
    
        return pc

def pc_retrans(item_list,trans_list,rotate_list):

    print ("\nReTrans PC")
    print ("Trans:",trans_list)
    print ("RotateY:",rotate_list,"\n")

    Cabin_b = []
    Cabin_r = []

    for k,item in enumerate(item_list):

        trans = trans_list[k]
        rotY = rotate_list[k]

        print ("RT:",k)

        if k == 0:
            print ("Trans Cabin:",item[0][0][1],trans)
            pc = item[1]

            Cabin_b = pc_getbbox(pc)
            Cabin_r = rotY

            trans = [-Cabin_b[0] + trans[0] , -Cabin_b[2] + trans[1] ,-Cabin_b[4] + trans[2]]
            boundary = Cabin_b
            rotate = Cabin_r

        elif k == 1:
            print ("Trans Cabin_noroof:",item[0][0][1],trans)
            pc = item[1]

            trans = [-Cabin_b[0] + trans[0] , -Cabin_b[2]+ trans[1] ,-Cabin_b[4] + trans[2]]
            boundary = Cabin_b
            rotate = Cabin_r

        else:
            print ("Trans PC:",item[0][0][1],trans)
            pc = item[1]
            boundary = pc_getbbox(pc)
            trans = [-boundary[0]+ trans[0],-boundary[2]+ trans[1],-boundary[4]+ trans[2]]
            rotate = rotY

        print ("D:["+str(round(boundary[1]-boundary[0],3))+","+str(round(boundary[3]-boundary[2],3))+","+str(round(boundary[5]-boundary[4],3))+"]\n",)
        
        item[1] = pc_rotY(pc,boundary,rotate)
        item[1] = pc_trans(pc,trans)

    return item_list
    

def get_box_whl(x_min, y_min, z_min, x_max, y_max, z_max):
    box_w = z_max - z_min
    box_h = y_max - y_min
    box_l = x_max - x_min
    return box_w, box_h, box_l