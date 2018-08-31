 #-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
#import random,os,sys,math

import points_tool as ptst
import color_tool as color_t

import xml.dom.minidom
from xml.dom.minidom import Document


def getConf(conf,attr="None"):

    conf_file = "./config.xml"

    re = {}

    if os.path.exists(conf_file):

        dom = xml.dom.minidom.parse(conf_file)

        ele=dom.getElementsByTagName(conf)
    
        if len(ele) == 0:
    
            print("Config parse Error",conf_file,conf)

        else:

            if attr == "None":
     
                conf = ele[0]

                conf_data = str(conf.firstChild.data)

                re['data'] = conf_data

            else:

                conf = ele[0]

                conf_data = str(conf.firstChild.data)
                conf_attr = str(conf.getAttribute(attr))

                re['data'] = conf_data
                re['attr'] = conf_attr

    else:

        print("config.xml Not Found!")

        os._exit(0)

    return re

def checkmkdir(path):

    if os.path.exists(path):

        print("Check",path,", Exists\n")

    else:

        os.mkdir(path)

        print("Check",path,", Not Exists!")
        print("mkdir",path,"\n")


def strlist2float(strlist):

    flist = []

    str_c = strlist.strip().split("[")[1].split("]")[0]

    eles = str_c.split(",")

    for e in eles:

        flist.append(float(e))

    return flist

def floatlist2str(list_f):

    if len(list_f) == 0:

        lstr = ""

    else:

        lstr = "["
  
        for f in list_f:
  
            lstr = lstr + str(f) + ","
  
        lstr = lstr[:-1] + "]"

    return lstr

def printlist(lname,list):

    print(lname,",size",len(list),":")

    for k,i in enumerate(list):

        print(k,i)

def checklistrange(c_list,low_b,up_b):

    for k,v in enumerate(c_list):

        if v < low_b[k] or v > up_b[k]:

            return False

    return True

def list_dis(list_A,list_B):

    return np.sqrt(np.sum(np.square(np.array(list_A) - np.array(list_B))))

def list_mid(list_A,list_B):

    return ((np.array(list_A) + np.array(list_B))/2).tolist()

def list_sub(list_A,list_B):

    re_list = []

    for k in range(len(list_A)):

        re_list.append(list_A[k] - list_B[k])

    return re_list


def list_sub_nonzero(list_A,list_B):

    re_list = []
    nonzero_list = []

    for k in range(len(list_A)):

        sub = list_A[k] - list_B[k]
        re_list.append(sub)

        if sub != 0:

            nonzero_list.append(k)

    return re_list,nonzero_list

def dir(root,type = 'f',addroot = True):

    dirList = []
    fileList = []

    files = os.listdir(root)  

    for f in files:
        if(os.path.isdir(root + f)):  
            if addroot == True:
                dirList.append(root + f)
            else:
                dirList.append(f)

        if(os.path.isfile(root + f)):          
            if addroot == True:           
                fileList.append(root + f)
            else:
                fileList.append(f)

    if type == "f":
        fileList.sort()
        return fileList

    elif type == "d":
        dirList.sort()
        return dirList

    else:
        print("ERROR: TMC.dir(root,type) type must be [f] for file or [d] for dir")

        return 0


def save_points(scan_points, obj_bbox, out_path):
    box_x1 = obj_bbox[0]
    box_y1 = obj_bbox[2]
    box_z1 = obj_bbox[4]
    box_x2 = obj_bbox[1]
    box_y2 = obj_bbox[3]
    box_z2 = obj_bbox[5]
    box_l = box_x2 - box_x1
    box_w = box_z2 - box_z1

    box_points = [[box_x1, box_y1, box_z1],
                  [box_x1, box_y1, box_z1 + box_w],
                  [box_x1 + box_l, box_y1, box_z1 + box_w],
                  [box_x1 + box_l, box_y1, box_z1],
                  [box_x2, box_y2, box_z2],
                  [box_x2, box_y2, box_z2 - box_w],
                  [box_x2 - box_l, box_y2, box_z2 - box_w],
                  [box_x2 - box_l, box_y2, box_z2]]
    with open(out_path, "w") as f:
        for p in box_points:
            f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
        f.writelines("l " + "1 " + "2\n")
        f.writelines("l " + "2 " + "3\n")
        f.writelines("l " + "3 " + "4\n")
        f.writelines("l " + "4 " + "1\n")
        f.writelines("l " + "5 " + "6\n")
        f.writelines("l " + "6 " + "7\n")
        f.writelines("l " + "7 " + "8\n")
        f.writelines("l " + "8 " + "5\n")
        f.writelines("l " + "1 " + "7\n")
        f.writelines("l " + "2 " + "8\n")
        f.writelines("l " + "3 " + "5\n")
        f.writelines("l " + "4 " + "6\n")
        for p in scan_points:
            f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")

    # print("save_obj to", out_path)

def save_dataset(scan_points,scan_points_seg,scan_points_label,scene_name,pts_path,seg_path,ply_parh,save_ply = True,use_color_map = False):

    print("\nSave dataset:")
  
    # out_dir = getConf("obj_config")['data']
    out_dir = "./out"

    checkmkdir(out_dir + "/" + pts_path + "/")
    checkmkdir(out_dir + "/" + seg_path + "/")

    out_pts_f = out_dir + "/" + pts_path + "/" + scene_name + ".pts"
    out_obj_f = out_dir + "/" + pts_path + "/" + scene_name + ".obj"
    out_seg_f = out_dir + "/" + seg_path + "/" + scene_name + ".seg"

    if save_ply:
        
        checkmkdir(out_dir + "/" + ply_parh + "/")
        out_ply_f = out_dir + "/" + ply_parh + "/" + scene_name + ".ply"
        if use_color_map:
            seg_with_color_map = [color_t.seg_color_ori(label%color_t.seg_color_map_len()) for label in scan_points_label]
            ptst.save_ply(scan_points, seg_with_color_map, out_ply_f)
        else:
            ptst.save_ply(scan_points, scan_points_seg, out_ply_f)
      
    ptst.save_pts(scan_points,out_pts_f)
    print("save_pts to",out_pts_f)

    ptst.save_obj(scan_points, out_obj_f)
    print("save_obj to", out_obj_f)

    ptst.save_seg(scan_points_label,out_seg_f,0)
    print("save_seg to",out_seg_f)

# def getClass(obj_list,eletype):

#     obj_path = getConf("obj_path")[0]

#     class_list = []

#     with open(obj_list,"r") as f:

#         for line in f.readlines():

#             if line[0] != "#":

#                 line_s = line.strip().split(" ")
#                 c1 = line_s[0]
#                 c2 = reduce(lambda x,y:x+y + " ", line_s[1:]).strip()

#                 if line[0] == "@":
    
#                     if c1 == "@" + eletype:
    
#                         class_num = c2
    
#                         print eletype + " Type Num:",class_num
    
#                 else:

#                     if c1.split("_")[0] == eletype:

#                         class_list.append([c1,obj_path + c2])


#     return class_list,class_num


# def getObj(obj_list,eletype,eledata):

#     objtag = eletype +"_" + eledata
    
#     print "\nGet Obj:"
#     print  "ObjTag ",objtag

#     class_list,class_num = getClass(obj_list,eletype)

#     if eledata == "random":

#         eledata = str(random.randint(0,int(class_num) - 1))

#         print "random obj index: ",eledata

#     elif int(eledata) >= int(class_num):

#         eledata = str(random.randint(0,int(class_num) - 1))

#         print "WARNING: obj index out of range, random obj index: ",eledata

#     for obj in class_list:
    
#         if obj[0].split("_")[1] == eledata:
    
#             obj_path = obj[1]
#             reobj = obj
#             print "obj_path:",obj_path

#     return reobj





# def parseXml(xmlfile,obj_list):

#     dom = xml.dom.minidom.parse(xmlfile)
#     root = dom.documentElement
#     item_list = []

#     if root.nodeName != "Huawei_Computer_Room":
#         print "Need A Huawei Computer Room Xml File!"
#         os._exit(0)
#     else:
#         print "Parse XML File:"
    
#     #generate cabin
#     scene=dom.getElementsByTagName('scene')
    
#     if len(scene) == 0:
    
#         print "No Scene Tag, Random Scene:"
#         eletype = "Cabin"
#         eledata = "random"
#         eletrans = "[0,0,0]"
#         elerotateY = "0"
#         elelabel = "0"

    
#     else:
    
#         scene = scene[0]
    
#         eletype = scene.getAttribute("type")
#         eledata = scene.firstChild.data
#         eletrans = scene.getAttribute("trans")
#         elerotateY = scene.getAttribute("rotateY")
#         elelabel = scene.getAttribute("label")
    
#     cabin_obj = [eletype + "_" + eledata,getObj(obj_list,eletype,eledata),strlist2float(eletrans),float(elerotateY),int(elelabel)]
#     cabin_noroof_obj = [eletype + "_" + eledata,getObj(obj_list,cabin_obj[0].split("_")[0] + "Noroof",cabin_obj[0].split("_")[1]),strlist2float(eletrans),float(elerotateY),int(elelabel)]

#     #generate items
#     items = dom.getElementsByTagName('item')

#     if len(items) != 0:

#         for item in items:

#             eletype = item.getAttribute("type").strip()
#             eledata = item.firstChild.data.strip()
#             eletrans = item.getAttribute("trans").strip()
#             elerotateY = item.getAttribute("rotateY")
#             elelabel = item.getAttribute("label")

#             item_obj = getObj(obj_list,eletype,eledata)
#             item_list.append([eletype + "_" + eledata,item_obj,strlist2float(eletrans),float(elerotateY),int(elelabel)])

#     #get item
#     return [cabin_obj,cabin_noroof_obj],item_list

# def generate_xml(rootElement):

#     doc = Document() 

#     rootlabel = doc.createElement(rootElement)
#     doc.appendChild(rootlabel)

#     return doc

# def save_xml(doc,out_xml,save_encodeing = 'utf-8'):

#     f = open(out_xml,'w')
#     f.write(doc.toprettyxml(indent = '',encoding = save_encodeing))
#     f.close()

#     print "xml save:",out_xml

# def xml_rootadd(doc,add_label,attrs,add_inner):

#     label = doc.createElement(add_label)

#     for attr in attrs:

#         label.setAttribute(str(attr[0]),str(attr[1]))

#     inner = doc.createTextNode(str(add_inner))
#     label.appendChild(inner)
#     doc.childNodes[0].appendChild(label)


# def xml_addscene(doc,ltype,objid,trans,rot,lid):

#     xml_rootadd(doc,"scene",[["type",ltype],["trans",trans],["rotateY",rot],["label",lid]],objid)

# def xml_additem(doc,ltype,objid,trans,rot,lid):

#     xml_rootadd(doc,"item",[["type",ltype],["trans",trans],["rotateY",rot],["label",lid]],objid)



# def getobjinfo(obj_list,obj_info_list):

#     obj_root_path = getConf("obj_path")[0]

#     print "read obj info list"
    
#     scene_list = []
#     item_list = []
#     wall_item_list = []
    
#     with open(obj_list,'r') as f:
    
#         for line in f.readlines():
    
#             line_s = line.strip().split(" ")
    
#             if len(line_s) > 0 and line[0] != "#" and line[0] != "@":
    
#                 obj_cla = line_s[0].split("_")[0]
#                 obj_id = line_s[0].split("_")[1]
#                 obj_path = obj_root_path + line_s[1]
        
#                 if obj_cla == "Cabin":
        
#                     scene_list.append([obj_cla,obj_id,obj_path,0,0,0,0,0,0,0])
        
#                 elif obj_cla != "CabinNoroof":

#                     if obj_cla == "AC":
        
#                         wall_item_list.append([obj_cla,obj_id,obj_path,0,0,0,0,0,0,0])

#                     else:

#                         item_list.append([obj_cla,obj_id,obj_path,0,0,0,0,0,0,0])
    
#     with open(obj_info_list,'r') as f:
    
#         for line in f.readlines():
    
#             line_s = line.strip().split(" ")
    
#             if len(line_s) > 0 and line[0] != "#" and line[0] != "@":

#                 print line_s
    
#                 obj_cla = line_s[0].split("_")[0]
#                 obj_id = line_s[0].split("_")[1]
                

#                 if obj_cla == "Cabin":
    
#                     scene_trans = strlist2float(line_s[2])
#                     scene_dim = strlist2float(line_s[3])
#                     label_id = int(line_s[4])
                   
#                     for s in scene_list:
    
#                         if s[1] == obj_id:
    
#                             s[3] = scene_trans[0]
#                             s[4] = scene_trans[1]
#                             s[5] = scene_trans[2]
#                             s[6] = scene_dim[0]
#                             s[7] = scene_dim[1]
#                             s[8] = scene_dim[2]
#                             s[9] = label_id
    
#                 elif obj_cla != "CabinNoroof":
    
#                     item_dim = strlist2float(line_s[2])
#                     label_id = int(line_s[3])
    
#                     for i in item_list:
    
#                         if i[0] == obj_cla and i[1] == obj_id:
    
#                             i[6] = item_dim[0]
#                             i[7] = item_dim[1]
#                             i[8] = item_dim[2]
#                             i[9] = label_id

#                     for i in wall_item_list:
    
#                         if i[0] == obj_cla and i[1] == obj_id:
    
#                             i[6] = item_dim[0]
#                             i[7] = item_dim[1]
#                             i[8] = item_dim[2]
#                             i[9] = label_id

#     return scene_list,item_list,wall_item_list
        
# def region2map(region_list,map_dim,x_bin,y_bin):

#     region_map = np.zeros((map_dim,map_dim)) - 1

#     for k,region in enumerate(region_list):

#         for x in range(map_dim):
    
#             for y in range(map_dim):
    
#                 xt = x*x_bin + x_bin/2
#                 zt = y*y_bin + y_bin/2
    
#                 if xt > region[0] and xt < region[1] and zt > region[2] and zt < region[3]:
    
#                     region_map[y][x] = k

#     return region_map

# def region2map_strict(region_list,map_dim,x_bin,y_bin):

#     region_map = np.zeros((map_dim,map_dim)) - 1

#     for k,region in enumerate(region_list):

#         for y in range(map_dim):
    
#             for x in range(map_dim):
    
#                 xt1 = x*x_bin
#                 yt1 = y*y_bin

#                 xt2 = x*x_bin + x_bin
#                 yt2 = y*y_bin + y_bin

#                 l = max(xt1,region[0])
#                 r = min(xt2,region[1])
#                 t = max(yt1,region[2])
#                 b = min(yt2,region[3])

#                 if r - l > 0 and b - t > 0:

#                     region_map[y][x] = k


#     return region_map

# #def map2region(map):


# def list_random_sample(list_s,samplenum = 4096):

#     lens = len(list_s)
#     list_new = []

#     samplelist = random.sample(range(lens-1), samplenum)

#     for i in samplelist:

#         list_new.append(list_s[i])

#     return list_new,samplelist

# def list_sample(list_s,index):

#     list_new = []

#     for i in index:

#         list_new.append(list_s[i])

#     return list_new

# def read_pts_and_seg(pts_f,seg_f):

#     pts_list = []

#     with open(pts_f,"r") as f_p:

#         lines_pts = f_p.readlines()
        
#     with open(seg_f,"r") as f_s:

#         lines_seg = f_s.readlines()

#     num_pts = len(lines_pts)
#     num_seg = len(lines_seg)

#     if num_pts == num_seg:

#         print "read pts:",pts_f,num_pts,"read seg:",seg_f,num_seg

#     else:

#         print "Read Error: num_pts != num_seg"

#     for i in range(num_pts):

#         line_p = lines_pts[i]
#         iine_s = lines_seg[i]


#         line_ps = line_p.strip().split(" ")
#         line_ss = iine_s.strip().split(" ")

#         pts_list.append([float(line_ps[0]),float(line_ps[1]),float(line_ps[2]),int(line_ss[0])])

#     return pts_list

# def read_free_point(xml_f):

#     free_point_list = []

#     dom = xml.dom.minidom.parse(xml_f)

#     free_point=dom.getElementsByTagName('free_point')[0]

#     free_point_str = free_point.firstChild.data.strip().replace('[', '').replace(']','').split(",")

#     print 

#     for p in range(len(free_point_str)/2):

#         free_point_list.append([float(free_point_str[2*p]),float(free_point_str[2*p+1])])

#     return free_point_list

# def read_center_point(xml_f):


#     dom = xml.dom.minidom.parse(xml_f)

#     center_point_dom=dom.getElementsByTagName('cabin_center')[0]

#     center_point_str = center_point_dom.firstChild.data.strip().replace('[', '').replace(']','').split(",")

#     return [float(center_point_str[0]),float(center_point_str[1]),float(center_point_str[2])]

# def FarthestSampling(point_list,sample_num):

#     sample_list = []

#     first_p = point_list[random.randint(0,len(point_list) - 1)]

#     sample_list.append(first_p)

#     for i in range(sample_num-1):

#         d_sum_list = []

#         for pt in point_list:

#             d_sum = 0

#             for pt_s in sample_list:

#                 d = math.sqrt((pt_s[0]-pt[0])**2 + (pt_s[1]-pt[1])**2)

#                 d_sum = d_sum + d

#             d_sum_list.append(d_sum)

        
#         sample_list.append(point_list[d_sum_list.index(max(d_sum_list))])

#         #print max(d_sum_list),d_sum_list.index(max(d_sum_list))

#     return sample_list

# def listdel(s_list,del_index):

#     n_list = []

#     for k,ele in enumerate(s_list):

#         if k not in del_index:

#             n_list.append(ele)

#     print "list del",len(s_list),len(del_index),len(n_list)

#     return n_list

# def getclasslist(obj_info_list):

#     label_list = []
   
#     with open(obj_info_list,'r') as f:
    
#         for line in f.readlines():
    
#             line_s = line.strip().split(" ")
    
#             if len(line_s) > 0 and line[0] != "#" and line[0] != "@":
    
#                 label_list.append(int(line_s[-1]))
                
#     return label_list

# def save_dataset(scan_points,scan_points_seg,scene_name,pts_path,seg_path,ply_parh,save_ply = True):

#     print "\nSave dataset:"
    
#     out_dir = getConf("out_dir")[0]

#     out_pts_f = out_dir + "/" + pts_path + "/" + scene_name + ".pts"
#     out_seg_f = out_dir + "/" + seg_path + "/" + scene_name + ".seg"

#     class_list = getclasslist(getConf("obj_info")[0])

#     seg_label,abn_seg_list = ptst.color_seg_nearest_limitclass(scan_points_seg,class_list)

#     if len(abn_seg_list) != 0:

#         seg_label_d = listdel(seg_label,abn_seg_list)
#         scan_points_d = listdel(scan_points,abn_seg_list)

#         if save_ply:

#             out_ply_f = out_dir + "/" + ply_parh + "/" + scene_name + ".ply"
#             scan_points_seg_d = listdel(scan_points_seg,abn_seg_list)
#             ptst.save_ply(scan_points_d, scan_points_seg_d, out_ply_f)

#         ptst.save_pts(scan_points_d,out_pts_f)
#         print "save_pts to",out_pts_f
    
#         ptst.save_seg(seg_label_d,out_seg_f,1)
#         print "save_seg to",out_seg_f

#     else:    

#         if save_ply:
        
#             out_ply_f = out_dir + "/" + ply_parh + "/" + scene_name + ".ply"
#             ptst.save_ply(scan_points, scan_points_seg, out_ply_f)
            
#         ptst.save_pts(scan_points,out_pts_f)
#         print "save_pts to",out_pts_f
    
#         ptst.save_seg(seg_label,out_seg_f,1)
#         print "save_seg to",out_seg_f
