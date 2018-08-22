# -*- coding: utf-8 -*-
import math

color = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(192,192,192),(255,97,0),(255,128,0),(250,128,114),(135,38,87),(30,144,255),(128,42,42),(0,201,87),(160,82,45),(255,127,80),(128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128)]

def seg_color_ori(color_id):

    return color[color_id]

def seg_color(color_id):

    r = color[color_id][0]*1.0/255
    g = color[color_id][1]*1.0/255
    b = color[color_id][2]*1.0/255

    return [r,g,b]

def seg_color_map_len():

    return len(color)

def id_to_color(c_id):

    b = c_id%256
    g = c_id//256
    r = c_id//(256*256)

    #print(r,g,b)

    r = r*1.0/255
    g = g*1.0/255
    b = b*1.0/255

    return [r,g,b]

def color_to_id(color):

    r = round(color[0]*255,0)
    g = round(color[1]*255,0)
    b = round(color[2]*255,0)

    return_id = int(b + g*256 + r*256*256)

    return return_id

def color_seg(seg_color):

    seg_list = []

    for seg_c in seg_color:

        if seg_c == (255,255,255):

            seg_list.append(0)

        else:

            seg_list.append(1)

    return seg_list

def color_seg_nearest(seg_color_list,max_d):

    seg_list = []
    abn_seg_list = []

    for i,seg_color in enumerate(seg_color_list):

        d_list = []

        for k,list_color in enumerate(color):

            d = math.sqrt((seg_color[0]-list_color[0])**2 + (seg_color[1]-list_color[1])**2 + (seg_color[2]-list_color[2])**2)

            d_list.append(d)

        min_d = min(d_list)

        if min_d > max_d:

            abn_seg_list.append(i)

        minindex = d_list.index(min_d)

        seg_list.append(minindex)

    return seg_list,abn_seg_list

def color_seg_nearest_limitclass(seg_color_list,class_list):

    seg_list = []
    abn_seg_list = []

    for i,seg_color in enumerate(seg_color_list):

        d_list = []

        for k,list_color in enumerate(color):

            d = math.sqrt((seg_color[0]-list_color[0])**2 + (seg_color[1]-list_color[1])**2 + (seg_color[2]-list_color[2])**2)

            d_list.append(d)

        minindex = d_list.index(min(d_list))

        if minindex not in class_list:

            abn_seg_list.append(i)

        seg_list.append(minindex)

    return seg_list,abn_seg_list
