 #-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from groundtruth import GetFootOfPerpendicular

import random
import numpy as np
import data_utils as du
import pipeline_tools as pl_t
import obj_tool as objt
import color_tool as color_t
import opengl_render_scanner as scanner

def builder_scene(g_begin,g_end,vscan_sample_rate,sample_rate):
    scene_file_path = du.getConf("scene_config")['data']
    obj_out_path = du.getConf("obj_config")['data']
    style_list_file = du.getConf("style_list")['data']
    #label_out_path=du.getConf("label_list")['data']
    #
    du.checkmkdir(obj_out_path)
    du.checkmkdir(obj_out_path+"train_label_seg/")
    du.checkmkdir(obj_out_path+"train_label_obj/")
    du.checkmkdir(obj_out_path+"seg/")
    du.checkmkdir(obj_out_path+"pts/")
    du.checkmkdir(obj_out_path+"axis_label_txt/")
    du.checkmkdir(obj_out_path+"axis_label_obj/")
    #read style
    dic_style = {}
    for style in [line.strip().split(" ") for line in open(style_list_file).readlines()]:
        dic_style[style[0]] = [[float(value) for value in p_list.split(",")] for p_list in style[1:]]
    style_num = len(dic_style.keys())

    #sample_rate=30
    iter_num=5
    #scan_point_num = 6

    scene_xml_list = du.dir(scene_file_path)
    scene_xml_list=scene_xml_list[g_begin:g_end]

    for scene_file in scene_xml_list:

        obj_out_file = obj_out_path + scene_file.strip().split("/")[-1].replace("txt","obj")
        label_txt_out_file = obj_out_path+"axis_label_txt/" + scene_file.strip().split("/")[-1].replace("txt","txt")
        label_obj_out_file=obj_out_path + "axis_label_obj/"+scene_file.strip().split("/")[-1].replace("txt","obj")

        #num_obj=len(scene_xml_list)
        #labels = obj_out_path+'axis_label_txt/'+str(i).zfill(6)+'.txt'
        segs = obj_out_path+'seg/'+scene_file.strip().split("/")[-1].replace("txt","seg")  
        pts = obj_out_path+'pts/'+scene_file.strip().split("/")[-1].replace("txt","pts")  
        outputs_seg = obj_out_path+'train_label_seg/'+scene_file.strip().split("/")[-1].replace("txt","seg")
        outputs_obj = obj_out_path+'train_label_obj/'+scene_file.strip().split("/")[-1].replace("txt","obj")  
        


        #print (label_obj_out_file)
        #print ('obj:',obj_out_file)
        #print('list:',label_txt_out_file)
        scene_name = scene_file.strip().split("/")[-1].split(".")[0]
        print('scene_name',scene_name)
#   
        #print(obj_out_file)

        #read scene_file
        v_dic = {}
        v_list = [[float(v) for v in ele.strip().split(" ")[1:]] for ele in open(scene_file,'r').readlines() if ele[0] == 'v']
        for k,v in enumerate(v_list):
            v_dic[k] = [v[0:3],v[3],int(v[4])]

        l_list = [[int(l) for l in ele.strip().split(" ")[1:]] for ele in open(scene_file,'r').readlines() if ele[0] == 'l']

        mesh_list = []
        label_v_list=[]
        label_l_list=[]
        num_k_points=0
        for k,l in enumerate(l_list):
            #print('k: ',k,' l:',l)
            #del l
            if l[3] == -1:
                continue

            #non corner
            if l[3] in [0,1]:

                #print(k,l)

                V1 = v_dic[l[0]]
                V2 = v_dic[l[1]]
                l_style = dic_style["L_" + str(l[2])]
                #print('---------------------->l_style:',l_style)
                v1 = V1[0]
                v2 = V2[0]
                if v1==v2: #avoid 
                    continue
                R = V1[1]

                #compute v_boundary
                pipe_len = du.list_dis(v1,v2)
                v_b_list = []
                sec_flag = 0
                len_sec_1 = 0.0
                len_sec_2 = 0.0
                for v_b_i,v_b in enumerate(l_style):
                    #print ('----> v_b_i, v_b: ',v_b_i,' ',v_b)
                    #first fixed length section
                    if sec_flag == 0 and v_b[2] == 0:
                        v_b_list.append([0,v_b[0:2]])
                    #first variable point
                    elif sec_flag == 0 and v_b[2] == 1:
                        sec_flag = 1
                        len_sec_1 = v_b[0]
                        v_b_list.append([1,v_b[0:2]])
                    #second variable point
                    elif sec_flag == 1 and v_b[2] == 1:
                        sec2_start_x = v_b[0]
                        v_b_list.append([2,v_b[0:2]])
                    #second fixed length section
                    elif sec_flag == 1 and v_b[2] == 0:
                        len_sec_2 = v_b[0] - sec2_start_x 
                        v_b_list.append([3,v_b[0:2]])

                len_v = pipe_len - len_sec_1 - len_sec_2

                for v_b_i,v_b in enumerate(v_b_list):
                    #print ('---------------------2_v-b_i, 2_v_b>',v_b_i,v_b)
                    if v_b[0] == 0 or v_b[0] == 1:
                        v_b_list[v_b_i] = [v_b[1][0],float(v_b[1][1])*R]
                    elif v_b[0] == 2:
                        v_b_list[v_b_i] = [len_sec_1 + len_v,float(v_b[1][1])*R]
                    elif v_b[0] == 3:
                        v_b_list[v_b_i] = [len_sec_1 + len_v + v_b[1][0] - sec2_start_x,float(v_b[1][1])*R]

                #build mesh
                points_b_list = []
                for v_b in v_b_list:
                    #print (v1,v2,v_b)

                    point_b = pl_t.points_on_circle(v1,v2,v_b,sample_rate)
                    #print(point_b)
                    points_b_list.append(point_b)

                vertexs = np.vstack(tuple(points_b_list))
                faces = pl_t.generate_triangle_mesh(vertexs,sample_rate)
                mesh_list.append([vertexs,faces,"L_" + str(k)])

                #cur_key_points =np.vstack((np.array(v1),np.array(v2)))
                #label
                #cur_line=range(2)
                cur_line =range(num_k_points+1,num_k_points + 3) 

                label_v_list.append(v1)
                label_v_list.append(v2)

                label_l_list.append(cur_line)
                num_k_points = num_k_points + 2
                #For Test
                #pl_t.write_obj(vertexs,faces,"./L_" + str(k)+".obj")

            #corner
            if l[3] == 2:

                for iter_i,iter_l in enumerate(l_list):
                    #print ('------iter_i,iter_l>',iter_i,iter_l)
                    if iter_l[3] == 2 and l[1] == iter_l[0]:
                        #??
                        l_list[k][3] = 3
                        l_list[iter_i][3] = 3

                        vertex1 = np.array(v_dic[l[0]][0])
                        vertex2 = np.array(v_dic[l[1]][0])
                        vertex3 = np.array(v_dic[iter_l[1]][0])

                        C_R = v_dic[l[1]][1]
                #if vertex1.any()==vertex2.any():
                   # continue 
                vertexs,faces,cur_key_points=pl_t.generate_elbow(vertex1,vertex2,vertex3,C_R,iter_num,sample_rate)
                mesh_list.append([vertexs,faces,"C_" + str(k) + "_" +  str(iter_i)])
                cur_num_key_points=cur_key_points.shape[0]

                temp_v=cur_key_points.tolist()
                for v_id, v in enumerate(temp_v):
                    label_v_list.append(v)
                cur_line=range(num_k_points+1,num_k_points +cur_num_key_points+1)

                label_l_list.append(cur_line)
                num_k_points = num_k_points+cur_num_key_points

                #For Test
                #pl_t.write_obj(vertexs,[],"./C_" + str(k)+".obj")
        ##every axis done


        print('meshlist:',len(mesh_list),' labellist: ',len(label_l_list))
        #save label file
        pl_t.write_line_obj(label_v_list,label_l_list,label_obj_out_file)
        pl_t.write_label_txt(label_txt_out_file,label_v_list,label_l_list)

        #merge mesh and save
        model_list = []
        for mesh_id,mesh in enumerate(mesh_list):
            #print ('mesh_id: ',mesh_id,'\nmesh: ',mesh)
            vertexs = mesh[0]
            faces = mesh[1]
            #TODO Record
            mesh_label = mesh[2]

            #model,label = objt.format_obj(vertexs,faces,mesh_id,(1,1,1))
            model,label = objt.format_obj(vertexs,faces,mesh_id,color_t.id_to_color(mesh_id))

            model_list.append([mesh_label,model,label])


        model_m = objt.obj_merge(model_list,scene_name,is_save_item=False,is_save_obj=True)
        #model_m = objt.obj_merge(model_list,scene_name,is_save_item=False,is_save_obj=False)

        #scan obj
        #compute scan point TODO config and random point
        scan_point = []
        obj_bbox = objt.obj_getbbox(model_m['pts'])
        #center
        scan_point.append([(obj_bbox[0]+obj_bbox[1])/2,(obj_bbox[2]+obj_bbox[3])/2,(obj_bbox[4]+obj_bbox[5])/2])
        #boundary
        scan_point.append([obj_bbox[0],(obj_bbox[2]+obj_bbox[3])/2,(obj_bbox[4]+obj_bbox[5])/2])
        scan_point.append([obj_bbox[1],(obj_bbox[2]+obj_bbox[3])/2,(obj_bbox[4]+obj_bbox[5])/2])
        scan_point.append([(obj_bbox[0]+obj_bbox[1])/2,(obj_bbox[2]+obj_bbox[3])/2,obj_bbox[4]])
        scan_point.append([(obj_bbox[0]+obj_bbox[1])/2,(obj_bbox[2]+obj_bbox[3])/2,obj_bbox[5]])
        #random
        #for i in range(scan_point_num):
            #scan_point.append([random.uniform(obj_bbox[0],obj_bbox[1]),random.uniform(obj_bbox[2],obj_bbox[3]),random.uniform(obj_bbox[4],obj_bbox[5])])

        scan_points,scan_points_seg,scan_points_label = scanner.virtualscan(model_m,scan_point,vscan_sample_rate)

        #ptst.pc_trans(scan_points,[-center_point[2],-center_point[1],-center_point[0]])

        du.save_dataset(scan_points,scan_points_seg,scan_points_label,scene_name,"pts","seg","plyshow",save_ply = True,
                        use_color_map=False)
        GetFootOfPerpendicular.__init__(pts,segs,label_txt_out_file,outputs_seg,outputs_obj)



#######################test######################
#sample_rate=30
#iter_num=5
#vscan_sample_rate=0.05   
#builder_scene(vscan_sample_rate,sample_rate,iter_num)

            







        
