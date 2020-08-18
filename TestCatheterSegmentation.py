import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import random
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy.testing as npt
from skimage import morphology
import copy
import math

np.random.seed(987654)
random.seed(1234569)

from UtilImage import *
from FluoroExtraction import *

GENERATED_PATH = "generated/"
DATA_PATH = "../generateTrainTestDataset/generated/"

EQUALIZATION = NORMALIZE_CONTRAST_STRETCHING

FORMAT_CONFIG = "{0:04d}"

DATA_DCM_FILE = 0
DATA_FRAME_ID = 1


global visited_array
global visited
global father
global loop_list

visited = dict()
father = dict()
loop_list = []
visited_array = None

def DFS(pos, img, seq, number):
    
    global visited
    global father
    global loop_list
    
    visited[pos] = 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_pos = (pos[0] + i, pos[1] + j)
            if new_pos[0] >=0 and new_pos[0] < img.shape[0] and new_pos[1] >=0 and new_pos[1] < img.shape[1] and \
                new_pos != pos and img[new_pos[0], new_pos[1]] == True:
                if visited[new_pos] == 1 and father[pos] != new_pos:
                    #print("Graph exist in image", seq, '-', number)
                    loop = []
                    temp = copy.copy(pos);
                    while temp != new_pos:
                        loop.append(temp)
                        #print(temp, "<-", end='')
                        temp = copy.copy(father[temp])
                    loop.append(temp)
                    loop_list.append((len(loop), loop))
                    #print(temp, len(loop_list))
                else:
                    if visited[new_pos] == 0:
                        father[new_pos] = copy.copy(pos)
                        DFS(new_pos, img, seq, number)
    visited[pos] = 2
    
    
def CheckLoopAndFind(binary_image, seq, number):
    
    global visited
    global father
    global loop_list
    
    father = dict()
    visited = dict()
    pos_list = []
    loop_list = []
    
    #print(binary_image.shape)
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j]:
                pos_list.append((i, j))
                visited[(i, j)] = 0
                father[(i, j)] = ((-9999, -9999))
    
    for pos in pos_list:
        if visited[pos] == 0:
            DFS(pos, binary_image, seq, number)
    
    if len(loop_list) > 0:
        loop_list.sort(reverse = True)
        if len(loop_list[0][1]) > 8:
            return True, loop_list[0][1]
        return False, None
    
    return False, None

def CreateImage(loop, img_shape):
    img = np.zeros(img_shape).astype(bool)
    #print(loop)
    for point in loop:
        img[point[0], point[1]] = True
    return img

def CutLoop(binary_image, loop_image):
    return np.logical_and(binary_image, np.logical_not(loop_image))

def FindConnectivity(img, start):
    global visited_array
    visited_array[start[0], start[1]] = True
    find_queue = [start]
    connectivity = []
    while len(find_queue) > 0:
        find_queue_head = find_queue.pop(0)
        connectivity.append(find_queue_head)
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_start = (find_queue_head[0] + i, find_queue_head[1] + j)
                if new_start[0] >= 0 and new_start[0] < img.shape[0] and new_start[1] >=0 and new_start[1] < img.shape[1]:
                    if img[new_start[0], new_start[1]] and visited_array[new_start[0], new_start[1]] == False:
                        find_queue.append(new_start)
                        visited_array[new_start[0], new_start[1]] = True
    return connectivity

def FindConnectivityRecursive(img, start):
    global visited_array
    visited_array[start[0], start[1]] = True
    connectivity = [start]
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_start = (start[0] + i, start[1] + j)
            if new_start[0] >= 0 and new_start[0] < img.shape[0] and new_start[1] >=0 and new_start[1] < img.shape[1]:
                if img[new_start[0], new_start[1]] and visited_array[new_start[0], new_start[1]] == False:
                    connectivity = connectivity + FindConnectivityRecursive(img, new_start)
    return connectivity 

def FindLongestOneConnectivitySubgraph(binary_image):
    global visited_array
    visited_array = np.zeros(binary_image.shape).astype(bool)
    connectivity_list = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] and visited_array[i, j] == False:
                connectivity = FindConnectivity(binary_image, (i, j))
                connectivity = list(set(connectivity))
                connectivity_list.append((len(connectivity), connectivity))
    connectivity_list.sort(reverse = True)
    return connectivity_list[0][1]

def Distance(point_in_subgraph, point_in_loop):
    return math.sqrt((point_in_subgraph[0] - point_in_loop[0])**2 + (point_in_subgraph[1] - point_in_loop[1])**2)

def FindClosePointPair(subgraph, loop):
    distancece_list = []
    for point_in_subgraph in subgraph:
        for point_in_loop in loop:
            distancece_list.append((Distance(point_in_subgraph, point_in_loop), point_in_subgraph, point_in_loop))
    distancece_list.sort()
    return (distancece_list[0][1], distancece_list[0][2])

def AdjustLoopSequence(point_on_loop, loop):
    #print(loop, len(loop), len(set(loop)))
    #assert loop[0] == loop[-1]
    ind = loop.index(point_on_loop)
    return loop[ind:] + loop[:ind] + [point_on_loop]

def FindNeighborsAndAdd(img, head_point, visited_graph, queue, possible_paths_list, shortest_path_from_start_to_current):
    has_neighbor = False
    for point_x in range(max(head_point[0] - 1, 0),  min(head_point[0] + 1, img.shape[0] - 1) + 1):
        for point_y in range(max(head_point[1] - 1, 0),  min(head_point[1] + 1, img.shape[1] - 1) + 1):
            if img[point_x, point_y] and visited_graph[point_x, point_y] == False:
                queue.append((point_x, point_y))
                visited_graph[point_x, point_y] = True
                possible_paths_list.append(shortest_path_from_start_to_current + [(point_x, point_y)])
    return queue, visited_graph, possible_paths_list

def BFS(img, start):
    queue, furtherst_point, possible_paths_list, shortest_path_from_start_to_current = [], None, [], None
    visited_graph = np.zeros(img.shape).astype(bool)
    queue.append(start)
    possible_paths_list.append([start])
    visited_graph[start[0], start[1]] = True
    while len(queue) != 0:      
        head_point, furtherst_point, shortest_path_from_start_to_current = queue[0], copy.copy(queue[0]), possible_paths_list[0]
        queue.pop(0)
        possible_paths_list.pop(0)
        assert shortest_path_from_start_to_current[-1] == furtherst_point
        queue, visited_graph, possible_paths_list = FindNeighborsAndAdd(img, head_point, visited_graph, queue, possible_paths_list, shortest_path_from_start_to_current)
    return furtherst_point, shortest_path_from_start_to_current

def FindLongestPathFromPoint(subgraph_image, point_on_subgraph):
    furtherst_point, path = BFS(subgraph_image, point_on_subgraph)
    path.reverse()
    return path

def FindLongestPathWithoutStartPoint(subgraph_image):
    for i in range(subgraph_image.shape[0]):
        for j in range(subgraph_image.shape[1]):
            if subgraph_image[i, j]:
                any_start = (i, j)
                furtherst_point, path = BFS(subgraph_image, any_start)
                furtherst_point, furtherst_path = BFS(subgraph_image, furtherst_point)
                furtherst_path.reverse()
                return furtherst_path

def FindSequence(binary_image, number, seq):
    global visited
    global father
    global loop_list
    global visited_array
    visited = dict()
    father = dict()
    loop_list = []
    visited_array = None
    exist, loop = CheckLoopAndFind(binary_image.astype(bool), seq, number)
    #print(exist, loop)
    if exist:
        loop_image = CreateImage(loop, binary_image.shape)
        binary_image = CutLoop(binary_image, loop_image)
    subgraph = FindLongestOneConnectivitySubgraph(binary_image)
    subgraph_image = CreateImage(subgraph, binary_image.shape)
    if exist:
        point_on_subgraph, point_on_loop = FindClosePointPair(subgraph, loop)
        loop = AdjustLoopSequence(point_on_loop, loop)
        longest_path_on_subgraph = FindLongestPathFromPoint(subgraph_image.astype(bool), point_on_subgraph)
        #print(longest_path_on_subgraph)
        #print(loop)
        return exist, longest_path_on_subgraph + loop
    else:
        return exist, FindLongestPathWithoutStartPoint(subgraph_image)

def FindMultiConnectivityViaSetForFinalFiltering(binary_image):
    global visited_array
    visited_array = np.zeros(binary_image.shape).astype(bool)
    connectivity_list = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] and visited_array[i, j] == False:
                connectivity = FindConnectivity(binary_image, (i, j))
                connectivity = set(connectivity)
                connectivity_list.append((len(connectivity), connectivity))
    connectivity_list.sort(reverse = True)
    return connectivity_list

def FillInImageAccordConnectivity(connectivity, img):
    for pos in connectivity:
        img[int(pos[0]), int(pos[1])] = 255
    return img

def FilterNoise(ini_seg_img, cnn_filtered_img):
    
    optimized_filtered_img = np.zeros((ini_seg_img.shape[0], ini_seg_img.shape[1])).astype(np.uint8)
    basic_filtered_img = np.zeros((ini_seg_img.shape[0], ini_seg_img.shape[1])).astype(np.uint8)
    
    ConnectivityList_InitialSeg = FindMultiConnectivityViaSetForFinalFiltering(ini_seg_img.astype(bool))
    #print('ConnectivityList_InitialSeg', ConnectivityList_InitialSeg)
    #ConnectivityList_CnnFiltered = FindMultiConnectivityViaSet(cnn_filtered_img.astype(bool))
    BasicConnectivityList = FindMultiConnectivityViaSetForFinalFiltering(np.logical_and(ini_seg_img.astype(bool), cnn_filtered_img.astype(bool)))
    #print('BasicConnectivityList', BasicConnectivityList)
    #assert len(BasicConnectivityList) == 1
    #basic_connectivity = set()
    
    ratio = 10
    
    #for i in range(len(ConnectivityList_InitialSeg)):
    #    if len(ConnectivityList_InitialSeg[i][1].intersection(ConnectivityList_CnnFiltered[0][1])) > 0:
    #        basic_connectivity = ConnectivityList_InitialSeg[i][1]
    #        basic_connectivity_index = i 
    #        break
    
    #assert len(basic_connectivity) > 0
    #optimized_filtered_img = FillInImageAccordConnectivity(basic_connectivity, optimized_filtered_img)
    
    #ConnectivityList_InitialSeg = ConnectivityList_InitialSeg[basic_connectivity_index + 1:]
    
    #first_filtered_ConnectivityList = []
    
    for i in range(len(ConnectivityList_InitialSeg)):
        if ConnectivityList_InitialSeg[i][0] >= ratio:
            optimized_filtered_img = FillInImageAccordConnectivity(ConnectivityList_InitialSeg[i][1], optimized_filtered_img)
            #first_filtered_ConnectivityList.append(ConnectivityList_InitialSeg[i][1])
        else:
            break
    
    #print('BasicConnectivityList[0][1]', BasicConnectivityList[0][1])
    
    basic_filtered_img = FillInImageAccordConnectivity(BasicConnectivityList[0][1], basic_filtered_img)
    
    return optimized_filtered_img, basic_filtered_img

def MergeIntoOneImage(optimized_filtered_img, basic_filtered_img):
    
    mixed_img = np.zeros((optimized_filtered_img.shape[0], optimized_filtered_img.shape[1], 3)).astype(np.uint8)
    
    for i in range(optimized_filtered_img.shape[0]):
        for j in range(optimized_filtered_img.shape[1]):
            if optimized_filtered_img[i, j] == 255:
                mixed_img[i, j, 0], mixed_img[i, j, 1], mixed_img[i, j, 2] = 255, 255, 255
    
    for i in range(basic_filtered_img.shape[0]):
        for j in range(basic_filtered_img.shape[1]):
            if basic_filtered_img[i, j] == 255:
                mixed_img[i, j, 0], mixed_img[i, j, 1], mixed_img[i, j, 2] = 255, 0, 0
    
    return mixed_img

def Distance(point_A, point_B):
    return math.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)

def DistanceGrid(point_A, point_B):
    return max(abs(point_A[0] - point_B[0]), abs(point_A[1] - point_B[1]))

def FindCloestPointInsideConnectivity(Connectivity, basic_point):
    #print(Connectivity, basic_point)
    cloest_point, min_distance = None, 9999999
    for point in Connectivity:
        ddd = Distance(point, basic_point)
        if ddd < min_distance:
            min_distance = ddd
            cloest_point = point
    return  cloest_point, min_distance

def FindMainConnectivityBasedOnStartAndTail(ConnectivityListIniSeg, basic_start, basic_tail):
    
    distance_list_inside_FindMainConnectivityBasedOnStartAndTail = []
    
    for i in range(len(ConnectivityListIniSeg)):
    
        cloest_start, cloest_start_distance = FindCloestPointInsideConnectivity(ConnectivityListIniSeg[i][1], basic_start)
        cloest_tail, cloest_tail_distance = FindCloestPointInsideConnectivity(ConnectivityListIniSeg[i][1], basic_tail)
        
        distance_inside_FindMainConnectivityBasedOnStartAndTail = cloest_start_distance + cloest_tail_distance
        
        distance_list_inside_FindMainConnectivityBasedOnStartAndTail.append((distance_inside_FindMainConnectivityBasedOnStartAndTail, \
                                                                             cloest_start, cloest_tail))
        
    distance_list_inside_FindMainConnectivityBasedOnStartAndTail.sort(reverse = False)
    
    return distance_list_inside_FindMainConnectivityBasedOnStartAndTail[0][1], \
        distance_list_inside_FindMainConnectivityBasedOnStartAndTail[0][2]

def FindNeighborsAndAdd(img, head_point, visited_graph, queue, possible_paths_list, shortest_path_from_start_to_current):
    has_neighbor = False
    for point_x in range(max(head_point[0] - 1, 0),  min(head_point[0] + 1, img.shape[0] - 1) + 1):
        for point_y in range(max(head_point[1] - 1, 0),  min(head_point[1] + 1, img.shape[1] - 1) + 1):
            if img[point_x, point_y] and visited_graph[point_x, point_y] == False:
                queue.append((point_x, point_y))
                visited_graph[point_x, point_y] = True
                possible_paths_list.append(shortest_path_from_start_to_current + [(point_x, point_y)])
    return queue, visited_graph, possible_paths_list

def FindShortstSequenceInsideAConnectivity(start, end, img_connect, img_size):
    img_M, img_N = img_size[0], img_size[1]
    visited_graph = np.zeros((img_M, img_N)).astype(bool)
    queue, furtherst_point, possible_paths_list, shortest_path_from_start_to_current = [], None, [], None
    queue.append(start)
    possible_paths_list.append([start])
    visited_graph[start[0], start[1]] = True
    while len(queue) != 0:      
        head_point, furtherst_point, shortest_path_from_start_to_current = queue[0], copy.copy(queue[0]), possible_paths_list[0]
        queue.pop(0)
        possible_paths_list.pop(0)
        assert shortest_path_from_start_to_current[-1] == furtherst_point
        if head_point == end:
            return shortest_path_from_start_to_current
        queue, visited_graph, possible_paths_list = \
            FindNeighborsAndAdd(img_connect, head_point, visited_graph, queue, possible_paths_list, shortest_path_from_start_to_current)
    return furtherst_point, shortest_path_from_start_to_current

def NeighborDegree(pos, img_connectivity, img_size):
    degree = 0
    for point_x in range(max(pos[0] - 1, 0),  min(pos[0] + 1, img_size[0] - 1) + 1):
        for point_y in range(max(pos[1] - 1, 0),  min(pos[1] + 1, img_size[1] - 1) + 1):
            if (point_x, point_y) != pos and img_connectivity[point_x, point_y]:
                degree += 1
    if degree == 0:
        degree = 1
    return degree

def FindAllPossibleShortestPath(img_connectivity, MainBoneSequenceHeadAndTail, img_size):
    ConnectivityList = FindMultiConnectivityViaSetForFinalFiltering(img_connectivity.astype(bool))
    #print(ConnectivityList)
    AllPossibleShortestPath = []
    # find all the point in img that has only one neighbor:
    for connectivity in ConnectivityList:
        cloest_point_to_head, ddd = FindCloestPointInsideConnectivity(connectivity[1], MainBoneSequenceHeadAndTail[0])
        cloest_point_to_tail, ddd = FindCloestPointInsideConnectivity(connectivity[1], MainBoneSequenceHeadAndTail[1])
        node_list_in_this_connectivity = []
        #print(connectivity[1], len(connectivity[1]))
        for pos in connectivity[1]:
            if NeighborDegree(pos, img_connectivity, img_size) == 1:
                    node_list_in_this_connectivity.append(pos)
        if DistanceGrid(MainBoneSequenceHeadAndTail[0], cloest_point_to_head) == 1:
            node_list_in_this_connectivity.append(cloest_point_to_head)
        if DistanceGrid(MainBoneSequenceHeadAndTail[1], cloest_point_to_tail) == 1:
            node_list_in_this_connectivity.append(cloest_point_to_tail)
        node_list_in_this_connectivity = list(set(node_list_in_this_connectivity))
        #print('node_list_in_this_connectivity', node_list_in_this_connectivity, MainBoneSequenceHeadAndTail)
        
        for i in range(0, len(node_list_in_this_connectivity) - 1):
            for j in range(i + 1, len(node_list_in_this_connectivity)):
                start, end = node_list_in_this_connectivity[i], node_list_in_this_connectivity[j]
                one_shortest_path = FindShortstSequenceInsideAConnectivity(start, end, img_connectivity, img_size)
                AllPossibleShortestPath.append((len(one_shortest_path), one_shortest_path))
                AllPossibleShortestPath.append((len(one_shortest_path), list(reversed(one_shortest_path))))
        if len(node_list_in_this_connectivity) == 1:
            #print('do exist')
            AllPossibleShortestPath.append((1, node_list_in_this_connectivity))
    return AllPossibleShortestPath

def VectorDotProduction(vec1, vec2):
    
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

def VecLen(vec):
    
    return math.sqrt(vec[0]**2 + vec[1]**2)

def generalLineFitParam(point_set, img_size):
    
    x, y = [], []
    for point in point_set:
        x.append(point[1])
        y.append(img_size[0] - point[0])
    
    n=len(x)
    mX=sum(x)/n
    mY=sum(y)/n
    sXX=sum((np.array(x)-mX)**2)
    sYY=sum((np.array(y)-mY)**2)
    sXY=sum((np.array(x)-mX)*(np.array(y)-mY))
    
    if sXY ==0 and sXX<sYY:
        isVertical=True
    else:
        isVertical=False
    
    if sXY ==0 and sXX>sYY:
        isHorizontal=True
    else:
        isHorizontal=False
    
    if sXY ==0 and sXX==sYY:
        isIndeterminate=True
    else:
        isIndeterminate=False
    
    a=0.0;b=0.0;c=0.0;
    
    if isVertical:
        a=1.0
        b=0.0
        c=-mX
    elif isHorizontal:
        a=0.0
        b=1.0
        c=-mY
    elif isIndeterminate:
        a=np.NaN
        b=np.NaN
        c=np.NaN
    else:
        slope = (sYY - sXX + ((sYY - sXX) * (sYY - sXX) + 4.0 * sXY * sXY)**0.5) / (2.0 * sXY);
        intercept = mY - slope * mX;
        if intercept >=0:
            kFactors=1
        else:
            kFactors=-1
        normFactor = kFactors * (slope * slope + 1.0)**0.5;
        a = (float)(-slope / normFactor);
        b = (float)(1.0 / normFactor);
        c = (float)(intercept / normFactor);
    #print("Line Param:a %f, b %f, c %f"%(a,b,c)) Ax+By+C=0 
    # the direction is (b, -a)
    #point_start, point_end = point_set[0], point_set[1]
    
    point_start, point_end = (point_set[0][1], img_size[0] - point_set[0][0]), (point_set[-1][1], img_size[0] - point_set[-1][0])
    point_direction = (point_end[0] - point_start[0], point_end[1] - point_start[1])
    if VectorDotProduction(point_direction, (b, -a)) > 0:
        return (b, -a)
    else:
        return (-b, a)

def AngleDifference(point_set1, point_set2, img_size):
    
    dir1 = generalLineFitParam(point_set1, img_size)
    dir2 = generalLineFitParam(point_set2, img_size)
    
    return VectorDotProduction(dir1, dir2) / (VecLen(dir1) * VecLen(dir2)), dir1, dir2
    #return math.acos(VectorDotProduction(dir1, dir2) / (VecLen(dir1) * VecLen(dir2))) * 180 / math.pi

def CrossProduct2D(vec_1, vec_2):
    a = np.array([vec_1[0], vec_1[1], 0])
    b = np.array([vec_2[0], vec_2[1], 0])
    result_cross_product = np.cross(a, b)
    #if result_cross_product[2] <= 0.1:
    #    return (None, None, None)
    return result_cross_product

def GiveScore(BoneSequence, PossibleExtraPath, ConsiderNPoint, MinPathLen, MaxRangeToAdd):
    add_score = 0
    if len(PossibleExtraPath) < MinPathLen:
        return None
    distance_bone_to_add = DistanceGrid(BoneSequence[-1], PossibleExtraPath[0])
    if distance_bone_to_add > MaxRangeToAdd:
        return None
    
    if distance_bone_to_add == 1:
        add_score += 10000
    
    LenBoneSequence = len(BoneSequence)
    
    #print(LenBoneSequence, ConsiderNPoint)
    
    BoneTailSequence = BoneSequence[LenBoneSequence - ConsiderNPoint:LenBoneSequence]
    ConnectLine = BoneSequence[LenBoneSequence - int(ConsiderNPoint/2):LenBoneSequence] + PossibleExtraPath[:int(ConsiderNPoint/2)]
    ExtraPathHeadSequence = PossibleExtraPath[:ConsiderNPoint]
    
    angle_bone_to_connectLine = AngleDifference(BoneTailSequence, ConnectLine, img_size)
    angle_connectLine_to_extraPath = AngleDifference(ConnectLine, ExtraPathHeadSequence, img_size)
    
    if angle_bone_to_connectLine >=0 and angle_connectLine_to_extraPath >= 0:
        add_score += angle_bone_to_connectLine * angle_connectLine_to_extraPath
    else:
        return None
    
    return add_score

def CalculateNormForMainBoneSequence(ASequence, ConsiderNPoint, img_size):
    
    LenASequence = len(ASequence)
    
    if LenASequence <= 2:
        return None, None
    
    #print(
    #     MainBoneSequence[LenMainBoneSequence - int(ConsiderNPoint/2) - ConsiderNPoint:LenMainBoneSequence - int(ConsiderNPoint/2)], \
    #     MainBoneSequence[LenMainBoneSequence - ConsiderNPoint:LenMainBoneSequence])
    
    angle_diff, vec_seq1, vec_seq2 = AngleDifference(
            ASequence[0:int(LenASequence * 2 / 3)], \
            ASequence[int(LenASequence / 3):LenASequence], \
                                                    img_size)
    
    #print(ASequence[0:int(LenASequence * 2 / 3)], vec_seq1)
    #print(ASequence[int(LenASequence / 3):LenASequence], vec_seq2)
    
    #print(CrossProduct2D(vec_seq1, vec_seq2)[2])
    
    #if angle_diff > 0.95:
    #    return None
    
    return CrossProduct2D(vec_seq1, vec_seq2)[2], angle_diff

def AngleChangeAbruptly(angle_diff):
    return angle_diff <= 0.75

def IsSameDirection(angle_and_dir_list, global_norm_of_cathether):
    norm_previous = copy.copy(global_norm_of_cathether)
    for item in angle_and_dir_list:
        norm_current, angle_score = item[1], item[0]
        if abs(norm_current) <= 0.1:
            # this means it is the same dir, we ignore these
            # norm_current = copy.copy(norm_previous)
            continue
        if norm_current * global_norm_of_cathether < 0:
            if AngleChangeAbruptly(angle_score) == True:
                return False
        """
        if abs(norm_previous) > 0.1:
            if norm_current * norm_previous < 0:
                # We can allow it to change direction only if the angle_score does not change too much
                # that is to say, if the angle change too much, and now cathether is in different direction, then we do not add
                if AngleChangeAbruptly(angle_score) == True:
                    return False
        norm_previous = copy.copy(norm_current)
        """
    return True

def ReasonableSegmentToAdd(JudegSequence, ConsiderNPoint, img_size, global_norm_of_cathether):
    angle_and_dir_list = []
    FinalScore = 0
    # the norm must be the same, if not the same, the only possibility is that it is changing slowly
    # that is to say, once the cos at t and cos at t + 1 changes abruptly, then it must be in the same direction
    LenOfJudgeSeq = len(JudegSequence)
    #print(LenOfJudgeSeq)
    #for i in range(0, LenOfJudgeSeq - ConsiderNPoint + 1):
    #    seq_previous, seq_later = JudegSequence[i:i+int(ConsiderNPoint/2)], JudegSequence[i+int(ConsiderNPoint/2):i+ConsiderNPoint]
    for i in range(0, LenOfJudgeSeq - int(ConsiderNPoint/2)):
        seq_previous, seq_later = JudegSequence[i:i+int(ConsiderNPoint/2)], JudegSequence[i+1:i+1+int(ConsiderNPoint/2)]
        angle_diff, vec_seq1, vec_seq2 = AngleDifference(seq_previous, seq_later, img_size)
        norm_current = CrossProduct2D(vec_seq1, vec_seq2)[2]
        #print(seq_previous, seq_later, angle_diff, norm_current, global_norm_of_cathether)
        #if angle_diff >= 0.9:
        #    norm_current = 0.0
        angle_and_dir_list.append((angle_diff, norm_current))
        FinalScore += angle_diff
    return IsSameDirection(angle_and_dir_list, global_norm_of_cathether), FinalScore
    """
    for i in range(len(angle_and_dir_list)):
        (angle_diff, norm_current) = angle_and_dir_list[i]
        if AngleChangeAbruptly(angle_diff):
            #print(JudegSequence[i:i+int(ConsiderNPoint/2)], JudegSequence[i+int(ConsiderNPoint/2):i+ConsiderNPoint], angle_diff, norm_current)
            return IsSameDirection(angle_and_dir_list, global_norm_of_cathether), FinalScore
    return True, FinalScore
    """
    #if angle_diff > Angle_Abrupt_Change_Max:    
    # if cos changes abruptly, then 
    

def GiveScore2(BoneSequence, PossibleExtraPath, ConsiderNPoint, MinPathLen, MaxRangeToAdd, img_size, global_norm_of_cathether):
    add_score = 0
    
    distance_bone_to_add = Distance(BoneSequence[-1], PossibleExtraPath[0])
    if distance_bone_to_add > MaxRangeToAdd:
        return None, None
    
    #print('hello', BoneSequence[-1], PossibleExtraPath)
    
    if DistanceGrid(BoneSequence[-1], PossibleExtraPath[0]) <= 1:
        # if this is connected as a neighbor, then we will consider 
        
        add_score = add_score + 10000 + int(len(PossibleExtraPath) >= 3 * MinPathLen)
        
    elif len(PossibleExtraPath) < MinPathLen:
        return None, None
    
    """
    
    loop_terminate_point = FindCloestPointInsideConnectivity(BoneSequence, PossibleExtraPath[-1])[0]
    if DistanceGrid(loop_terminate_point,  PossibleExtraPath[-1]) == 1 and \
        DistanceGrid(PossibleExtraPath[0], PossibleExtraPath[-1]) > 1 and DistanceGrid(loop_terminate_point, BoneSequence[-1]) >= 5:
        
        #print('loop')
        #print(BoneSequence)
        #print('loop--')
        #print( PossibleExtraPath)
        
        
        add_score += 100000000
    
    """
    
    #print('hello', PossibleExtraPath, min(int(ConsiderNPoint/2), len(PossibleExtraPath)))
    
    LenBoneSequence = len(BoneSequence)
    
    #print(LenBoneSequence, ConsiderNPoint)
    
    BoneTailSequence = BoneSequence[LenBoneSequence - ConsiderNPoint:LenBoneSequence]
    
    FindSuitableConnectLine = False
    SuitableConnectLineLen = None
    
    for i in range(min(int(ConsiderNPoint/2), len(PossibleExtraPath)), 0, -1):
    
        ConnectLine = BoneSequence[LenBoneSequence - int(ConsiderNPoint/2):LenBoneSequence] + \
            PossibleExtraPath[:i]
        
        #ExtraPathHeadSequence = PossibleExtraPath[:ConsiderNPoint]
        BoneTailSequence_To_ConnectLine = BoneTailSequence + AddLinePoints(BoneTailSequence[-1], PossibleExtraPath[0]) + PossibleExtraPath[:i]
        
        #norm_of_BoneTailSequence_and_ConnectLine, angle_of_BoneTailSequence_and_ConnectLine = \
        #    CalculateNormForMainBoneSequence(BoneTailSequence_To_ConnectLine, ConsiderNPoint, img_size)
    
        #angle_bone_to_connectLine, vec_BoneTailSequence, vec_ConnectLine = AngleDifference(BoneTailSequence, ConnectLine, img_size)
        #norm_of_BoneTailSequence_and_ConnectLine = CrossProduct2D(vec_BoneTailSequence, vec_ConnectLine)[2]
        
        
            
        #    print(BoneTailSequence,  BoneTailSequence_To_ConnectLine, norm_of_BoneTailSequence_and_ConnectLine, \
        #             angle_of_BoneTailSequence_and_ConnectLine, norm_of_cather)
        
        #if angle_of_BoneTailSequence_and_ConnectLine > 0.88 or abs(norm_of_cather) < 0.1 or \
        #    norm_of_BoneTailSequence_and_ConnectLine * norm_of_cather > 0:
        ReasonableToAdd, ReasonableScore = ReasonableSegmentToAdd(BoneTailSequence_To_ConnectLine, ConsiderNPoint, img_size, global_norm_of_cathether)
        
        #if DistanceGrid(BoneSequence[-1], PossibleExtraPath[0]) > 1:
        #    print(BoneTailSequence, BoneTailSequence_To_ConnectLine, ReasonableToAdd, global_norm_of_cathether)
        if ReasonableToAdd == True:
            # The critical function to add reasonable segment
            #if angle_bone_to_connectLine > 0.0:
            FindSuitableConnectLine = True
            SuitableConnectLineLen = copy.copy(i)
            
        break
        
        """
        #print(PossibleExtraPath, vec_BoneTailSequence, vec_ConnectLine)
        norm_of_BoneTailSequence_and_ConnectLine = CrossProduct2D(vec_BoneTailSequence, vec_ConnectLine)[2]
        # get the z - axis
        if len(PossibleExtraPath) == 1:
            print('hellooo', BoneTailSequence, ConnectLine, vec_BoneTailSequence, vec_ConnectLine, norm_of_BoneTailSequence_and_ConnectLine)
    
        if  norm_of_cather == None or norm_of_BoneTailSequence_and_ConnectLine == None or \
            norm_of_BoneTailSequence_and_ConnectLine * norm_of_cather > 0:
            FindSuitableConnectLine = True
            SuitableConnectLineLen = copy.copy(i)
            break
        """
    
    #angle_connectLine_to_extraPath = AngleDifference(ConnectLine, ExtraPathHeadSequence)
    
    if FindSuitableConnectLine == True:
        add_score += ReasonableScore
        if add_score >= 10000:
            #print(PossibleExtraPath, 'dfdff')
            return add_score, SuitableConnectLineLen
        return add_score, SuitableConnectLineLen
    else:
        return None, None
        """
        if add_score >= 10000:
            #print(PossibleExtraPath)
            return add_score, len(PossibleExtraPath)
        else:
            return None, None
        """
    #return add_score, SuitableConnectLineLen

def SelectRemainSeg(MainBoneSequence, PossibleInitialSegPathSet, ConsiderNPoint, MinPathLen, MaxRangeToAdd):
    possible_path, possible_path_index, path_max_score = None, None, -99999999
    for i in range(len(PossibleInitialSegPathSet)):
        path_score = GiveScore(MainBoneSequence, PossibleInitialSegPathSet[i][1], ConsiderNPoint, MinPathLen, MaxRangeToAdd)
        if path_score != None and path_score > path_max_score:
            path_max_score = path_score
            possible_path = PossibleInitialSegPathSet[i][1]
            possible_path_index = copy.copy(i)
    return possible_path, possible_path_index

def SelectRemainSeg2(MainBoneSequence, PossibleInitialSegPathSet, ConsiderNPoint, MinPathLen, MaxRangeToAdd, img_size, global_norm_of_cather):
    possible_path, possible_path_index, path_max_score, possible_path_len = None, None, -99999999, None
    terminate_or_not = False
    for i in range(len(PossibleInitialSegPathSet)):
        path_score, SuitableConnectLineLen = \
            GiveScore2(MainBoneSequence, PossibleInitialSegPathSet[i][1], ConsiderNPoint, MinPathLen, MaxRangeToAdd, img_size, global_norm_of_cather)
        #print(PossibleInitialSegPathSet[i][1], path_score)
        if path_score != None and path_score > path_max_score:
            path_max_score = path_score
            possible_path = PossibleInitialSegPathSet[i][1]
            possible_path_index = copy.copy(i)
            if path_score >= 100000000:
                possible_path_len = len(PossibleInitialSegPathSet[i][1])
                terminate_or_not = True
            else:
                possible_path_len = copy.copy(SuitableConnectLineLen)
    return possible_path, possible_path_index, possible_path_len, terminate_or_not

def AddLinePoints(start_point, end_point):
    point_list = []
    vec_dir = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    vec_dir_N = DistanceGrid(start_point, end_point)
    if vec_dir_N <= 1:
        return []
    for t in range(0, vec_dir_N + 1):
        current_point = (start_point[0] + t * vec_dir[0] * 1.0 / vec_dir_N, start_point[1] + t * vec_dir[1] * 1.0 / vec_dir_N)
        if len(point_list) == 0 or point_list[-1] != current_point:
            point_list.append(current_point)
    return point_list[1:-1]

def FilterIter3RelaxImg(iter2_img, iter3_relax_img):
    iter2_img_seg = FindMultiConnectivityViaSetForFinalFiltering(iter2_img)
    iter3_img_seg = FindMultiConnectivityViaSetForFinalFiltering(iter3_relax_img)
    MinSetLen = 8
    img_size = (iter2_img.shape[0], iter2_img.shape[1])
    result_img = np.zeros(img_size).astype(np.uint8)
    for small_connectivity_iter3 in iter3_img_seg:
        for small_connectivity_iter2 in iter2_img_seg:
            if small_connectivity_iter2[0] >= MinSetLen and len(small_connectivity_iter3[1].intersection(small_connectivity_iter2[1])) > 0:
                result_img = FillInImageAccordConnectivity(small_connectivity_iter3[1], result_img)
    return result_img

def finalPostProcessingFilter(final_seg_img_without_filtering, initial_seg_img, iter2_seg_img):
        
        #optimized_filtered_img = FilterIter3RelaxImg(imageio.imread(iter2_seg_img_path), optimized_filtered_img_ini)
    
        SupplementNPoint, ConsiderNPoint, MinPathLen, MaxRangeToAdd = 6, 8, 6, 15

        Norm_of_Cathether, Norm_of_Cathether_Reverse = None, None
        
        #print('hello ----**** ')
        
        #imageio.imwrite('skeleton_optimized_filtered_img.png', final_seg_img_without_filtering)
    
        optimized_filtered_img_ini, basic_filtered_img = FilterNoise(final_seg_img_without_filtering, initial_seg_img)
        
        #print('hello ----')
        
        
        
        optimized_filtered_img = FilterIter3RelaxImg(iter2_seg_img, optimized_filtered_img_ini)
        
        #print('hello ----**** ')
        
        #optimized_img_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + \
        #     str(index) + '_filtered_UsingMethod5.png'
        
        #erosion_img = cv2.dilate(imageio.imread(seg_img), kernel)
        
        #skeleton_img = imageio.imread(seg_img)
        
        skeleton_optimized_filtered_img = morphology.skeletonize(optimized_filtered_img.astype(bool)).astype(bool)
        #skeleton_optimized_filtered_img = (skeleton_optimized_filtered_img.astype(int) * 255).astype(np.uint8)
        
        skeleton_basic_filtered_img = morphology.skeletonize(basic_filtered_img.astype(bool)).astype(bool)
        #skeleton_basic_filtered_img = (skeleton_basic_filtered_img.astype(int) * 255).astype(np.uint8)
        
        
        
        label_exist_loop_ori, skeleton_sequence_tempp_ori = FindSequence(skeleton_optimized_filtered_img, seq, index)
        
        label_exist_loop_basic, skeleton_sequence_tempp = FindSequence(skeleton_basic_filtered_img, seq, index)
        
        img_size = (skeleton_optimized_filtered_img.shape[0], skeleton_optimized_filtered_img.shape[1])
        
        final_sequence_tempp = None
        
        if label_exist_loop_basic == True:
            
            #print(skeleton_sequence_tempp_ori)
            #print(set(skeleton_sequence_tempp) - set(skeleton_sequence_tempp_ori))
            
            # If loop exist, we can directly finish this procedure
            
            final_sequence_tempp = skeleton_sequence_tempp
        
        
        else:
            
            # If not, we find the start, and tail of this sequence
            
            basic_start, basic_tail = skeleton_sequence_tempp[0], skeleton_sequence_tempp[-1]
            
            ConnectivityList_InitialSeg = FindMultiConnectivityViaSetForFinalFiltering(skeleton_optimized_filtered_img)
            
            ClosetStart, CloestEnd = FindMainConnectivityBasedOnStartAndTail(ConnectivityList_InitialSeg, basic_start, basic_tail)
            
            MainBoneSequence = FindShortstSequenceInsideAConnectivity(ClosetStart, CloestEnd, skeleton_optimized_filtered_img.astype(bool), img_size)
            
            #print('hello')
            
            if label_exist_loop_ori == True and len(set(MainBoneSequence) - set(skeleton_sequence_tempp_ori).intersection(set(MainBoneSequence))) == 0:
                
                #assert len(set(MainBoneSequence) - set(skeleton_sequence_tempp_ori)) == 0
                
                final_sequence_tempp = skeleton_sequence_tempp_ori
            
            else:
            
                #print('MainBoneSequence', MainBoneSequence)

                connectivity_img = np.zeros(img_size).astype(np.uint8)

                connectivity_img = FillInImageAccordConnectivity(MainBoneSequence, connectivity_img)

                #print('debug')

                PossibleInitialSegPathSet = FindAllPossibleShortestPath(np.logical_and(skeleton_optimized_filtered_img.astype(bool), np.logical_xor(skeleton_optimized_filtered_img.astype(bool), connectivity_img.astype(bool)).astype(bool)), [MainBoneSequence[0], MainBoneSequence[-1]], img_size)
                #print('ConnectivityList_InitialSeg')

                #imageio.imwrite('temp.png', \
                #    (np.logical_xor(skeleton_optimized_filtered_img.astype(bool), \
                #                    connectivity_img.astype(bool)).astype(int) * 255).astype(np.uint8))

                #for a in ConnectivityList_InitialSeg:
                #    print(a)

                #print('PossibleInitialSegPathSet')

                #for a in PossibleInitialSegPathSet:
                #    print(a)
                iter_k = 0
                
                global_Norm_of_Cathether, Score_of_Cathether = CalculateNormForMainBoneSequence(MainBoneSequence, ConsiderNPoint, img_size)
                
                if global_Norm_of_Cathether == None and Score_of_Cathether == None:
                    
                    return np.zeros((img_size[0], img_size[1])).astype(np.uint8)
                
                global_Norm_of_Cathether_Reverse, Score_of_Cathether_Reverse = CalculateNormForMainBoneSequence(list(reversed(MainBoneSequence)), \
                                                                                                         ConsiderNPoint, img_size)

                while 1:

                    #print('find')

                    #if Norm_of_Cathether == None:

                        #print('Norm_of_Cathether')
                    #previous_Norm_of_Cathether = copy.copy(Norm_of_Cathether)

                    front_Norm_of_Cathether, Score_of_Cathether = CalculateNormForMainBoneSequence(MainBoneSequence[len(MainBoneSequence) - int(ConsiderNPoint * 3 / 2): len(MainBoneSequence)], ConsiderNPoint, img_size)
                    #print(front_Norm_of_Cathether, Score_of_Cathether, global_Norm_of_Cathether)
                    
                    #if Score_of_Cathether >= 0.9:
                        
                    #    front_Norm_of_Cathether = 0.0
                    
                    if front_Norm_of_Cathether == None or abs(front_Norm_of_Cathether) <= 0.1:
                        
                        front_Norm_of_Cathether = copy.copy(global_Norm_of_Cathether)
                    
                    #assert Norm_of_Cathether 
                    
                    #print('Norm_of_Cathether', Norm_of_Cathether)

                    SelectAddPath, SelectAddSegIndex, SelectAddSegLen, terminate_or_not = SelectRemainSeg2(MainBoneSequence, PossibleInitialSegPathSet, ConsiderNPoint, MinPathLen, MaxRangeToAdd, img_size, front_Norm_of_Cathether)

                    if SelectAddSegIndex != None:

                        #print('SelectAddPath', SelectAddPath[:SelectAddSegLen])

                        Mid_Line = []

                        if DistanceGrid(MainBoneSequence[-1], SelectAddPath[0]) > 1:

                            Mid_Line = AddLinePoints(MainBoneSequence[-1], SelectAddPath[0])

                        MainBoneSequence = MainBoneSequence + Mid_Line + SelectAddPath[:SelectAddSegLen]

                        #PossibleInitialSegPathSet.pop(SelectAddSegIndex)
                    #print('find reverse')

                    #if Norm_of_Cathether_Reverse == None:
                    
                    #previous_Norm_of_Cathether_Reverse = copy.copy(Norm_of_Cathether_Reverse)
                    
                    reversed_MainBoneSequence = list(reversed(MainBoneSequence))

                    back_Norm_of_Cathether_Reverse, Score_of_Cathether_Reverse = CalculateNormForMainBoneSequence(reversed_MainBoneSequence[len(reversed_MainBoneSequence) - int(ConsiderNPoint * 3 / 2):len(reversed_MainBoneSequence)], ConsiderNPoint, img_size)
                    #if Score_of_Cathether_Reverse >= 0.9:
                        
                    #    back_Norm_of_Cathether_Reverse = 0.0
                    
                    if back_Norm_of_Cathether_Reverse == None or abs(back_Norm_of_Cathether_Reverse) <= 0.1:
                        
                        back_Norm_of_Cathether_Reverse = copy.copy(global_Norm_of_Cathether_Reverse)
                    
                    #print('Norm_of_Cathether_Reverse', Norm_of_Cathether_Reverse)

                        #print('Norm_of_Cathether_Reverse ', Norm_of_Cathether_Reverse)

                    SelectAddPath, SelectAddSegIndexReverse, SelectAddSegLen, terminate_or_not_reverse = SelectRemainSeg2(list(reversed(MainBoneSequence)), PossibleInitialSegPathSet, ConsiderNPoint, MinPathLen, MaxRangeToAdd, img_size, back_Norm_of_Cathether_Reverse)
                    if SelectAddSegIndexReverse != None:

                        reversed_add_sequence = list(reversed(SelectAddPath[:SelectAddSegLen]))

                        Mid_Line = []

                        if DistanceGrid(reversed_add_sequence[-1], MainBoneSequence[0]) > 1:

                            Mid_Line = AddLinePoints(reversed_add_sequence[-1], MainBoneSequence[0])

                        MainBoneSequence = reversed_add_sequence + Mid_Line + MainBoneSequence

                        #print('SelectAddPathReverse', list(reversed(SelectAddPath[:SelectAddSegLen])))

                        #PossibleInitialSegPathSet.pop(SelectAddSegIndex)

                        #else:

                        #    break
                    #print('pp', SelectAddSegIndex , SelectAddSegIndexReverse, terminate_or_not_reverse, terminate_or_not)

                    if SelectAddSegIndex == None and SelectAddSegIndexReverse == None:

                        break

                    if terminate_or_not_reverse == True or terminate_or_not == True:

                        break

                    connectivity_img = np.zeros(img_size).astype(np.uint8)

                    connectivity_img = FillInImageAccordConnectivity(MainBoneSequence, connectivity_img)
                    
                    if FindSequence(connectivity_img.astype(bool), seq, index)[0] == True:
                        
                        # if loop detected then we should terminate early
                        
                        break
                    
                    #extra_img = np.logical_and(skeleton_optimized_filtered_img.astype(bool), \
                    #                                                                   np.logical_xor(skeleton_optimized_filtered_img.astype(bool), \
                    #                                                                   connectivity_img.astype(bool)).astype(bool))
                    

                    #imageio.imwrite('extra_img' + str(iter_k) + '.png', (extra_img.astype(int) * 255).astype(np.uint8))

                    PossibleInitialSegPathSet = FindAllPossibleShortestPath(np.logical_and(skeleton_optimized_filtered_img.astype(bool), np.logical_xor(skeleton_optimized_filtered_img.astype(bool), connectivity_img.astype(bool)).astype(bool)), [MainBoneSequence[0], MainBoneSequence[-1]], img_size)
                    
                    #if iter_k == 2:
                        
                    #    for a in PossibleInitialSegPathSet:
                    #        print('dd', a)

                    iter_k += 1

                    #print('PossibleInitialSegPathSet')

                    #for a in PossibleInitialSegPathSet:
                    #    print('dd', a)



                #print('MainBoneSequence', MainBoneSequence)

                #print('PossibleInitialSegPathSet')

                #for one_path in PossibleInitialSegPathSet:

                #    print(one_path)

                    #print(PossibleInitialSegPathSet)

                final_sequence_tempp = MainBoneSequence
            
                # then we find the basic sequence that has smallest distance to basic_start and basic_tail
            
            
        
        #filter_img = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + \
        #     str(index) + '_filtered_UsingSegAndEnlargedInitial.png'
        #skeleton_optimized_filtered_img2 = morphology.skeletonize(optimized_filtered_img.astype(bool)).astype(bool)
        #skeleton_optimized_filtered_img2 = (skeleton_optimized_filtered_img2.astype(int) * 255).astype(np.uint8)
        
        #mixed_img = MergeIntoOneImage(skeleton_optimized_filtered_img, connectivity_img)
        
        result_img = np.zeros((img_size[0], img_size[1])).astype(np.uint8)
        
        result_img = FillInImageAccordConnectivity(final_sequence_tempp, result_img)
        
        final_connectivity_set = FindMultiConnectivityViaSetForFinalFiltering(result_img.astype(bool))
        
        final_connectivity_set.sort(reverse = True)
        
        result_img = np.zeros((img_size[0], img_size[1])).astype(np.uint8)
        
        result_img = FillInImageAccordConnectivity(final_connectivity_set[0][1], result_img)

        #vis_new_save_folder = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis_UsingPrevious_optimizedMethod26/'

        #if os.path.exists(vis_new_save_folder) == False:
        #    os.makedirs(vis_new_save_folder)
        
        print('done', index)
        #initial_seg_new_save_path2 = vis_new_save_folder + str(index) + '_data_initial.png'
        
        #skeleton_optimized_filtered_img = (skeleton_optimized_filtered_img.astype(int) * 255).astype(np.uint8)
        
        #imageio.imwrite(vis_new_save_folder + str(index) + '_image_ori.png', imageio.imread(ori_img_path))
        #imageio.imwrite(vis_new_save_folder + str(index) + '_initial_seg.png', imageio.imread(initial_seg_img_path))
        #imageio.imwrite(vis_new_save_folder + str(index) + '_skeleton.png',  skeleton_optimized_filtered_img2)
        #imageio.imwrite(vis_new_save_folder + str(index) + '_optimized_img.png', imageio.imread(optimized_img_path))
        #imageio.imwrite(vis_new_save_folder + str(index) + '_final_result.png',result_img)
        return result_img
        #imageio.imwrite(vis_new_save_folder + str(index) + '_iter2_seg_Iter2Item20Epoch150Final.png', imageio.imread(one_seg_img))
        #imageio.imwrite(vis_new_save_folder + str(index) + '_iter2_seg_Iter2Item20Epoch156Final.png', imageio.imread(two_seg_img))
        #imageio.imwrite(vis_new_save_folder + str(index) + '_iter2_seg_Iter2Item20Epoch150V6Final.png', imageio.imread(three_seg_img))   














































"""

def FindConnectivity(img, start):
    global visited_array
    visited_array[start[0], start[1]] = True
    connectivity = [start]
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_start = (start[0] + i, start[1] + j)
            if new_start[0] >= 0 and new_start[0] < img.shape[0] and new_start[1] >=0 and new_start[1] < img.shape[1]:
                if img[new_start[0], new_start[1]] and visited_array[new_start[0], new_start[1]] == False:
                    connectivity = connectivity + FindConnectivity(img, new_start)
    return connectivity
"""

def FindMultiConnectivity(binary_image):
    global visited_array
    visited_array = np.zeros(binary_image.shape).astype(bool)
    connectivity_list = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] and visited_array[i, j] == False:
                connectivity = FindConnectivity(binary_image, (i, j))
                connectivity = list(set(connectivity))
                connectivity_list.append((len(connectivity), connectivity))
    connectivity_list.sort()
    return connectivity_list

def DelectSmallConnectivity(connectivity_list, N):
    for i in range(len(connectivity_list)):
        if connectivity_list[i][0] > N:
            break
    return connectivity_list[i:]

def TheEnlargeConnectivityList(connectivity_list, img_i_max, img_j_max, radius):
    new_connectivity_list = []
    for i in range(len(connectivity_list)):
        new_connectivity_list.append((connectivity_list[i][0], TheEnlargeConnectivity(connectivity_list[i][1], img_i_max, img_j_max, radius)))
    return new_connectivity_list

def TheEnlargeConnectivity(connectivity, img_i_max, img_j_max, radius):
    enlarged_connectivity = []
    for pos in connectivity:
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                new_pos = (pos[0] + i, pos[1] + j)
                if new_pos[0] >=0 and new_pos[0] < img_i_max and new_pos[1] >=0 and new_pos[1] < img_j_max:
                    enlarged_connectivity.append(new_pos)
    return enlarged_connectivity

def CreateImageFromConnectivityList(connectivity_list, img_shape):
    img = np.zeros((img_shape[0], img_shape[1])).astype(np.uint8)
    for i in range(len(connectivity_list)):
        for pos in connectivity_list[i][1]:
            img[pos[0], pos[1]] = 255
    return img

def PrcoessImageBySymbol(img, delete_small, N, enlarge, radius):
    connectivity_list = FindMultiConnectivity(img)
    if delete_small:
        connectivity_list = DelectSmallConnectivity(connectivity_list, N)
    if enlarge:
        connectivity_list = TheEnlargeConnectivityList(connectivity_list, img.shape[0], img.shape[1], radius)
    return CreateImageFromConnectivityList(connectivity_list, img.shape)

def GetExtraImage(predicted_img, data_img):
	extra_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1])).astype(np.uint8)
	for i in range(predicted_img.shape[0]):
		for j in range(predicted_img.shape[1]):
			if predicted_img[i, j] > 125 and data_img[i, j] < 125:
				extra_img[i, j] = 255
	return extra_img


def GenerateCircleList(radius):
	circle_list = []
	x, y = 0, 0
	for i in range(x - radius, x + radius + 1):
		for j in range(y - radius, y + radius + 1):
			if ((i - x)**2 + (j - y)**2) <= radius**2:
				circle_list.append((i, j))
	circle_np = np.array(circle_list)
	return circle_np

def CreateAttentionFromSegmentation(initial_segmentation_img, ori_img):
	radius, kernel_num, kernel_size = 100, 5, (11, 11)
	draw_index_zero = GenerateCircleList(radius)
	center_x, center_y = FindCenter(initial_segmentation_img)
	draw_index = draw_index_zero + np.array([int(center_y), int(center_x)])
	attention_map = np.zeros((initial_segmentation_img.shape[0], initial_segmentation_img.shape[1])).astype('float32')
	w, h = attention_map.shape[0], attention_map.shape[1]
	for index in draw_index:
		if 0 <= index[0] and index[0] < w and 0 <= index[1] and index[1] < h:
			attention_map[index[0], index[1]] = 255
	i = 0
	while i < kernel_num:
		attention_map = cv2.GaussianBlur(attention_map, kernel_size, 0)
		attention_map = attention_map * 255 / np.max(attention_map[:,:])
		i += 1
	attention_img = ori_img * attention_map / 255.0
	return attention_img.astype(np.uint8)

def FindCropBorder(crop_center, crop_size, global_img_size):
	center_x, center_y = crop_center[0], crop_center[1]
	crop_w, crop_h = crop_size[0], crop_size[1]
	if center_x - crop_w / 2 < 0:
		center_x = crop_w / 2
	elif center_x + crop_w / 2 >= global_img_size[1]:
		center_x = global_img_size[1] - 1 - crop_w / 2
	if center_y - crop_h / 2 < 0:
		center_y = crop_h / 2
	elif center_y + crop_h / 2 >= global_img_size[0]:
		center_y = global_img_size[0] - 1 - crop_h / 2
	left_upper_x, left_upper_y = int(center_x - crop_w / 2) + 1, int(center_y - crop_h / 2) + 1
	right_lower_x, right_lower_y = int(center_x + crop_w / 2), int(center_y + crop_h / 2)
	return ((left_upper_x, left_upper_y), (right_lower_x, right_lower_y))

def InitializationToSaveAttentionMapAndClipImg(initial_segmentation_img, ori_img):
	radius, kernel_num, kernel_size = 100, 5, (11, 11)
	draw_index_zero = GenerateCircleList(radius)
	center_x, center_y = FindCenter(initial_segmentation_img.astype('float32'))
	draw_index = draw_index_zero + np.array([int(center_y), int(center_x)])
	attention_map = np.zeros((initial_segmentation_img.shape[0], initial_segmentation_img.shape[1])).astype('float32')
	w, h = attention_map.shape[0], attention_map.shape[1]
	for index in draw_index:
		if 0 <= index[0] and index[0] < w and 0 <= index[1] and index[1] < h:
			attention_map[index[0], index[1]] = 255
	i = 0
	while i < kernel_num:
		attention_map = cv2.GaussianBlur(attention_map, kernel_size, 0)
		attention_map = attention_map * 255 / np.max(attention_map[:,:])
		i += 1
	attention_img = ori_img.astype('float32') * attention_map / 255.0
	crop_size = (192, 192)
	global_img_size = (512, 512)
	crop_border = FindCropBorder((center_x, center_y), crop_size, global_img_size)
	((left_upper_x, left_upper_y), (right_lower_x, right_lower_y)) = crop_border
	current_clipped_img = ori_img[left_upper_y: right_lower_y + 1, left_upper_x: right_lower_x + 1]
	return attention_img.astype(np.uint8), current_clipped_img

attentionSubmodule = FluoroExtraction('Attention')

iniSegmentationModule = FluoroExtraction('Ini_Seg')

incrementalModule = FluoroExtraction('Incremental')

fineSegmentationModule = FluoroExtraction('Fine_Seg')

fineSegmentationModuleAtIter1 = FluoroExtraction('Fine_Seg_Iter1')

finalRelaxtionSubmodule = FluoroExtraction('Final_Relaxtion')

path = '/home/zsj/data/CTSAWorkspace/sequence'
data_list = []
for seq in [19]:
		path_seq = path + str(seq) + '/icon/'
		imgs_in_seq = os.listdir(path_seq)
		imgs_in_seq.sort()
		NB_CHANNEL = 4
		save_path = path + str(seq) + '/all_module_integrate_no_recursive_to_find_connectivity/'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		else:
			print('has folder')
		
		if seq != 22:
			for index in range(0, 3):
				ini_seg_global = imageio.imread('/home/zsj/data/CTSAWorkspace/Eva/seq' + str(seq) + '/' + str(index) + '_human_label.png')
				current_global_img = imageio.imread(path + str(seq) + '/icon/' + imgs_in_seq[index])
				attention_img, current_clipped_img = InitializationToSaveAttentionMapAndClipImg(ini_seg_global, current_global_img)
				imageio.imwrite(save_path + str(index) + '_attention_from_initial_segmentation.png', attention_img)
				imageio.imwrite(save_path + str(index) + '_input.png', current_clipped_img)  
				#weird_offset_array = np.load('/home/zsj/data/weird_offset_array.npy')
		
		clip_offset_array = np.zeros((len(imgs_in_seq), 4))

		for index in range(3, len(imgs_in_seq)):
			global_img = np.empty([1, 4, 512, 512], dtype=np.float32)
			current_global_img_path = path + str(seq) + '/icon/' + imgs_in_seq[index]
			if os.path.exists(current_global_img_path) == False:
				print('not exist ', current_global_img_path)
			current_global_img = imageio.imread(current_global_img_path)
			for j in range(0, 4):
				if j == 0:
					global_img[0, j][...] = current_global_img.astype('float32') / 255.0
				else:
					previous_attention_img_path = save_path + str(index - j) + '_attention_from_initial_segmentation.png'
					if os.path.exists(previous_attention_img_path) == False:
						print('not exist ', previous_attention_img_path)
					global_img[0, j][...] = imageio.imread(previous_attention_img_path).astype('float32') / 255.0
			crop_center, crop_border, ini_seg_global = attentionSubmodule.ExtractCenterline(global_img, 'GetAttentionMap')
			((left_upper_x, left_upper_y), (right_lower_x, right_lower_y)) = crop_border
			#remember to delete the weird_offset_i and weird_offset_j after finishing debugging
			#weird_offset_i, weird_offset_j = weird_offset_array[index, 0], weird_offset_array[index, 1]
			#current_clipped_img = current_global_img[left_upper_y + weird_offset_i: right_lower_y + 1 + weird_offset_i, left_upper_x + weird_offset_j: right_lower_x + 1 + weird_offset_j]
			current_clipped_img = current_global_img[left_upper_y: right_lower_y + 1, left_upper_x: right_lower_x + 1]
			clipped_result_generated_early_time = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis_UsingPrevious/' + str(index) + '_image_ori.png'
			#npt.assert_array_equal(current_clipped_img, imageio.imread(clipped_result_generated_early_time))
			attention_img = CreateAttentionFromSegmentation(ini_seg_global.astype('float32'), current_global_img.astype('float32'))
			attention_img_save_path = save_path + str(index) + '_attention_from_initial_segmentation.png'
			imageio.imwrite(attention_img_save_path, attention_img)
			ini_seg_global_save_path = save_path + str(index) + '_global_predict.png'
			imageio.imwrite(ini_seg_global_save_path, ini_seg_global)
			ini_seg_result_generated_early_time = path + str(seq) + '/segmented_using_previous_attention_and_current_img/' + str(index) + '_initial_segmentation.png'
			#npt.assert_array_equal(ini_seg_global, imageio.imread(ini_seg_result_generated_early_time))
			attention_result_generated_early_time = path + str(seq) + '/segmented_using_previous_attention_and_current_img/' + str(index) + '_attention_from_initial_segmentation.png'
			#npt.assert_array_equal(attention_img, imageio.imread(attention_result_generated_early_time))    
			imageio.imwrite(save_path + str(index) + '_input.png', current_clipped_img)            
            
            
            
			clip_img_ori = np.empty([1, 4, 192, 192], dtype=np.float32)
			for j in range(0, 4):
				input_img_path = save_path + str(index - j) + '_input.png'
				clip_img_ori[0, j][...] = imageio.imread(input_img_path).astype('float32') / 255.0
			centerline, ini_seg = iniSegmentationModule.ExtractCenterline(clip_img_ori, 'InitialSegmentationFromClipImage')
			
			# Revised July 17 2020
			
			#temppp = ini_seg_global[left_upper_y: right_lower_y + 1, left_upper_x: right_lower_x + 1] 
			#ini_seg[0, 0] = temppp.astype('float32') / 255.0
			# Revised July 17 2020
			
			#SaveImage(save_path + str(index) + '_predict.png', (ini_seg[0, 0]*255).astype(np.uint8))
			#verify_ini_seg = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_initial_segmentation.png')
			#npt.assert_array_equal((ini_seg[0, 0]*255).astype(np.uint8), verify_ini_seg)
			imageio.imwrite(save_path + str(index) + '_initial_seg.png', (ini_seg[0, 0]*255).astype(np.uint8))
			# then we incremental the image
			ini_seg_result = np.empty([1, 1, 192, 192], dtype=np.float32)
			ini_seg_result[0, 0][...] = (ini_seg[0, 0] * 255).astype(np.uint8).astype('float32') / 255.0
			centerline, ini_seg_result_incremental = incrementalModule.ExtractCenterline(ini_seg_result, 'IncrementAtEndPoints')
			ini_seg_result_extra_img = GetExtraImage(((ini_seg_result_incremental[0, 0]).astype(int)*255).astype(np.uint8), (ini_seg[0, 0]*255).astype(np.uint8))
			#verify_ini_seg_result_incremental = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_predicted_extra_incremental_based_on_initial_segmentation.png')
			#SaveImage(save_path + str(index) + '_iter0_incremental_extra.png', ini_seg_result_extra_img)    
			#npt.assert_array_equal(verify_ini_seg_result_incremental, ini_seg_result_extra_img)
			# then we perform the fine segmentation
			filtered_initial_segmentation = PrcoessImageBySymbol((ini_seg[0, 0] * 255).astype(np.uint8).astype(bool), True, 3, False, 0)
			filtered_extra_incremental_img = PrcoessImageBySymbol(ini_seg_result_extra_img.astype(bool), True, 2, True, 2)
			final_overlapped_img_for_fine_segmentation = np.logical_or(filtered_initial_segmentation, filtered_extra_incremental_img)
			extra_part_img_final = np.logical_xor(final_overlapped_img_for_fine_segmentation, filtered_initial_segmentation)           
			#imageio.imwrite(save_path + str(index) + '_iter0_data_extra.png', (extra_part_img_final.astype(int) * 255).astype(np.uint8))
			#verify_extra_part_img_final = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(index) + '_data_extra.png')
			#npt.assert_array_equal(verify_extra_part_img_final, (extra_part_img_final.astype(int) * 255).astype(np.uint8))
			fine_seg_input = np.empty([1, 2, 192, 192], dtype=np.float32)
			fine_seg_input[0, 0][...] = imageio.imread(save_path + str(index) + '_input.png').astype('float32') / 255.0
			fine_seg_input[0, 1][...] = (extra_part_img_final.astype(int) * 255).astype(np.uint8).astype('float32') / 255.0
			centerline, fine_seg_output = fineSegmentationModule.ExtractCenterline(fine_seg_input, 'FineSegIter')
			fine_seg_output = np.logical_or(((fine_seg_output[0, 0]).astype(int)*255).astype(np.uint8).astype(bool), (ini_seg[0, 0] * 255).astype(np.uint8).astype(bool))
			fine_seg_output = (fine_seg_output.astype(int) * 255).astype(np.uint8)
			#verify_fine_seg_output = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_iter0_seg.png')
			#verify_fine_seg_output = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(i + 3) + '_fine_seg_train_using_full_label_UsingSeperateChannel.png')
			#npt.assert_array_equal(verify_fine_seg_output, fine_seg_output)
			imageio.imwrite(save_path + str(index) + '_iter1_seg.png', fine_seg_output)
			"""
			iter0_seg_as_input_for_incremental = PrcoessImageBySymbol(fine_seg_output.astype(bool), True, 3, False, 0).astype(bool)
			verify_iter0_seg_as_input_for_incremental = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter1/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(i + 3) + '_data_initial.png')
			npt.assert_array_equal((iter0_seg_as_input_for_incremental.astype(int) * 255).astype(np.uint8), verify_iter0_seg_as_input_for_incremental)
			"""
			iter0_seg_input = np.empty([1, 1, 192, 192], dtype=np.float32)
			iter0_seg_input[0, 0][...] = fine_seg_output.astype('float32') / 255.0
			centerline, iter0_seg_output = incrementalModule.ExtractCenterline(iter0_seg_input, 'IncrementAtEndPoints')
			iter0_seg_output_extra_img = GetExtraImage(((iter0_seg_output[0, 0]).astype(int)*255).astype(np.uint8), fine_seg_output)
			#verify_iter0_seg_result_incremental = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter1/seq' + str(seq) + '/' + str(index) + '_predicted_extra_incremental_based_on_initial_segmentation.png')
			#SaveImage(save_path + str(index) + '_iter1_incremental_extra.png', iter0_seg_output_extra_img)    
			#npt.assert_array_equal(verify_iter0_seg_result_incremental, iter0_seg_output_extra_img)
			
			filtered_initial_segmentation_iter1 = PrcoessImageBySymbol(fine_seg_output.astype(bool), True, 3, False, 0)
			filtered_extra_incremental_img_iter1 = PrcoessImageBySymbol(iter0_seg_output_extra_img.astype(bool), True, 2, True, 2)
			final_overlapped_img_for_fine_segmentation_iter1 = np.logical_or(filtered_initial_segmentation_iter1, filtered_extra_incremental_img_iter1)
			extra_part_img_final_iter1 = np.logical_xor(final_overlapped_img_for_fine_segmentation_iter1, filtered_initial_segmentation_iter1)
			#imageio.imwrite(save_path + str(i + 3) + '_iter0_data_extra.png', (extra_part_img_final.astype(int) * 255).astype(np.uint8))
			#verify_extra_part_img_final_iter1 = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter1/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(index) + '_data_extra.png')
			#npt.assert_array_equal(verify_extra_part_img_final_iter1, (extra_part_img_final_iter1.astype(int) * 255).astype(np.uint8))
            
			fine_seg_input_iter1 = np.empty([1, 2, 192, 192], dtype=np.float32)
			fine_seg_input_iter1[0, 0][...] = imageio.imread(save_path + str(index) + '_input.png').astype('float32') / 255.0
			fine_seg_input_iter1[0, 1][...] = (extra_part_img_final_iter1.astype(int) * 255).astype(np.uint8).astype('float32') / 255.0
			centerline, fine_seg_output_iter1 = fineSegmentationModuleAtIter1.ExtractCenterline(fine_seg_input_iter1, 'FineSegIter1')
			fine_seg_output_iter1 = np.logical_or(((fine_seg_output_iter1[0, 0]).astype(int)*255).astype(np.uint8).astype(bool), fine_seg_output.astype(bool))
			fine_seg_output_iter1 = (fine_seg_output_iter1.astype(int) * 255).astype(np.uint8)
			#verify_fine_seg_output = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(i + 4 - 1) + '_iter0_seg.png') 
			#verify_fine_seg_output_iter1 = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '/' + str(index) + '_iter1_segmentation.png')
			#verify_fine_seg_output = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(i + 3) + '_fine_seg_train_using_full_label_UsingSeperateChannel.png')
			#npt.assert_array_equal(verify_fine_seg_output_iter1, fine_seg_output_iter1)
			imageio.imwrite(save_path + str(index) + '_iter2_seg.png', fine_seg_output_iter1)
            
			iter1_seg_input = np.empty([1, 1, 192, 192], dtype=np.float32)
			iter1_seg_input[0, 0][...] = fine_seg_output_iter1.astype('float32') / 255.0
			centerline, iter1_seg_output = incrementalModule.ExtractCenterline(iter1_seg_input, 'IncrementAtEndPoints')
			iter1_seg_output_extra_img = GetExtraImage(((iter1_seg_output[0, 0]).astype(int)*255).astype(np.uint8), fine_seg_output_iter1)
			#verify_iter1_seg_result_incremental = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '/' + str(index) + '_predicted_extra_incremental_based_on_iter0_segmentation.png')
			#SaveImage(save_path + str(i + 3) + '_iter1_incremental_extra.png', iter0_seg_output_extra_img)    
			#npt.assert_array_equal(verify_iter1_seg_result_incremental, iter1_seg_output_extra_img)
			
			filtered_initial_segmentation_iter2 = PrcoessImageBySymbol(fine_seg_output_iter1.astype(bool), True, 3, False, 0)
			filtered_extra_incremental_img_iter2 = PrcoessImageBySymbol(iter1_seg_output_extra_img.astype(bool), True, 2, True, 4)
			final_overlapped_img_for_fine_segmentation_iter2 = np.logical_or(filtered_initial_segmentation_iter2, filtered_extra_incremental_img_iter2)
			extra_part_img_final_iter2 = np.logical_xor(final_overlapped_img_for_fine_segmentation_iter2, filtered_initial_segmentation_iter2)
			#imageio.imwrite(save_path + str(i + 3) + '_iter0_data_extra.png', (extra_part_img_final.astype(int) * 255).astype(np.uint8))
			#verify_extra_part_img_final_iter2 = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(index) + '_data_extra.png')
			#imageio.imwrite('3_iter2_data_extra.png', (extra_part_img_final_iter2.astype(int) * 255).astype(np.uint8))
			#npt.assert_array_equal(verify_extra_part_img_final_iter2, (extra_part_img_final_iter2.astype(int) * 255).astype(np.uint8))


			#last_iter_seg_name = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '/' + str(index) + '_iter1_segmentation.png'
			#data_img_name = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(index) + '_data_extra.png'
			#ori_img_name = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '/' + str(index) + '_image_ori.png'       
			final_relaxtion_input = np.empty([1, 2, 192, 192], dtype=np.float32)
			final_relaxtion_input[0, 0][...] = imageio.imread(save_path + str(index) + '_input.png').astype('float32') / 255.0
			final_relaxtion_input[0, 1][...] = (extra_part_img_final_iter2.astype(int) * 255).astype(np.uint8).astype('float32') / 255.0
			centerline, final_relaxtion_output = finalRelaxtionSubmodule.ExtractCenterline(final_relaxtion_input, 'FinalRelaxtionToIncreaseSubtleness')
			final_seg_img_without_filtering = np.logical_or(((final_relaxtion_output[0, 0]).astype(float) * 255).astype(np.uint8).astype(bool), fine_seg_output_iter1.astype(bool))
			#verify_final_seg_img_without_filtering = imageio.imread('/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis_UsingPrevious/' + str(index) + '_image_seg.png')
			#npt.assert_array_equal(verify_final_seg_img_without_filtering, (final_seg_img_without_filtering.astype(float) * 255).astype(np.uint8))
			imageio.imwrite(save_path + str(index) + '_iter3_relax.png', (final_seg_img_without_filtering.astype(float) * 255).astype(np.uint8))
			#initial_clip_seg = imageio.imread( '/home/zsj/data/CTSAWorkspace/sequence' + str(seq) + '/clip_seg_multi_previous_using_all_train_and_valid_as_train_verify/' + str(index) + '_predict.png')
			final_seg_result = finalPostProcessingFilter((final_seg_img_without_filtering.astype(float) * 255).astype(np.uint8), (ini_seg[0, 0] * 255).astype(np.uint8), fine_seg_output_iter1)
			#verify_final_seg_result = imageio.imread(path + str(seq) + '/all_module_integrate/' + str(index) + '_result_intersect_without_weird_offset.png')
			#verify_final_seg_result = imageio.imread(save_path + str(index) + '_result_intersect_without_weird_offset_without_recursive_finding_connectivity.png')
			#npt.assert_array_equal(final_seg_result, verify_final_seg_result)
			imageio.imwrite(save_path + str(index) + '_result_intersect_without_weird_offset_without_recursive_finding_connectivity.png', final_seg_result)
			clip_offset_array[index, 0], clip_offset_array[index, 1], clip_offset_array[index, 2], clip_offset_array[index, 3] = copy.copy(left_upper_y), copy.copy(right_lower_y + 1), copy.copy(left_upper_x), copy.copy(right_lower_x + 1)
			#clip_offset_array[index, 0], clip_offset_array[index, 1], clip_offset_array[index, 2], clip_offset_array[index, 3] = copy.copy(left_upper_y + weird_offset_i), copy.copy(right_lower_y + 1 + weird_offset_i), copy.copy(left_upper_x + weird_offset_j), copy.copy(right_lower_x + 1 + weird_offset_j)
		np.save('/home/zsj/data/clip_offset_array_without_weird_offset_without_recursive_finding_connectivity_' + str(seq) + '.npy', clip_offset_array)