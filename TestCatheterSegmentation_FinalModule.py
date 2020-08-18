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

np.random.seed(987654)
random.seed(1234569)

from UtilImage import *
from FluoroExtraction import *

GENERATED_PATH = "generated/"
DATA_PATH = "../generateTrainTestDataset/generated/"
WEIGHT_FILE = "../trainCatheterSegmentation/generated/bestTrainWeight_Attention.h5"

EQUALIZATION = NORMALIZE_CONTRAST_STRETCHING

FORMAT_CONFIG = "{0:04d}"

DATA_DCM_FILE = 0
DATA_FRAME_ID = 1

NB_GENERATED_SEQUENCES = 20
dataList = []
for i in range(0,NB_GENERATED_SEQUENCES):
	dataList.append([DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + ".dcm", 3])

fluoroExtraction = FluoroExtraction(WEIGHT_FILE)

def FindCenter(img):
	
	rows, cols = img.shape[0], img.shape[1]
	x = np.ones((rows, 1)) * np.array(range(1, cols + 1)).astype('float32').reshape((1, -1))
	y = np.array(range(1, rows + 1)).astype('float32').reshape((-1, 1)) * np.ones((1, cols))
	area = np.sum(img[:,:]) 
	meanx = np.sum((img * x)[:,:]) / area 
	meany = np.sum((img * y)[:,:]) / area 
	return meanx - 1, meany - 1

def GenerateCircleList(radius):
	circle_list = []
	x, y = 0, 0
	for i in range(x - radius, x + radius + 1):
		for j in range(y - radius, y + radius + 1):
			if ((i - x)**2 + (j - y)**2) <= radius**2:
				circle_list.append((i, j))
	circle_np = np.array(circle_list)
	return circle_np

def CreateAttentionFromSegmentationAndSave(initial_segmentation_img, ori_img, save_path, draw_zero_index, radius, kernel_num, kernel_size):
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

path = '/home/zsj/data/CTSAWorkspace/sequence'
data_list = []
radius, kernel_num, kernel_size = 100, 5, (11, 11)
draw_index_zero = GenerateCircleList(radius)
NB_CHANNEL = 4
for seq in range(22, 23):
	path_seq = path + str(seq) + '/icon/'
	imgs_in_seq = os.listdir(path_seq)
	imgs_in_seq.sort()
	for i in range(0, len(imgs_in_seq) - NB_CHANNEL + 1):
		X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
		for j in range(0, NB_CHANNEL):
			if j == 0:
				if os.path.exists(path + str(seq) + '/icon/' + imgs_in_seq[i + NB_CHANNEL - 1 - j]) == False:
					print('not exist ********************************** ')
				X[0, j][...] = imageio.imread(path + str(seq) + '/icon/' + imgs_in_seq[i + NB_CHANNEL - 1 - j]).astype('float32') / 255.0
			else:
				previous_attention_img_path = ''
				if i + NB_CHANNEL - 1 - j < NB_CHANNEL - 1:
					previous_attention_img_path = path + str(seq) + '/attention_img_using_GT_segmentation_radius100/' + imgs_in_seq[i + NB_CHANNEL - 1 - j]
				else:
					previous_attention_img_path = path + str(seq) + '/segmented_using_previous_attention_and_current_img_verify/' + str(i + NB_CHANNEL - 1 - j) + '_attention_from_initial_segmentation.png'
				if os.path.exists(previous_attention_img_path) == False:
					print('not exist ********************************** ')
				X[0, j][...] = imageio.imread(previous_attention_img_path).astype('float32') / 255.0
		centerline, Y = fluoroExtraction.ExtractCenterline(X)
		save_path = path + str(seq) + '/segmented_using_previous_attention_and_current_img_verify/'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if i == 0:
			for k in range(0, NB_CHANNEL - 1):
				SaveImage(save_path + str(k) + '_attention_from_initial_segmentation.png', (X[0, NB_CHANNEL - 1 - k]*255).astype(np.uint8))
				#SaveImage(save_path + str(i + NB_CHANNEL - 1) + '_input.png', (X[0, j]*255).astype(np.uint8))
		ini_seg_path = save_path + str(i + NB_CHANNEL - 1) + '_initial_segmentation.png'
		SaveImage(ini_seg_path, (Y[0, 0] * 255).astype(np.uint8))
		initial_segmentation_img = imageio.imread(ini_seg_path).astype('float32')
		ori_img = imageio.imread(path + str(seq) + '/icon/' + imgs_in_seq[i + NB_CHANNEL - 1]).astype('float32')
		attention_img_save_path = save_path + str(i + NB_CHANNEL - 1) + '_attention_from_initial_segmentation.png'
		attention_img = CreateAttentionFromSegmentationAndSave(initial_segmentation_img, ori_img, attention_img_save_path, draw_zero_index, radius, kernel_num, kernel_size)
		imageio.imwrite(attention_img_save_path, attention_img)
		attention_result_generated_early_time = path + str(seq) + '/segmented_using_previous_attention_and_current_img/' + str(i + NB_CHANNEL - 1) + '_attention_from_initial_segmentation.png' 
		npt.assert_array_equal(imageio.imread(attention_img_save_path), imageio.imread(attention_result_generated_early_time))
		ini_seg_result_generated_early_time = path + str(seq) + '/segmented_using_previous_attention_and_current_img/' + str(i + NB_CHANNEL - 1) + '_initial_segmentation.png'
		npt.assert_array_equal(imageio.imread(ini_seg_path), imageio.imread(ini_seg_result_generated_early_time))      