import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import random
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import copy

np.random.seed(987654)
random.seed(1234569)

from UtilImage import *
from FluoroExtraction import *

GENERATED_PATH = "generated/"
DATA_PATH = "../generateTrainTestDataset/generated/"
WEIGHT_FILE = "../trainCatheterSegmentation/generated/bestTrainWeight_IncrementalLearning.h5"

EQUALIZATION = NORMALIZE_CONTRAST_STRETCHING

FORMAT_CONFIG = "{0:04d}"

DATA_DCM_FILE = 0
DATA_FRAME_ID = 1

NB_GENERATED_SEQUENCES = 20
dataList = []
for i in range(0,NB_GENERATED_SEQUENCES):
	dataList.append([DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + ".dcm", 3])

fluoroExtraction = FluoroExtraction(WEIGHT_FILE)

def VisPredictedImage(predicted_img, data_img):
	overlapped_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3)).astype(np.uint8)
	extra_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1])).astype(np.uint8)
	mixed_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1])).astype(np.uint8)
	for i in range(predicted_img.shape[0]):
		for j in range(predicted_img.shape[1]):
			if predicted_img[i, j] > 125 and data_img[i, j] < 125:
				overlapped_img[i, j, 0] = 0
				overlapped_img[i, j, 1] = 255
				overlapped_img[i, j, 2] = 0
				extra_img[i, j] = 255
				mixed_img[i, j] = 255
			elif predicted_img[i, j] > 125 and data_img[i, j] > 125:
				overlapped_img[i, j, 0] = 255
				overlapped_img[i, j, 1] = 255
				overlapped_img[i, j, 2] = 0
				mixed_img[i, j] = 255
			elif predicted_img[i, j] < 125 and data_img[i, j] > 125:
				overlapped_img[i, j, 0] = 255
				overlapped_img[i, j, 1] = 0
				overlapped_img[i, j, 2] = 0
				mixed_img[i, j] = 255
	return extra_img, mixed_img, overlapped_img
	
path = '/home/zsj/data/CTSAWorkspace/sequence'
NB_CHANNEL = 1
data_list = []
for seq in range(22, 23):
	path_seq = path + str(seq) + '/icon/'
	imgs_in_seq = os.listdir(path_seq)
	imgs_in_seq.sort()
	img_data = []
	for index in range(3, len(imgs_in_seq)):
		print(index)   
		for kind in range(100):
			data_img_name = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(index) + '_data_initial.png'
			#if os.path.exists(data_img_name) == False:
			#	break
        
			X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
			X[0, 0][...] = imageio.imread(data_img_name).astype('float32') / 255.0
			centerline, Y = fluoroExtraction.ExtractCenterline(X)
			predicted_extra_incremental_based_on_initial_segmentation_path = \
				'/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_predicted_extra_incremental_based_on_initial_segmentation.png'
			predicted_mixed_incremental_based_on_initial_segmentation_path = \
				'/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_predicted_mixed_incremental_based_on_initial_segmentation.png'
			predicted_overlapped_incremental_based_on_initial_segmentation_path = \
				'/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter0/seq' + str(seq) + '/' + str(index) + '_predicted_overlapped_incremental_based_on_initial_segmentation.png'
			extra_img, mixed_img, overlapped_img = VisPredictedImage(((Y[0, 0]).astype(int)*255).astype(np.uint8), imageio.imread(data_img_name))
			imageio.imwrite(predicted_extra_incremental_based_on_initial_segmentation_path, extra_img)
			imageio.imwrite(predicted_mixed_incremental_based_on_initial_segmentation_path, mixed_img)
			imageio.imwrite(predicted_overlapped_incremental_based_on_initial_segmentation_path, overlapped_img)
			break