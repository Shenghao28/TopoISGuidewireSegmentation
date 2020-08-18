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
WEIGHT_FILE = "../trainCatheterSegmentation/generated/bestTrainWeight_SegFilterBasedOnPrevious.h5"

EQUALIZATION = NORMALIZE_CONTRAST_STRETCHING

FORMAT_CONFIG = "{0:04d}"

DATA_DCM_FILE = 0
DATA_FRAME_ID = 1

NB_GENERATED_SEQUENCES = 20
dataList = []
for i in range(0,NB_GENERATED_SEQUENCES):
	dataList.append([DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + ".dcm", 3])

fluoroExtraction = FluoroExtraction(WEIGHT_FILE)


path = '/home/zsj/data/CTSAWorkspace/sequence'
NB_CHANNEL = 2
data_list = []
for seq in range(1, 10):
	path_seq = path + str(seq) + '/icon/'
	imgs_in_seq = os.listdir(path_seq)
	imgs_in_seq.sort()
	for ind in range(3, len(imgs_in_seq)):
		X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
		data_img_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + str(ind) + '_iter2_seg_Iter2Item20Epoch150Final.png'
		X[0, 0][...] = imageio.imread(data_img_path).astype('float32') / 255.0
		ori_img_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + str(ind) + '_image_ori.png'
		X[0, 1][...] = imageio.imread(ori_img_path).astype('float32') / 255.0
		centerline, Y = fluoroExtraction.ExtractCenterline(X)
		filtered_seg_img_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + str(ind) + '_filtered_UsingSegAndOri.png'
		imageio.imwrite(filtered_seg_img_path, ((Y[0, 0]).astype(float)*255).astype(np.uint8).astype(bool).astype(np.uint8))