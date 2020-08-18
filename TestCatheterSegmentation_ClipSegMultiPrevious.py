import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import random
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

np.random.seed(987654)
random.seed(1234569)

from UtilImage import *
from FluoroExtraction import *

GENERATED_PATH = "generated/"
DATA_PATH = "../generateTrainTestDataset/generated/"
WEIGHT_FILE = "../trainCatheterSegmentation/generated/bestTrainWeightClipSegMultiPrevious.h5"

EQUALIZATION = NORMALIZE_CONTRAST_STRETCHING

FORMAT_CONFIG = "{0:04d}"

DATA_DCM_FILE = 0
DATA_FRAME_ID = 1

NB_GENERATED_SEQUENCES = 20
dataList = []
for i in range(0,NB_GENERATED_SEQUENCES):
	dataList.append([DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + ".dcm", 3])

fluoroExtraction = FluoroExtraction(WEIGHT_FILE)


"""
ini_pred = Y[0, 0] * 255.0
atten_pred = ini_pred
i = 0
while i < 1:
	atten_pred = cv2.GaussianBlur(atten_pred,(2,2),2)
	#print('*****', np.max(atten_pred[:,:]))
	atten_pred = atten_pred * 255 / np.max(atten_pred[:,:])
	#atten_pred = atten_pred.astype(np.uint8)
	#print('-----', np.max(atten_pred[:,:]))
	i = i + 1
#imageio.imwrite('/home/zsj/data/check/atten_pred' + str(loop) + '.png', atten_pred.astype(np.uint8))
#print(np.max(atten_pred[:, :]))
atten_map = atten_pred / 255.0
final_img = ((1.0 - atten_map) * ori_img)
loop = loop + 1
imageio.imwrite('/home/zsj/data/check/final' + str(loop) + '.png', final_img.astype(np.uint8))
"""


path = '/home/zsj/data/CTSAWorkspace/sequence'
data_list = []
for seq in range(1, 10):
		path_seq = path + str(seq) + '/icon/'
		imgs_in_seq = os.listdir(path_seq)
		imgs_in_seq.sort()
		for i in range(0, len(imgs_in_seq) - NB_CHANNEL + 1):
			X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
			#print('predicting -------------------  ')
			for j in range(0, NB_CHANNEL):
				X[0, j][...] = imageio.imread(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + NB_CHANNEL - 1 - j]).astype('float32') / 255.0
				print(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + NB_CHANNEL - 1 - j])
				if os.path.exists(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + NB_CHANNEL - 1 - j]) == False:
					print('not exist ********************************** ')
			centerline, Y = fluoroExtraction.ExtractCenterline(X)
			save_path = path + str(seq) + '/clip_seg_multi_previous/'
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			for j in range(1):
				if i == 0:
					for k in range(0, NB_CHANNEL - 1):
						SaveImage(save_path + str(k) + '_input.png', (X[0, NB_CHANNEL - 1 - k]*255).astype(np.uint8))
				SaveImage(save_path + str(i + NB_CHANNEL - 1) + '_input.png', (X[0, j]*255).astype(np.uint8))
			SaveImage(save_path + str(i + NB_CHANNEL - 1) + '_predict.png', (Y[0, 0]*255).astype(np.uint8))
			rgbImage = GrayToRGB(X[0, 0])
			rgbImage = np.moveaxis(rgbImage, 2, 0)
			DrawCenterline(rgbImage, centerline, [0., 0.5, 1.], [1., 0., 0.], _size=4., _hls=True)
			rgbImage = np.moveaxis(rgbImage, 0, -1)
			SaveImage(save_path + str(i + NB_CHANNEL - 1) + '_predict_centerline.png', (rgbImage*255).astype(np.uint8))


"""
TOTAL_SIZE = len(data_list)

for i in range(TOTAL_SIZE):
	#assert(dataList[i][DATA_FRAME_ID] >= NB_CHANNEL - 1)
	#assert(IsFileExist(dataList[i][DATA_DCM_FILE]) == True)
	#dcmInfo = ReadOnlyDicomInfo(dataList[i][DATA_DCM_FILE])
	#print("dcmInfo.Columns " + str(dcmInfo.Columns) + " dcmInfo.Rows " + str(dcmInfo.Rows) + " dcmInfo.NumberOfFrames " + str(dcmInfo.NumberOfFrames))
	X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
	for j in range(NB_CHANNEL):
		#currentFrameId = dataList[i][DATA_FRAME_ID] - j
		#frame = GetFloat32DicomFrame(dataList[i][DATA_DCM_FILE], currentFrameId, _normalize=EQUALIZATION)
		#X[i, 0, j][...] = frame
		#X[0, j][...] = imageio.imread('/home/zsj/data/debug/resize/ori_' + str(i) + '_' + str(NB_CHANNEL-1-j) + '.png').astype('float32') / 255.0
		X[0, j][...] = imageio.imread(dataset_list[imgId][NB_CHANNEL-1-j]).astype('float32') / 255.0
		#new_imgg = imageio.imread('/home/zsj/data/debug/resize/ori_' + str(i) + '_' + str(NB_CHANNEL-1-j) + '.png')
		#frame_newg = cv2.resize((frame * 255).astype(np.uint8), (512, 512), interpolation = cv2.INTER_CUBIC).astype(np.uint8) 
		#np.testing.assert_equal(new_imgg, frame_newg)
		#print('equal')
	centerline, Y = fluoroExtraction.ExtractCenterline(X)
	for j in range(1):
		SaveImage(GENERATED_PATH + "dcm_" + str(i) + "_X" + str(j) + ".png", (X[0, j]*255).astype(np.uint8))
	SaveImage(GENERATED_PATH + "dcm_" + str(i) + "_Y0.png", (Y[0, 0]*255).astype(np.uint8))
	rgbImage = GrayToRGB(X[0, 0])
	rgbImage = np.moveaxis(rgbImage, 2, 0)
	DrawCenterline(rgbImage, centerline, [0., 0.5, 1.], [1., 0., 0.], _size=4., _hls=True)
	rgbImage = np.moveaxis(rgbImage, 0, -1)
	SaveImage(GENERATED_PATH + "dcm_" + str(i) + "_X0centerline.png", (rgbImage*255).astype(np.uint8))
"""


"""
X = np.empty([1, NB_CHANNEL, SIZE_Y, SIZE_X], dtype=np.float32)
ori_img = imageio.imread('/home/zsj/data/CTSAWorkspace/sequence22/icon/navi00000235.png').astype('float32')
N = 10
loop = 0
last_Y = np.zeros((SIZE_Y, SIZE_X))
while loop < N:
	if loop == 0:
		X[0, 0][...] = imageio.imread('/home/zsj/data/CTSAWorkspace/sequence22/icon/navi00000235.png').astype('float32') / 255.0
	else:
		X[0, 0][...] = imageio.imread('/home/zsj/data/check/final' + str(loop) + '.png').astype('float32') / 255.0 
	X[0, 1][...] = imageio.imread('/home/zsj/data/CTSAWorkspace/sequence22/icon/navi00000234.png').astype('float32') / 255.0
	X[0, 2][...] = imageio.imread('/home/zsj/data/CTSAWorkspace/sequence22/icon/navi00000233.png').astype('float32') / 255.0
	X[0, 3][...] = imageio.imread('/home/zsj/data/CTSAWorkspace/sequence22/icon/navi00000232.png').astype('float32') / 255.0
	centerline, Y = fluoroExtraction.ExtractCenterline(X)
	if loop == 0:
		last_Y = Y
	else:
		Y = np.logical_xor(Y, last_Y).astype('float32')  
	#print(centerline, type(centerline), Y[0, 0].dtype, np.max(Y[0, 0, :, :]))
	ini_pred = Y[0, 0] * 255.0
	#print('*****', np.max(Y[0, 0]))
	#imageio.imwrite('/home/zsj/data/check/ini_pred' + str(loop) + '.png', ini_pred.astype(np.uint8))
	atten_pred = ini_pred
	i = 0
	while i < 1:
		atten_pred = cv2.GaussianBlur(atten_pred,(2,2),2)
		#print('*****', np.max(atten_pred[:,:]))
		atten_pred = atten_pred * 255 / np.max(atten_pred[:,:])
		#atten_pred = atten_pred.astype(np.uint8)
		#print('-----', np.max(atten_pred[:,:]))
		i = i + 1
	#imageio.imwrite('/home/zsj/data/check/atten_pred' + str(loop) + '.png', atten_pred.astype(np.uint8))
	#print(np.max(atten_pred[:, :]))
	atten_map = atten_pred / 255.0
	final_img = ((1.0 - atten_map) * ori_img)
	loop = loop + 1
	imageio.imwrite('/home/zsj/data/check/final' + str(loop) + '.png', final_img.astype(np.uint8))   
print('done ***************')
"""