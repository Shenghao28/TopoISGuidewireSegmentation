import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import random

np.random.seed(987654)
random.seed(1234569)

from File import *
from FluoroDataObject import *
from DataAugmentation import *
from FluoroExtraction import *

GENERATED_PATH = "generated/"
DATA_PATH = "../generateTrainTestDataset/generated/"

FORMAT_CONFIG = "{0:04d}"

# load training data [SET_PROCEDURE, SET_FLUORO_FILE, SET_FLUORO_FRAME, SET_FLUORO_CENTERLINE, SET_FLUORO_INFO] and save it in X.h5 and Y.h5
NB_GENERATED_SEQUENCES = 20
dataList = []
for i in range(NB_GENERATED_SEQUENCES):
	dataList.append([i, DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + ".dcm", 3, DATA_PATH + "sequence" + str(FORMAT_CONFIG).format(i) + "centerline" + str(FORMAT_CONFIG).format(3) + ".txt"])

# the following code is used for previous attention added segmentation
path = '/home/zsj/data/CTSAWorkspace/sequence'
NB_CHANNEL = 4
data_list = []
for seq in range(1, 10):
	path_seq = path + str(seq) + '/icon/'
	imgs_in_seq = os.listdir(path_seq)
	imgs_in_seq.sort()
	img_data = []
	for i in range(0, len(imgs_in_seq) - NB_CHANNEL + 1):
		data = []
		for j in range(0, NB_CHANNEL - 1):
			data.append(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + j])
			if os.path.exists(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + j]) == False:
				print('not exist ********************* ')
		data.append(path + str(seq) + '/clipped_img/' + imgs_in_seq[i + NB_CHANNEL - 1])
		data.append(path + str(seq) + '/clipped_label/' + imgs_in_seq[i + NB_CHANNEL - 1])
		if os.path.exists(path + str(seq) + '/clipped_label/' + imgs_in_seq[i + NB_CHANNEL - 1]) == False:
			print('not exist *****************')
		data_list.append(data)


TOTAL_SIZE = len(data_list)
train_data_ind = np.load(TRAIN_id_path).astype(int)
validation_data_ind = np.load(VALIDATION_id_path).astype(int)
TRAIN_dataset = []
for i in range(train_data_ind.shape[0]):
    TRAIN_dataset.append(data_list[train_data_ind[i]])
VALIDATION_dataset = []
for i in range(validation_data_ind.shape[0]):
    VALIDATION_dataset.append(data_list[validation_data_ind[i]])

# warning if you change dataList, remove the files X.h5 and Y.h5. If not, the previous X.h5 and Y.h5 files will be loaded.
#dataObject = FluoroDataObject(dataList, SIZE_X, SIZE_Y, NB_CHANNEL, _savePath = GENERATED_PATH, _pctTrainingSet = 0.5)

fluoroExtraction = FluoroExtraction()

nbUsedChannel = NB_CHANNEL
nbEpoch = 300
batchsize = 4
EPOCH_SIZE = len(TRAIN_dataset)
print('epoch_size ---------------- ', EPOCH_SIZE, )
is3Dshape = False
# data augmentation
nbData = -1
keepPctOriginal = 0.5
trans = 0.16 # +/- proportion of the image size
rot = 9 # +/- degree
zoom = 0.12 # +/- factor
shear = 0 # +/- in radian x*np.pi/180
elastix = 0 # in pixel
intensity = 0.07 # +/- factor
hflip = True
vflip = True

modelCheckpoint = ModelCheckpoint(GENERATED_PATH + "bestTrainWeight_ClipSegMultiPrevious" + ".h5", monitor='loss', save_best_only=True, save_weights_only=True)
modelCheckpointTest = ModelCheckpoint(GENERATED_PATH + "bestTestWeight_ClipSegMultiPrevious" + ".h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
checkpointLists = [modelCheckpoint, modelCheckpointTest]



def ImgGenerator():
	for image in GenerateImageOnTheFly(TRAIN_dataset, SIZE_Y, SIZE_X, NB_CHANNEL, None, 0, None, None, None, _batchSize=batchsize, _epochSize=EPOCH_SIZE, _nbData=nbData, _keepPctOriginal=keepPctOriginal, _trans=trans, _rot=rot, _zoom=zoom, _shear=shear, _elastix=elastix, _intensity=intensity, _hflip=hflip, _vflip=vflip, _3Dshape=is3Dshape):
		yield image # ((batchsize, SIZE_X, SIZE_Y, NB_CHANNEL),(batchsize, SIZE_X, SIZE_Y, 1))

# with new tensorflow, we have to use tf.data for the evaluation (also advised for the train data as well). Here the generator manage the batchsize, so we have to give him only the data not the batch
def ValidationGenerator():
	for image in GenerateValidationOnTheFly(VALIDATION_dataset, SIZE_Y, SIZE_X, NB_CHANNEL, None, 0, None, None, None, _batchSize=batchsize, _3Dshape=is3Dshape):
		yield image # ((SIZE_X, SIZE_Y, NB_CHANNEL),(SIZE_X, SIZE_Y, 1))

validation_generator_dataset = tf.data.Dataset.from_generator(ValidationGenerator, output_types=(tf.float32,tf.float32), output_shapes=(tf.TensorShape((SIZE_Y, SIZE_X, 4)), tf.TensorShape((SIZE_Y, SIZE_X, 1)))).batch(batchsize)

stepsPerEpoch = math.ceil(EPOCH_SIZE/batchsize)
#validationSteps = math.ceil(len(dataObject.m_TestSetList)/batchsize)
validationSteps = math.ceil(len(VALIDATION_dataset)/batchsize)

history = fluoroExtraction.m_Model.fit(x=ImgGenerator(),y=None, batch_size=None, epochs=nbEpoch, verbose=2, callbacks=checkpointLists, validation_data=validation_generator_dataset, steps_per_epoch=stepsPerEpoch, validation_steps=validationSteps, workers=1)
fluoroExtraction.m_Model.save_weights(GENERATED_PATH + "lastweight_ClipSegMultiPrevious" + ".h5")
SavePickle(history.history, GENERATED_PATH + "history_ClipSegMultiPrevious" + ".pickle")