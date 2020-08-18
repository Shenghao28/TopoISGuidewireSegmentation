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

# the following code is used for previous attention added segmentation
path = '/home/zsj/data/CTSAWorkspace/sequence'
NB_CHANNEL = 4
data_list = []
for seq in range(1, 10):
	path_seq = path + str(seq) + '/icon/'
	imgs_in_seq = os.listdir(path_seq)
	imgs_in_seq.sort()
	img_data = []
	for ind in range(3, len(imgs_in_seq)):
		data = []
		data_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/seq' + str(seq) + '_vis/' + str(ind) + '_iter2_seg_Iter2Item20Epoch150Final.png'
		data.append(data_path)
		for j in range(1, NB_CHANNEL):
			previous_gt_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(ind - j) + '_human_annotated.png'
			data.append(previous_gt_path)
			if os.path.exists(previous_gt_path) == False:
				print('not exist ********************* ', previous_gt_path)  
		label_path = '/home/zsj/data/CTSAWorkspace/clipped_incremental_learning/iter2/train_dataset_for_fine_segmentation/seq' + str(seq) + '/' + str(ind) + '_human_annotated.png'
		data.append(label_path)
		if os.path.exists(label_path) == False:
			print('not exist *****************', label_path)
		data_list.append(data)

print(data_list)

TOTAL_SIZE = len(data_list)
#train_data_ind = np.load(TRAIN_id_path).astype(int)
#validation_data_ind = np.load(VALIDATION_id_path).astype(int)
TRAIN_dataset = []
for i in range(TOTAL_SIZE):
	TRAIN_dataset.append(data_list[i])
VALIDATION_dataset = []
for i in range(TOTAL_SIZE):
	VALIDATION_dataset.append(data_list[i])

# warning if you change dataList, remove the files X.h5 and Y.h5. If not, the previous X.h5 and Y.h5 files will be loaded.
#dataObject = FluoroDataObject(dataList, SIZE_X, SIZE_Y, NB_CHANNEL, _savePath = GENERATED_PATH, _pctTrainingSet = 0.5)

fluoroExtraction = FluoroExtraction()

nbUsedChannel = NB_CHANNEL
nbEpoch = 200
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

modelCheckpoint = ModelCheckpoint(GENERATED_PATH + "bestTrainWeight_SegFilterBasedOnPrevious" + ".h5", monitor='loss', save_best_only=True, save_weights_only=True)
modelCheckpointTest = ModelCheckpoint(GENERATED_PATH + "bestTestWeight_SegFilterBasedOnPrevious" + ".h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
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
fluoroExtraction.m_Model.save_weights(GENERATED_PATH + "lastweight_SegFilterBasedOnPrevious" + ".h5")
SavePickle(history.history, GENERATED_PATH + "history_SegFilterBasedOnPrevious" + ".pickle")