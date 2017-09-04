#!/bin/sh

## MODIFY PATH for YOUR SETTING
ROOT_DIR=

CAFFE_DIR=./
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=drusen_area

if [ "${EXP}" = "voc12" ]; then
    NUM_LABELS=21
    DATA_ROOT=${ROOT_DIR}/home/rasha/Downloads/VOCdevkit/VOC2012
else
    NUM_LABELS=2
#	DATA_ROOT=${ROOT_DIR}/home/rasha/Desktop/test_area
	DATA_ROOT=
#    DATA_ROOT=${ROOT_DIR}/home/rasha/OCT-Data
#    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
NET_ID=RESNET-50
#NET_ID=MSc-COCO-LargeFOV

## Variables used for weakly or semi-supervisedly training
#TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug

#TRAIN_SET_STRONG=train
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

DEV_ID=1

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=1
RUN_TEST=0
RUN_TRAIN2=0
RUN_TEST2=0
RUN_TEST_SNAPSHOTS=0
TEMP=0


## Training #1 (on train_aug)

if [ ${TEMP} -eq 1 ]; then
a=100
while [ $a -lt 7200 ]
do
   echo $a
   a=`expr $a + 1000`
done
fi 

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel
#	MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_119224.caffemodel`
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
#		CMD="${CMD}" |& tee fold1_log.txt 
		echo Running ${CMD} && ${CMD}
fi

#--gpu=${DEV_ID} \

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
#				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
#                TEST_ITER=`cat /home/rasha/Desktop/DataFromUniClinic/Multi-fold-training-data/Fold01/train_files_one.txt | wc -l`
				TEST_ITER=`cat ${EXP}/list/val_one.txt | wc -l`
#				TEST_ITER=`cat ${EXP}/list/val_fold1_weights.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/test.caffemodel
				if [ ! -f ${MODEL} ]; then
#						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
#						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_119224.caffemodel`
#						MODEL=`ls -t ${EXP}/model/resnet-101-org-no-weights-all-data/train_iter_*.caffemodel | head -n 1`
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_750.caffemodel`
				fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

## Test multiple snapshots

if [ ${RUN_TEST_SNAPSHOTS} -eq 1 ]; then
    #
    for TEST_SET in val; do
		SNAPSHOT=100
		while [ ${SNAPSHOT} -lt 7200 ]
			do
				TEST_ITER=`cat ${EXP}/list/val_small.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/train_iter_${SNAPSHOT}.caffemodel
	#			echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/small/snapshot_$SNAPSHOT
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}_snapshot_$SNAPSHOT.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}_snapshot_$SNAPSHOT.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
			echo ${MODEL}
			echo Running ${CMD} && ${CMD}
			SNAPSHOT=`expr $SNAPSHOT + 1000`
			echo Done
		done
    done
fi


## Training #2 (finetune on trainval_aug)

if [ ${RUN_TRAIN2} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=trainval${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init2.caffemodel
    if [ ! -f ${MODEL} ]; then
				MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
		echo Running ${CMD} && ${CMD}
fi

## Test #2 on official test set

if [ ${RUN_TEST2} -eq 1 ]; then
    #
    for TEST_SET in val test; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
				fi
				#
				echo Testing2 net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features2/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi
