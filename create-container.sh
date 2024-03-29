LOGICNETS_PATH="/D:/source/logicnets-nas"
LOGICNETS_MOUNT_POINT="/workspace/logicnets"
winpty docker run --gpus all -v ${LOGICNETS_PATH}:${LOGICNETS_MOUNT_POINT} -it c981c0bd411cbd045636b2fc1736db2cbafa28b758cae2301ebc9b80fb32ced9

