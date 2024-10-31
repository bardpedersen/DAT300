USR_NAME="dat300-h24-30"
HOST_NAME="${USR_NAME}@filemanager.orion.nmbu.no"

# Note here that we have an extra slash at the end of the directory we are copying from, and we don't have it at the end of the directory we are copying to

REMOTE_DIR1="${HOST_NAME}:~/CA3/model_bard_2.png"

CURRENT_DIR=$(pwd)
LOCAL_DIR="${CURRENT_DIR}"


rsync -avzP ${REMOTE_DIR1} ${LOCAL_DIR}
