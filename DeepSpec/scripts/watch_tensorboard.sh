

tensorboard_path=${1:-}
PORT=${2:-6006}


echo "ssh -L $PORT:localhost:$PORT h20"
echo "http://localhost:$PORT"

tensorboard --logdir \
    $tensorboard_path \
    --port $PORT \
    --bind_all
