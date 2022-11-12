docker container run --rm -it \
      -p 8888:8888 \
      -u 501:20 \
      -v "$(pwd)/notebooks:/notebooks" \
      mlaljup jupyter lab --ip=0.0.0.0

#      -u $(id -u ${USER}):$(id -g ${USER}) \

