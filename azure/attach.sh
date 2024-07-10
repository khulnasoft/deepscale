#!/bin/bash

name=${1-deepscale}
docker exec -i -w /home/deepscale -t $name /bin/bash
