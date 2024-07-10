#!/bin/bash

sphinx-apidoc -f -o source ../../deepscale
make html
