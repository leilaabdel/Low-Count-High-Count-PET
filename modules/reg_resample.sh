#!/bin/bash

source /autofs/space/celer_001/users/software_2/build_sirf/INSTALL/bin/env_sirf.sh ; reg_resample -ref $1 -flo $2 -res $3 -inter 0
