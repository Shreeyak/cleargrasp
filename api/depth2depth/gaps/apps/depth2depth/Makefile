#
# Application name 
#

NAME=depth2depth



#
# With Ceres
# 

#USER_LIBS=-L/usr/local/lib -lceres -lccolamd -lcamd -lcolamd -lamd -Wl,-Bdynamic -lcxsparse -lccolamd -lcamd -lcolamd -lcholmod -lamd -llapack -lopenblas -Wl,-Bstatic -lcxsparse -Wl,-Bdynamic -lgomp -lglog -lgflags -lgfortran -lhdf5 -ldl -lm

#USER_LIBS=-L/usr/local/lib -lceres -lgflags -lglog -lcamd -lcholmod -lccolamd -lcolamd -lcamd -lamd -lmetis -lcxsparse -lsuitesparseconfig -fopenmp -lpthread -lgomp -llapack -lopenblas -lgfortran -lhdf5 -ldl -lm



#
# Without Ceres
#

#USER_LIBS=-lhdf5

#USER_LIBS = -L/usr/lib/x86_64-linux-gnu/ -lhdf5_serial
#USER_CFLAGS=-DRN_USE_CSPARSE /usr/lib/x86_64-linux-gnu/

USER_LIBS=-L/usr/include/hdf5/serial/ -lhdf5_serial
USER_CFLAGS=-DRN_USE_CSPARSE "/usr/include/hdf5/serial/"


#
# Source files
#

CCSRCS=$(NAME).cpp 



#
# Libraries
#

PKG_LIBS=-lR2Shapes -lRNMath -lRNBasics -ljpeg -lpng -lCSparse



#
# Include standard makefile
#

include ../../makefiles/Makefile.apps















