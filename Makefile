CC = g++

CFLAGS = -std=c++1y -pipe -O2 -fPIC \
		 -I/usr/local/lib \
		 -I/usr/local/include/essentia/ \
		 -I/usr/local/include/essentia/scheduler/ \
		 -I/usr/local/include/essentia/streaming/  \
		 -I/usr/local/include/essentia/utils \
		 -I/usr/include/taglib \
		 -D__STDC_CONSTANT_MACROS \
		 -I./src/include/
		 #-I/usr/include/qt4 \
		 #-I/usr/include/qt4/QtCore
LFLAGS = -lessentia -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lsamplerate -lpng \
		 -ltag -lfftw3f -lavresample #-lQtCore


OBJDIR = build

SRC = $(shell find . -name "*.cpp")
OBJ = $(addprefix $(OBJDIR)/,$(notdir $(SRC:.cpp=.o)))

EXEC = s2pam_train

all: $(OBJDIR) $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) $(LFLAGS) -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

build/train.o: src/training/train.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/render.o: src/utils/render.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/helper.o: src/utils/helper.cpp
	$(CC) $< -c $(CFLAGS) -o $@

clean:
	rm $(OBJ)
	rmdir $(OBJDIR)
	rm $(EXEC)
