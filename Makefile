CC = g++

CFLAGS = -std=c++17 -pipe -O2 -g -fPIC \
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
		 -ltag -lfftw3f -lavresample -lstdc++fs -lpthread#-lQtCore

OBJDIR = build

SRC = $(shell find . -name "*.cpp")
OBJ = $(addprefix $(OBJDIR)/,$(notdir $(SRC:.cpp=.o)))

EXEC_FEATURE = s2pam_featureExtraction
EXEC_TRAIN = s2pam_trainNN

all: $(OBJDIR) featureExtraction trainNN

featureExtraction: $(EXEC_FEATURE)

trainNN: $(EXEC_TRAIN)

$(EXEC_FEATURE): build/featureExtraction.o build/render.o build/helper.o build/wave_read.o
	$(CC) build/featureExtraction.o build/render.o build/helper.o build/wave_read.o $(LFLAGS) -o $@

$(EXEC_TRAIN): build/trainNN.o
	$(CC) build/trainNN.o $(LFLAGS) -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

build/featureExtraction.o: src/main/featureExtraction.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/trainNN.o: src/main/trainNN.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/render.o: src/utils/render.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/helper.o: src/utils/helper.cpp
	$(CC) $< -c $(CFLAGS) -o $@

build/wave_read.o: src/utils/wave_read.cpp
	$(CC) $< -c $(CFLAGS) -o $@

clean:
	rm $(OBJ)
	rmdir $(OBJDIR)
	rm $(EXEC_FEATURE)
	rm $(EXEC_TRAIN)
