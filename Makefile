CC = gcc
CFLAGS = -g -Wall -Wno-unused-function -Wconversion -std=c99 -fopenmp -mavx -mfma -pthread
LDFLAGS = -fopenmp
CUNIT = -lcunit
PYTHON = -I/home/gyl/anaconda3/envs/numc/include/python3.6m -L/home/gyl/anaconda3/envs/numc/lib -lpython3.6m -Wl,-rpath=/home/gyl/anaconda3/envs/numc/lib

install:
	if [ ! -f files.txt ]; then touch files.txt; fi
	rm -rf build
	xargs rm -rf < files.txt
	python3 setup.py install --record files.txt

uninstall:
	if [ ! -f files.txt ]; then touch files.txt; fi
	rm -rf build
	xargs rm -rf < files.txt

clean:
	rm -f *.o
	rm -f test
	rm -rf build
	rm -rf __pycache__

test:
	rm -f test
	$(CC) $(CFLAGS) tests/mat_test.c src/matrix.c -o test $(LDFLAGS) $(CUNIT) $(PYTHON)
	./test

.PHONY: test
