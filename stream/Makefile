CC = gcc
CFLAGS = -O2 -Ofast -fargument-noalias -march=native -ffreestanding -fopenmp

all: stream

stream.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream

clean:
	rm -f stream *.o
