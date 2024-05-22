CC = mpic++
CFLAGS = -O3 -Wall -I/usr/include
LIBS =
DEPS =
OBJ = apply-filter.o papply-filter.o

all: apply-filter papply-filter

apply-filter: apply-filter.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

papply-filter: papply-filter.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

serialsharpen: apply-filter
	./apply-filter oimage.txt sharpen_kernel.txt

serialedge: apply-filter
	./apply-filter oimage.txt edge_kernel.txt
	
serialblur: apply-filter
	./apply-filter oimage.txt blur_kernel.txt
	
parallelsharpen: papply-filter
	./papply-filter 4 sharpen sharpen_kernel.txt

paralleledge: papply-filter
	./papply-filter 4 edge edge_kernel.txt
	
parallelblur: papply-filter
	./papply-filter 4 blur blur_kernel.txt
	
clean:
	rm -f *.o *~ core apply-filter papply-filter

