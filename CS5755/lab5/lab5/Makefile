CC = gcc
CFLAGS = -lm -g -pg 
EXTRAFLAGS = -mavx2 -mfma -O3
LDFLAGS = -fprofile-arcs -ftest-coverage

.PHONY: gpt
gpt: gpt2.c
	$(CC) -o $@ gpt2.c $(CFLAGS) $(LDFLAGS) $(EXTRAFLAGS)


.PHONY: cachegrind
cachegrind:
	valgrind --tool=cachegrind ./gpt
	cg_annotate cachegrind.out.* > tempcachegrind.txt


.PHONY: gprof
gprof:
	./gpt
	gprof gpt gmon.out > tempgprof.txt

.PHONY: gcov
gcov:
	./gpt
	mv gpt-gpt2.gcda gpt2.gcda
	mv gpt-gpt2.gcno gpt2.gcno
	gcov gpt2.c

.PHONY: clean
clean:
	rm -f *.gcov *.gcda *.gcno gmon.out cachegrind.out.*