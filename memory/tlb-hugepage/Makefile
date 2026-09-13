CC     ?= gcc
CFLAGS := -O2 -Wall -Wextra -g

all: tlb_bench

tlb_bench: tlb_bench.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f tlb_bench

.PHONY: all clean
