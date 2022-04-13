CLANG = clang

SOURCES = $(wildcard array/*.c linked_list/*.c)
EXECS = $(addsuffix .out, $(basename $(SOURCES)))

all: $(EXECS)

%.ll: %.c
	$(CLANG) -S -emit-llvm -O1 $^ -o $@

array/%.out: array/%.ll test_array.ll
	$(CLANG) -O1 $^ -o $@

linked_list/%.out: linked_list/%.ll test_list.ll
	$(CLANG) -O1 $^ -o $@

clean:
	rm -rf *.ll *.out */*.ll */*.out

.PHONY: clean
