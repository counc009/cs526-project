CLANG = clang
OPT = opt

NUM_THREADS = 8

SOURCES = $(wildcard array/*.c linked_list/*.c)
OPTS = $(addsuffix .opt.ll, $(basename $(SOURCES)))
EXECS = $(addsuffix .out, $(basename $(SOURCES)))

OPT_FLAGS = -enable-new-pm=0 -load ../build/pass/libPS-DSWP.so -tbaa -psdswp -debug

all: exec
exec: $(EXECS)
opt: $(OPTS)

%.ll: %.c
	$(CLANG) -S -emit-llvm -O1 $^ -o $@

array/%.opt.ll: array/%.ll
	$(OPT) $(OPT_FLAGS) -num-threads=$(NUM_THREADS) $^ -S -o $@

array/%.out: array/%.opt.ll test_array.ll
	$(CLANG) -O1 $^ -o $@

linked_list/%.opt.ll: linked_list/%.ll
	$(OPT) $(OPT_FLAGS) -num-threads=$(NUM_THREADS) -asll=node $^ -S -o $@

linked_list/%.out: linked_list/%.opt.ll test_list.ll
	$(CLANG) -O1 $^ -o $@

clean:
	rm -rf *.ll *.out */*.ll */*.out

.PHONY: clean
.PRECIOUS: %.ll
