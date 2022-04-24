CLANG = clang
OPT = opt

NUM_THREADS = 8

SOURCES = $(wildcard array/*.c linked_list/*.c)
OPTS = $(addsuffix .opt.ll, $(basename $(SOURCES)))
PARS = $(addsuffix .par.ll, $(basename $(SOURCES)))
EXECS = $(addsuffix .out, $(basename $(SOURCES)))

PRE_PSDSWP_FLAGS = -sccp -simplifycfg -loop-simplify
PSDSWP_FLAGS = -enable-new-pm=0 -load ../build/pass/libPS-DSWP.so -tbaa -psdswp -debug

all: exec
exec: $(EXECS)
opt: $(OPTS)
par: $(PARS)

%.ll: %.c
	$(CLANG) -S -emit-llvm -O1 $^ -o $@

array/%.opt.ll: array/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

array/%.par.ll: array/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) $^ -S -o $@

array/%.ref: array/%.opt.ll test_array.ll
	$(CLANG) -O3 $^ -o $@

array/%.exec: array/%.par.ll test_array.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.opt.ll: linked_list/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

linked_list/%.par.ll: linked_list/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) -asll=node $^ -S -o $@

linked_list/%.ref: linked_list/%.opt.ll test_list.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.exec: linked_list/%.par.ll test_list.ll
	$(CLANG) -O3 $^ -o $@

clean:
	rm -rf *.ll *.out */*.ll */*.out

.PHONY: clean
.PRECIOUS: %.ll
