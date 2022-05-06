CLANG = clang
OPT = opt

NUM_THREADS = 8

SOURCES = $(wildcard array/*.c linked_list/*.c tree/*.c nested_array/*.c nested_list/*.c)
OPTS = $(addsuffix .opt.ll, $(basename $(SOURCES)))
PARS = $(addsuffix .par.ll, $(basename $(SOURCES)))
EXECS = $(addsuffix .exec, $(basename $(SOURCES)))
REFS = $(addsuffix .ref, $(basename $(SOURCES)))

PRE_PSDSWP_FLAGS = -sccp -simplifycfg -loop-simplify
PSDSWP_FLAGS = -enable-new-pm=0 -load ../build/pass/libPS-DSWP.so -tbaa -psdswp -debug

all: exec refs
exec: $(EXECS)
refs: $(REFS)
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

array/%.exec: array/%.par.ll test_array.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.opt.ll: linked_list/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

linked_list/%.par.ll: linked_list/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) -asll=node $^ -S -o $@

linked_list/%.ref: linked_list/%.opt.ll test_list.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.exec: linked_list/%.par.ll test_list.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

tree/%.opt.ll: tree/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

tree/%.par.ll: tree/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) $^ -S -o $@

tree/%.ref: tree/%.opt.ll test_tree.ll
	$(CLANG) -O3 $^ -o $@

tree/%.exec: tree/%.par.ll test_tree.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

nested_array/%.opt.ll: nested_array/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

nested_array/%.par.ll: nested_array/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) $^ -S -o $@

nested_array/%.ref: nested_array/%.opt.ll test_nested_array.ll
	$(CLANG) -O3 $^ -o $@

nested_array/%.exec: nested_array/%.par.ll test_nested_array.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

nested_list/%.opt.ll: nested_list/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

nested_list/%.par.ll: nested_list/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) -asll=node -asll=lnode $^ -S -o $@

nested_list/%.ref: nested_list/%.opt.ll test_nested_list.ll
	$(CLANG) -O3 $^ -o $@

nested_list/%.exec: nested_list/%.par.ll test_nested_list.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

clean:
	rm -rf *.ll *.exec *.ref */*.ll */*.exec */*.ref

.PHONY: clean all
.PRECIOUS: %.ll
