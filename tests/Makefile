CLANG = clang
OPT = opt

NUM_THREADS = 8

SOURCES = $(wildcard array/*.c linked_list/*.c)
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

array/%.ref: array/%.opt.ll test_array.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

array/%.exec: array/%.par.ll test_array.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.opt.ll: linked_list/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

linked_list/%.par.ll: linked_list/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) -asll=node $^ -S -o $@

linked_list/%.ref: linked_list/%.opt.ll test_list.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

linked_list/%.exec: linked_list/%.par.ll test_list.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

tree/%.opt.ll: tree/%.ll
	$(OPT) $(PRE_PSDSWP_FLAGS) $^ -S -o $@

tree/%.par.ll: tree/%.opt.ll
	$(OPT) $(PSDSWP_FLAGS) -num-threads=$(NUM_THREADS) $^ -S -o $@

tree/%.ref: tree/%.opt.ll test_tree.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

tree/%.exec: tree/%.par.ll test_tree.ll psdswp.ll
	$(CLANG) -O3 $^ -o $@

clean:
	rm -rf *.ll *.exec *.ref */*.ll */*.exec */*.ref

# Build the executables that are currently working
working: array/increment.ref array/increment.exec array/incdec.ref array/incdec.exec array/incdecnext.ref array/incdecnext.exec array/incdecprev.ref array/incdecprev.exec array/squarenons.ref array/squarenons.exec array/runningsum.ref array/runningsum.exec array/sum.ref array/sum.exec linked_list/increment.ref linked_list/increment.exec linked_list/incdec.ref linked_list/incdec.exec linked_list/incdecnext.ref linked_list/incdecnext.exec linked_list/incdecprev.ref linked_list/incdecprev.exec linked_list/squarenons.ref linked_list/squarenons.exec linked_list/runningsum.ref linked_list/runningsum.exec linked_list/sum.ref linked_list/sum.exec

trees: tree/increment.ref tree/increment.exec tree/incdec.ref tree/incdec.exec tree/squarenons.ref tree/squarenons.exec

.PHONY: clean working trees
.PRECIOUS: %.ll
