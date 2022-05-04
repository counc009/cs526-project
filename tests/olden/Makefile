CLANG = clang
OPT = opt

NUM_THREADS = 8


PRE_PSDSWP_FLAGS = -sccp -simplifycfg -loop-simplify
PSDSWP_FLAGS = -enable-new-pm=0 -load ../build/pass/libPS-DSWP.so -tbaa -psdswp -debug


clean:
	rm -rf *.ll *.exec *.ref */*.ll */*.exec */*.ref


.PHONY: clean 

