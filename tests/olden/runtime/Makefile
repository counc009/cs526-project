.KEEP_STATE:

OLDENDIR = /u/ac/carlisle/olden_v1.01
RCC = $(OLDENDIR)/mcc/sparc/backend/mcc
TARGET		= target
SRCS		= cache.c cache_asm.asm cache_handlers.c call-stub.asm debug.c future.asm init.c mem.c migrate.c migrate_asm.asm msg.c msg2.c rpc.c rpc_handlers.c stack.asm mccvis.asm vis.o
OBJ1		= $(SRCS:%.c=%.o) 
OBJS		= $(OBJ1:%.asm=%.o) 
PLAINOBJS = plain.o plain-mem.o plain2.o
INCDIRS		= /usr/include . $(OLDENDIR)/mcc/sparc/backend
NODELIBS	= 
LIBDIRS		= 

CC		= cc
#CPPFLAGS = -DVISUALIZE
CPPFLAGS =

CFLAGS		= 

$(TARGET):	$(OBJS) $(PLAINOBJS)
	ar rcv libolden.a $(OBJS) 
	ranlib libolden.a
	date > target

debug.s:	debug.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) debug.c | $(RCC) > debug.s
mem.s:	mem.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) mem.c | $(RCC) > mem.s
cache.s:	cache.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) cache.c | $(RCC) > cache.s
#  init.c, migrate.c, msg.c, rpc.c all compiled using defaults

msg2.o: msg2.c
	cc $(CPPFLAGS) $(INCDIRS:%=-I%) -c msg2.c 
cache_handlers.o: cache_handlers.c constants.h cache.h
	cc $(CPPFLAGS) $(INCDIRS:%=-I%) -c cache_handlers.c 
rpc_handlers.o: rpc_handlers.c constants.h msg.h rpc.h future-cell.h
	cc $(CPPFLAGS) $(INCDIRS:%=-I%) -c rpc_handlers.c 

plain.s:	plain.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) plain.c | $(RCC) > plain.s
plain-mem.s:	plain-mem.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) plain-mem.c | $(RCC) > plain-mem.s

stack.s: registers.h constants.h 
call-stub.s: constants.h registers.h call-stub.h migrate.h rpc.h future-cell.h
migrate.s: constants.h migrate.h call-stub.h cache.h msg.h
rpc.s: constants.h future-cell.h msg.h rpc.h 
msg.s: migrate.h constants.h msg.h call-stub.h future-cell.h rpc.h threads.h cache.h
migrate_asm.s: constants.h registers.h stack_frame.h migrate.h
init.s: constants.h msg.h
cache.s: constants.h cache.h

%.s:    %.c
	/lib/cpp $(CPPFLAGS) $(INCDIRS:%=-I%) $< | $(RCC) -m > $*.s
%.s:	%.asm
	/lib/cpp -D_ASM_ $(CPPFLAGS) $(INCDIRS:%=-I%) $< > $*.s
%.o:	%.s
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS) *~ .make.state .nse_depinfo

