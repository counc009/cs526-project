.global _mccvisFstackPush
.align 4
.proc 4

_mccvisFstackPush:
    save %sp,-72,%sp
    call _visFstackPush; nop
    ret; restore

.global _mccvisFstackPop
.align 4
.proc 4

_mccvisFstackPop:
    save %sp,-72,%sp
    call _visFstackPop; nop
    ret; restore

.global _mccvisProcTouch
.align 4
.proc 4

_mccvisProcTouch:
    save %sp,-72,%sp
    call _visProcTouch; nop
    ret; restore

.global _mccvisStackAlloc
.align 4
.proc 4

_mccvisStackAlloc:
    save %sp,-72,%sp
    mov %g6,%o0
    call _visStackAlloc; mov %g4,%o1
    ret; restore

.global _mccvisStackDeactivate
.align 4
.proc 4

_mccvisStackDeactivate:
    save %sp,-72,%sp
    call _visStackDeactivate; mov %g6,%o0
    ret; restore
    
.global _mccvisStackDeallocate
.align 4
.proc 4

_mccvisStackDeallocate:
    save %sp,-72,%sp
    call _visStackAlloc; mov %g6,%o0
    ret; restore

