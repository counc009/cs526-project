.seg "text"
.global ___prof
.align 4
.proc 4
___prof:
   /* have 32 for storing oregs, 88 for f.e. for calling prof2 */
   /* make sure prof2 uses register windows */
   add %sp,-120,%sp
   st %o0,[%sp+88]
   st %o1,[%sp+92]
   st %o2,[%sp+96]
   st %o3,[%sp+100]
   st %o4,[%sp+104]
   st %o5,[%sp+108]
   st %o6,[%sp+112]
   st %o7,[%sp+116]
   mov %g3,%o0
   mov %g4,%o1
   mov %g5,%o2
   call ___prof2; mov %g6,%o3
   ld [%sp+88],%o0
   ld [%sp+92],%o1
   ld [%sp+96],%o2
   ld [%sp+100],%o3
   ld [%sp+104],%o4
   ld [%sp+108],%o5
   ld [%sp+112],%o6
   ld [%sp+116],%o7
   jmpl %o7+8,%g0; add %sp,120,%sp
   
