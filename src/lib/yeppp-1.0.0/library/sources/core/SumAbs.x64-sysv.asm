;                       Yeppp! library implementation
;                   This file is auto-generated by Peach-Py,
;        Portable Efficient Assembly Code-generator in Higher-level Python,
;                  part of the Yeppp! library infrastructure
; This file is part of Yeppp! library and licensed under the New BSD license.
; See LICENSE.txt for the full text of the license.

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.Nehalem progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V32f_S32f_Nehalem_constants:
	.c0: DD 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.Nehalem progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V32f_S32f_Nehalem
_yepCore_SumAbs_V32f_S32f_Nehalem:
%else
section .text
global __yepCore_SumAbs_V32f_S32f_Nehalem
__yepCore_SumAbs_V32f_S32f_Nehalem:
%endif
	.ENTRY:
	MOVDQA xmm13,  [rel _yepCore_SumAbs_V32f_S32f_Nehalem_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 3
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 3
	JNZ .return_misaligned_pointer
	XORPS xmm14, xmm14
	TEST rdx, rdx
	JZ .return_ok
	XORPS xmm15, xmm15
	XORPS xmm7, xmm7
	XORPS xmm6, xmm6
	XORPS xmm5, xmm5
	XORPS xmm4, xmm4
	XORPS xmm3, xmm3
	TEST rdi, 15
	JZ .source_16b_aligned
	.source_16b_misaligned:
	MOVSS xmm2, [rdi]
	ANDPS xmm2, xmm13
	ADDPS xmm14, xmm2
	ADD rdi, 4
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 15
	JNZ .source_16b_misaligned
	.source_16b_aligned:
	SUB rdx, 28
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPS xmm2, [rdi]
	MOVUPS xmm8, [byte rdi + 16]
	MOVUPS xmm9, [byte rdi + 32]
	MOVUPS xmm1, [byte rdi + 48]
	ANDPS xmm2, xmm13
	MOVUPS xmm11, [byte rdi + 64]
	ANDPS xmm8, xmm13
	ADDPS xmm14, xmm2
	MOVUPS xmm12, [byte rdi + 80]
	ANDPS xmm9, xmm13
	ADDPS xmm15, xmm8
	MOVUPS xmm10, [byte rdi + 96]
	ANDPS xmm1, xmm13
	ADDPS xmm7, xmm9
	ADD rdi, 112
	ANDPS xmm11, xmm13
	ADDPS xmm6, xmm1
	SUB rdx, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPS xmm2, [rdi]
	ANDPS xmm12, xmm13
	ADDPS xmm5, xmm11
	MOVUPS xmm8, [byte rdi + 16]
	ANDPS xmm10, xmm13
	ADDPS xmm4, xmm12
	MOVUPS xmm9, [byte rdi + 32]
	ADDPS xmm3, xmm10
	MOVUPS xmm1, [byte rdi + 48]
	ANDPS xmm2, xmm13
	MOVUPS xmm11, [byte rdi + 64]
	ANDPS xmm8, xmm13
	ADDPS xmm14, xmm2
	MOVUPS xmm12, [byte rdi + 80]
	ANDPS xmm9, xmm13
	ADDPS xmm15, xmm8
	MOVUPS xmm10, [byte rdi + 96]
	ANDPS xmm1, xmm13
	ADDPS xmm7, xmm9
	ADD rdi, 112
	ANDPS xmm11, xmm13
	ADDPS xmm6, xmm1
	SUB rdx, 28
	JAE .process_batch
	.process_batch_epilogue:
	ANDPS xmm12, xmm13
	ADDPS xmm5, xmm11
	ANDPS xmm10, xmm13
	ADDPS xmm4, xmm12
	ADDPS xmm3, xmm10
	.batch_process_finish:
	ADD rdx, 28
	JZ .reduce_batch
	.process_single:
	MOVSS xmm8, [rdi]
	ANDPS xmm8, xmm13
	ADDPS xmm14, xmm8
	ADD rdi, 4
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	ADDPS xmm14, xmm15
	ADDPS xmm7, xmm6
	ADDPS xmm5, xmm4
	ADDPS xmm14, xmm7
	ADDPS xmm5, xmm3
	ADDPS xmm14, xmm5
	MOVHLPS xmm8, xmm14
	ADDPS xmm14, xmm8
	MOVSHDUP xmm8, xmm14
	ADDSS xmm14, xmm8
	.return_ok:
	MOVSS [rsi], xmm14
	XOR eax, eax
	.return:
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.SandyBridge progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V32f_S32f_SandyBridge_constants:
	.c0: DD 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.SandyBridge progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V32f_S32f_SandyBridge
_yepCore_SumAbs_V32f_S32f_SandyBridge:
%else
section .text
global __yepCore_SumAbs_V32f_S32f_SandyBridge
__yepCore_SumAbs_V32f_S32f_SandyBridge:
%endif
	.ENTRY:
	VMOVDQA ymm14,  [rel _yepCore_SumAbs_V32f_S32f_SandyBridge_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 3
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 3
	JNZ .return_misaligned_pointer
	VXORPS xmm15, xmm15, xmm15
	TEST rdx, rdx
	JZ .return_ok
	VXORPS xmm7, xmm7, xmm7
	VXORPS xmm6, xmm6, xmm6
	VXORPS xmm5, xmm5, xmm5
	VXORPS xmm4, xmm4, xmm4
	VXORPS xmm3, xmm3, xmm3
	VXORPS xmm2, xmm2, xmm2
	TEST rdi, 31
	JZ .source_32b_aligned
	.source_32b_misaligned:
	VMOVSS xmm1, [rdi]
	VANDPS xmm1, xmm1, xmm14
	VADDPS ymm15, ymm15, ymm1
	ADD rdi, 4
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 31
	JNZ .source_32b_misaligned
	.source_32b_aligned:
	SUB rdx, 56
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPS ymm1, [rdi]
	VMOVUPS ymm8, [byte rdi + 32]
	VMOVUPS ymm9, [byte rdi + 64]
	VMOVUPS ymm11, [byte rdi + 96]
	VMOVUPS ymm13, [dword rdi + 128]
	VANDPS ymm1, ymm1, ymm14
	VMOVUPS ymm12, [dword rdi + 160]
	VANDPS ymm8, ymm8, ymm14
	VADDPS ymm15, ymm15, ymm1
	VMOVUPS ymm10, [dword rdi + 192]
	VANDPS ymm9, ymm9, ymm14
	VADDPS ymm7, ymm7, ymm8
	ADD rdi, 224
	VANDPS ymm11, ymm11, ymm14
	VADDPS ymm6, ymm6, ymm9
	SUB rdx, 56
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPS ymm1, [rdi]
	VANDPS ymm13, ymm13, ymm14
	VADDPS ymm5, ymm5, ymm11
	VMOVUPS ymm8, [byte rdi + 32]
	VANDPS ymm12, ymm12, ymm14
	VADDPS ymm4, ymm4, ymm13
	VMOVUPS ymm9, [byte rdi + 64]
	VANDPS ymm10, ymm10, ymm14
	VADDPS ymm3, ymm3, ymm12
	VMOVUPS ymm11, [byte rdi + 96]
	VADDPS ymm2, ymm2, ymm10
	VMOVUPS ymm13, [dword rdi + 128]
	VANDPS ymm1, ymm1, ymm14
	VMOVUPS ymm12, [dword rdi + 160]
	VANDPS ymm8, ymm8, ymm14
	VADDPS ymm15, ymm15, ymm1
	VMOVUPS ymm10, [dword rdi + 192]
	VANDPS ymm9, ymm9, ymm14
	VADDPS ymm7, ymm7, ymm8
	ADD rdi, 224
	VANDPS ymm11, ymm11, ymm14
	VADDPS ymm6, ymm6, ymm9
	SUB rdx, 56
	JAE .process_batch
	.process_batch_epilogue:
	VANDPS ymm13, ymm13, ymm14
	VADDPS ymm5, ymm5, ymm11
	VANDPS ymm12, ymm12, ymm14
	VADDPS ymm4, ymm4, ymm13
	VANDPS ymm10, ymm10, ymm14
	VADDPS ymm3, ymm3, ymm12
	VADDPS ymm2, ymm2, ymm10
	.batch_process_finish:
	ADD rdx, 56
	JZ .reduce_batch
	.process_single:
	VMOVSS xmm8, [rdi]
	VANDPS xmm8, xmm8, xmm14
	VADDPS ymm15, ymm15, ymm8
	ADD rdi, 4
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	VADDPS ymm15, ymm15, ymm7
	VADDPS ymm6, ymm6, ymm5
	VADDPS ymm4, ymm4, ymm3
	VADDPS ymm15, ymm15, ymm6
	VADDPS ymm4, ymm4, ymm2
	VADDPS ymm15, ymm15, ymm4
	VEXTRACTF128 xmm8, ymm15, 1
	VADDPS xmm15, xmm15, xmm8
	VUNPCKHPD xmm8, xmm15, xmm15
	VADDPS xmm15, xmm15, xmm8
	VMOVSHDUP xmm8, xmm15
	VADDSS xmm15, xmm15, xmm8
	.return_ok:
	VMOVSS [rsi], xmm15
	XOR eax, eax
	.return:
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.Bulldozer progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V32f_S32f_Bulldozer_constants:
	.c0: DD 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.Bulldozer progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V32f_S32f_Bulldozer
_yepCore_SumAbs_V32f_S32f_Bulldozer:
%else
section .text
global __yepCore_SumAbs_V32f_S32f_Bulldozer
__yepCore_SumAbs_V32f_S32f_Bulldozer:
%endif
	.ENTRY:
	VMOVDQA ymm13,  [rel _yepCore_SumAbs_V32f_S32f_Bulldozer_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 3
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 3
	JNZ .return_misaligned_pointer
	VXORPS xmm14, xmm14, xmm14
	TEST rdx, rdx
	JZ .return_ok
	VXORPS xmm15, xmm15, xmm15
	VXORPS xmm7, xmm7, xmm7
	VXORPS xmm6, xmm6, xmm6
	VXORPS xmm5, xmm5, xmm5
	VXORPS xmm4, xmm4, xmm4
	TEST rdi, 31
	JZ .source_32b_aligned
	.source_32b_misaligned:
	VMOVSS xmm3, [rdi]
	VANDPS xmm3, xmm3, xmm13
	VADDPS xmm14, xmm14, xmm3
	ADD rdi, 4
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 31
	JNZ .source_32b_misaligned
	.source_32b_aligned:
	SUB rdx, 32
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPS xmm3, [rdi]
	VMOVUPS xmm8, [byte rdi + 16]
	VMOVUPS ymm9, [byte rdi + 32]
	VMOVUPS xmm11, [byte rdi + 64]
	VANDPS xmm3, xmm3, xmm13
	VMOVUPS xmm10, [byte rdi + 80]
	VANDPS xmm8, xmm8, xmm13
	VADDPS xmm14, xmm14, xmm3
	VMOVUPS ymm12, [byte rdi + 96]
	VANDPS ymm9, ymm9, ymm13
	VADDPS xmm15, xmm15, xmm8
	ADD rdi, 128
	VANDPS xmm11, xmm11, xmm13
	VADDPS ymm7, ymm7, ymm9
	SUB rdx, 32
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPS xmm3, [rdi]
	VANDPS xmm10, xmm10, xmm13
	VADDPS xmm6, xmm6, xmm11
	VMOVUPS xmm8, [byte rdi + 16]
	VANDPS ymm12, ymm12, ymm13
	VADDPS xmm5, xmm5, xmm10
	VMOVUPS ymm9, [byte rdi + 32]
	VADDPS ymm4, ymm4, ymm12
	VMOVUPS xmm11, [byte rdi + 64]
	VANDPS xmm3, xmm3, xmm13
	VMOVUPS xmm10, [byte rdi + 80]
	VANDPS xmm8, xmm8, xmm13
	VADDPS xmm14, xmm14, xmm3
	VMOVUPS ymm12, [byte rdi + 96]
	VANDPS ymm9, ymm9, ymm13
	VADDPS xmm15, xmm15, xmm8
	ADD rdi, 128
	VANDPS xmm11, xmm11, xmm13
	VADDPS ymm7, ymm7, ymm9
	SUB rdx, 32
	JAE .process_batch
	.process_batch_epilogue:
	VANDPS xmm10, xmm10, xmm13
	VADDPS xmm6, xmm6, xmm11
	VANDPS ymm12, ymm12, ymm13
	VADDPS xmm5, xmm5, xmm10
	VADDPS ymm4, ymm4, ymm12
	.batch_process_finish:
	ADD rdx, 32
	JZ .reduce_batch
	.process_single:
	VMOVSS xmm8, [rdi]
	VANDPS xmm8, xmm8, xmm13
	VADDPS xmm14, xmm14, xmm8
	ADD rdi, 4
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	VADDPS xmm14, xmm14, xmm15
	VADDPS ymm7, ymm7, ymm6
	VADDPS ymm5, ymm5, ymm4
	VADDPS ymm14, ymm14, ymm7
	VADDPS ymm14, ymm14, ymm5
	VEXTRACTF128 xmm8, ymm14, 1
	VADDPS xmm14, xmm14, xmm8
	VUNPCKHPD xmm8, xmm14, xmm14
	VADDPS xmm14, xmm14, xmm8
	VMOVSHDUP xmm8, xmm14
	VADDSS xmm14, xmm14, xmm8
	.return_ok:
	VMOVSS [rsi], xmm14
	XOR eax, eax
	.return:
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.Nehalem progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V64f_S64f_Nehalem_constants:
	.c0: DQ 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.Nehalem progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V64f_S64f_Nehalem
_yepCore_SumAbs_V64f_S64f_Nehalem:
%else
section .text
global __yepCore_SumAbs_V64f_S64f_Nehalem
__yepCore_SumAbs_V64f_S64f_Nehalem:
%endif
	.ENTRY:
	MOVDQA xmm13,  [rel _yepCore_SumAbs_V64f_S64f_Nehalem_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 7
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 7
	JNZ .return_misaligned_pointer
	XORPD xmm14, xmm14
	TEST rdx, rdx
	JZ .return_ok
	XORPD xmm15, xmm15
	XORPD xmm7, xmm7
	XORPD xmm6, xmm6
	XORPD xmm5, xmm5
	XORPD xmm4, xmm4
	XORPD xmm3, xmm3
	TEST rdi, 15
	JZ .source_16b_aligned
	.source_16b_misaligned:
	MOVSD xmm2, [rdi]
	ANDPD xmm2, xmm13
	ADDPD xmm14, xmm2
	ADD rdi, 8
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 15
	JNZ .source_16b_misaligned
	.source_16b_aligned:
	SUB rdx, 14
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPD xmm2, [rdi]
	MOVUPD xmm8, [byte rdi + 16]
	MOVUPD xmm9, [byte rdi + 32]
	MOVUPD xmm1, [byte rdi + 48]
	ANDPD xmm2, xmm13
	MOVUPD xmm11, [byte rdi + 64]
	ANDPD xmm8, xmm13
	ADDPD xmm14, xmm2
	MOVUPD xmm12, [byte rdi + 80]
	ANDPD xmm9, xmm13
	ADDPD xmm15, xmm8
	MOVUPD xmm10, [byte rdi + 96]
	ANDPD xmm1, xmm13
	ADDPD xmm7, xmm9
	ADD rdi, 112
	ANDPD xmm11, xmm13
	ADDPD xmm6, xmm1
	SUB rdx, 14
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPD xmm2, [rdi]
	ANDPD xmm12, xmm13
	ADDPD xmm5, xmm11
	MOVUPD xmm8, [byte rdi + 16]
	ANDPD xmm10, xmm13
	ADDPD xmm4, xmm12
	MOVUPD xmm9, [byte rdi + 32]
	ADDPD xmm3, xmm10
	MOVUPD xmm1, [byte rdi + 48]
	ANDPD xmm2, xmm13
	MOVUPD xmm11, [byte rdi + 64]
	ANDPD xmm8, xmm13
	ADDPD xmm14, xmm2
	MOVUPD xmm12, [byte rdi + 80]
	ANDPD xmm9, xmm13
	ADDPD xmm15, xmm8
	MOVUPD xmm10, [byte rdi + 96]
	ANDPD xmm1, xmm13
	ADDPD xmm7, xmm9
	ADD rdi, 112
	ANDPD xmm11, xmm13
	ADDPD xmm6, xmm1
	SUB rdx, 14
	JAE .process_batch
	.process_batch_epilogue:
	ANDPD xmm12, xmm13
	ADDPD xmm5, xmm11
	ANDPD xmm10, xmm13
	ADDPD xmm4, xmm12
	ADDPD xmm3, xmm10
	.batch_process_finish:
	ADD rdx, 14
	JZ .reduce_batch
	.process_single:
	MOVSD xmm8, [rdi]
	ANDPD xmm8, xmm13
	ADDPD xmm14, xmm8
	ADD rdi, 8
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	ADDPD xmm14, xmm15
	ADDPD xmm7, xmm6
	ADDPD xmm5, xmm4
	ADDPD xmm14, xmm7
	ADDPD xmm5, xmm3
	ADDPD xmm14, xmm5
	MOVHLPS xmm8, xmm14
	ADDSD xmm14, xmm8
	.return_ok:
	MOVSD [rsi], xmm14
	XOR eax, eax
	.return:
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.SandyBridge progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V64f_S64f_SandyBridge_constants:
	.c0: DQ 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.SandyBridge progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V64f_S64f_SandyBridge
_yepCore_SumAbs_V64f_S64f_SandyBridge:
%else
section .text
global __yepCore_SumAbs_V64f_S64f_SandyBridge
__yepCore_SumAbs_V64f_S64f_SandyBridge:
%endif
	.ENTRY:
	VMOVDQA ymm14,  [rel _yepCore_SumAbs_V64f_S64f_SandyBridge_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 7
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 7
	JNZ .return_misaligned_pointer
	VXORPD xmm15, xmm15, xmm15
	TEST rdx, rdx
	JZ .return_ok
	VXORPD xmm7, xmm7, xmm7
	VXORPD xmm6, xmm6, xmm6
	VXORPD xmm5, xmm5, xmm5
	VXORPD xmm4, xmm4, xmm4
	VXORPD xmm3, xmm3, xmm3
	VXORPD xmm2, xmm2, xmm2
	TEST rdi, 31
	JZ .source_32b_aligned
	.source_32b_misaligned:
	VMOVSD xmm1, [rdi]
	VANDPD xmm1, xmm1, xmm14
	VADDPD ymm15, ymm15, ymm1
	ADD rdi, 8
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 31
	JNZ .source_32b_misaligned
	.source_32b_aligned:
	SUB rdx, 28
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPD ymm1, [rdi]
	VMOVUPD ymm8, [byte rdi + 32]
	VMOVUPD ymm9, [byte rdi + 64]
	VMOVUPD ymm11, [byte rdi + 96]
	VMOVUPD ymm13, [dword rdi + 128]
	VANDPD ymm1, ymm1, ymm14
	VMOVUPD ymm12, [dword rdi + 160]
	VANDPD ymm8, ymm8, ymm14
	VADDPD ymm15, ymm15, ymm1
	VMOVUPD ymm10, [dword rdi + 192]
	VANDPD ymm9, ymm9, ymm14
	VADDPD ymm7, ymm7, ymm8
	ADD rdi, 224
	VANDPD ymm11, ymm11, ymm14
	VADDPD ymm6, ymm6, ymm9
	SUB rdx, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPD ymm1, [rdi]
	VANDPD ymm13, ymm13, ymm14
	VADDPD ymm5, ymm5, ymm11
	VMOVUPD ymm8, [byte rdi + 32]
	VANDPD ymm12, ymm12, ymm14
	VADDPD ymm4, ymm4, ymm13
	VMOVUPD ymm9, [byte rdi + 64]
	VANDPD ymm10, ymm10, ymm14
	VADDPD ymm3, ymm3, ymm12
	VMOVUPD ymm11, [byte rdi + 96]
	VADDPD ymm2, ymm2, ymm10
	VMOVUPD ymm13, [dword rdi + 128]
	VANDPD ymm1, ymm1, ymm14
	VMOVUPD ymm12, [dword rdi + 160]
	VANDPD ymm8, ymm8, ymm14
	VADDPD ymm15, ymm15, ymm1
	VMOVUPD ymm10, [dword rdi + 192]
	VANDPD ymm9, ymm9, ymm14
	VADDPD ymm7, ymm7, ymm8
	ADD rdi, 224
	VANDPD ymm11, ymm11, ymm14
	VADDPD ymm6, ymm6, ymm9
	SUB rdx, 28
	JAE .process_batch
	.process_batch_epilogue:
	VANDPD ymm13, ymm13, ymm14
	VADDPD ymm5, ymm5, ymm11
	VANDPD ymm12, ymm12, ymm14
	VADDPD ymm4, ymm4, ymm13
	VANDPD ymm10, ymm10, ymm14
	VADDPD ymm3, ymm3, ymm12
	VADDPD ymm2, ymm2, ymm10
	.batch_process_finish:
	ADD rdx, 28
	JZ .reduce_batch
	.process_single:
	VMOVSD xmm8, [rdi]
	VANDPD xmm8, xmm8, xmm14
	VADDPD ymm15, ymm15, ymm8
	ADD rdi, 8
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	VADDPD ymm15, ymm15, ymm7
	VADDPD ymm6, ymm6, ymm5
	VADDPD ymm4, ymm4, ymm3
	VADDPD ymm15, ymm15, ymm6
	VADDPD ymm4, ymm4, ymm2
	VADDPD ymm15, ymm15, ymm4
	VEXTRACTF128 xmm8, ymm15, 1
	VADDPD xmm15, xmm15, xmm8
	VUNPCKHPD xmm8, xmm15, xmm15
	VADDSD xmm15, xmm15, xmm8
	.return_ok:
	VMOVSD [rsi], xmm15
	XOR eax, eax
	.return:
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

%ifidn __OUTPUT_FORMAT__, elf64
section .rodata.Bulldozer progbits alloc noexec nowrite align=32
%else
section .rodata align=32
%endif
_yepCore_SumAbs_V64f_S64f_Bulldozer_constants:
	.c0: DQ 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF


%ifidn __OUTPUT_FORMAT__, elf64
section .text.Bulldozer progbits alloc exec nowrite align=16
global _yepCore_SumAbs_V64f_S64f_Bulldozer
_yepCore_SumAbs_V64f_S64f_Bulldozer:
%else
section .text
global __yepCore_SumAbs_V64f_S64f_Bulldozer
__yepCore_SumAbs_V64f_S64f_Bulldozer:
%endif
	.ENTRY:
	VMOVDQA ymm13,  [rel _yepCore_SumAbs_V64f_S64f_Bulldozer_constants.c0]
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 7
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 7
	JNZ .return_misaligned_pointer
	VXORPD xmm14, xmm14, xmm14
	TEST rdx, rdx
	JZ .return_ok
	VXORPD xmm15, xmm15, xmm15
	VXORPD xmm7, xmm7, xmm7
	VXORPD xmm6, xmm6, xmm6
	VXORPD xmm5, xmm5, xmm5
	VXORPD xmm4, xmm4, xmm4
	TEST rdi, 31
	JZ .source_32b_aligned
	.source_32b_misaligned:
	VMOVSD xmm3, [rdi]
	VANDPD xmm3, xmm3, xmm13
	VADDPD xmm14, xmm14, xmm3
	ADD rdi, 8
	SUB rdx, 1
	JZ .reduce_batch
	TEST rdi, 31
	JNZ .source_32b_misaligned
	.source_32b_aligned:
	SUB rdx, 16
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPD xmm3, [rdi]
	VMOVUPD xmm8, [byte rdi + 16]
	VMOVUPD ymm9, [byte rdi + 32]
	VMOVUPD xmm11, [byte rdi + 64]
	VANDPD xmm3, xmm3, xmm13
	VMOVUPD xmm10, [byte rdi + 80]
	VANDPD xmm8, xmm8, xmm13
	VADDPD xmm14, xmm14, xmm3
	VMOVUPD ymm12, [byte rdi + 96]
	VANDPD ymm9, ymm9, ymm13
	VADDPD xmm15, xmm15, xmm8
	ADD rdi, 128
	VANDPD xmm11, xmm11, xmm13
	VADDPD ymm7, ymm7, ymm9
	SUB rdx, 16
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPD xmm3, [rdi]
	VANDPD xmm10, xmm10, xmm13
	VADDPD xmm6, xmm6, xmm11
	VMOVUPD xmm8, [byte rdi + 16]
	VANDPD ymm12, ymm12, ymm13
	VADDPD xmm5, xmm5, xmm10
	VMOVUPD ymm9, [byte rdi + 32]
	VADDPD ymm4, ymm4, ymm12
	VMOVUPD xmm11, [byte rdi + 64]
	VANDPD xmm3, xmm3, xmm13
	VMOVUPD xmm10, [byte rdi + 80]
	VANDPD xmm8, xmm8, xmm13
	VADDPD xmm14, xmm14, xmm3
	VMOVUPD ymm12, [byte rdi + 96]
	VANDPD ymm9, ymm9, ymm13
	VADDPD xmm15, xmm15, xmm8
	ADD rdi, 128
	VANDPD xmm11, xmm11, xmm13
	VADDPD ymm7, ymm7, ymm9
	SUB rdx, 16
	JAE .process_batch
	.process_batch_epilogue:
	VANDPD xmm10, xmm10, xmm13
	VADDPD xmm6, xmm6, xmm11
	VANDPD ymm12, ymm12, ymm13
	VADDPD xmm5, xmm5, xmm10
	VADDPD ymm4, ymm4, ymm12
	.batch_process_finish:
	ADD rdx, 16
	JZ .reduce_batch
	.process_single:
	VMOVSD xmm8, [rdi]
	VANDPD xmm8, xmm8, xmm13
	VADDPD xmm14, xmm14, xmm8
	ADD rdi, 8
	SUB rdx, 1
	JNZ .process_single
	.reduce_batch:
	VADDPD xmm14, xmm14, xmm15
	VADDPD ymm7, ymm7, ymm6
	VADDPD ymm5, ymm5, ymm4
	VADDPD ymm14, ymm14, ymm7
	VADDPD ymm14, ymm14, ymm5
	VEXTRACTF128 xmm8, ymm14, 1
	VADDPD xmm14, xmm14, xmm8
	VUNPCKHPD xmm8, xmm14, xmm14
	VADDSD xmm14, xmm14, xmm8
	.return_ok:
	VMOVSD [rsi], xmm14
	XOR eax, eax
	.return:
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return
