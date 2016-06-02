;                       Yeppp! library implementation
;                   This file is auto-generated by Peach-Py,
;        Portable Efficient Assembly Code-generator in Higher-level Python,
;                  part of the Yeppp! library infrastructure
; This file is part of Yeppp! library and licensed under the New BSD license.
; See LICENSE.txt for the full text of the license.

%ifidn __OUTPUT_FORMAT__, elf64
section .text.Nehalem progbits alloc exec nowrite align=16
global _yepCore_Min_V32fV32f_V32f_Nehalem
_yepCore_Min_V32fV32f_V32f_Nehalem:
%else
section .text
global __yepCore_Min_V32fV32f_V32f_Nehalem
__yepCore_Min_V32fV32f_V32f_Nehalem:
%endif
	.ENTRY:
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 3
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 3
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 3
	JNZ .return_misaligned_pointer
	TEST rcx, rcx
	JZ .return_ok
	TEST rdx, 15
	JZ .source_z_16b_aligned
	.source_z_16b_misaligned:
	MOVSS xmm2, [rdi]
	ADD rdi, 4
	MOVSS xmm1, [rsi]
	ADD rsi, 4
	MINSS xmm2, xmm1
	MOVSS [rdx], xmm2
	ADD rdx, 4
	SUB rcx, 1
	JZ .return_ok
	TEST rdx, 15
	JNZ .source_z_16b_misaligned
	.source_z_16b_aligned:
	SUB rcx, 28
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPS xmm2, [rdi]
	MOVUPS xmm8, [byte rdi + 16]
	MOVUPS xmm4, [rsi]
	MOVUPS xmm9, [byte rdi + 32]
	MOVUPS xmm3, [byte rsi + 16]
	MOVUPS xmm12, [byte rdi + 48]
	MOVUPS xmm13, [byte rsi + 32]
	MOVUPS xmm14, [byte rdi + 64]
	MOVUPS xmm10, [byte rsi + 48]
	MOVUPS xmm11, [byte rdi + 80]
	MOVUPS xmm5, [byte rsi + 64]
	MINPS xmm2, xmm4
	MOVUPS xmm6, [byte rdi + 96]
	MOVUPS xmm7, [byte rsi + 80]
	MINPS xmm8, xmm3
	MOVAPS [rdx], xmm2
	ADD rdi, 112
	MOVUPS xmm15, [byte rsi + 96]
	MINPS xmm9, xmm13
	MOVAPS [byte rdx + 16], xmm8
	SUB rcx, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPS xmm2, [rdi]
	ADD rsi, 112
	MINPS xmm12, xmm10
	MOVAPS [byte rdx + 32], xmm9
	MOVUPS xmm8, [byte rdi + 16]
	MOVUPS xmm4, [rsi]
	MINPS xmm14, xmm5
	MOVAPS [byte rdx + 48], xmm12
	MOVUPS xmm9, [byte rdi + 32]
	MOVUPS xmm3, [byte rsi + 16]
	MINPS xmm11, xmm7
	MOVAPS [byte rdx + 64], xmm14
	MOVUPS xmm12, [byte rdi + 48]
	MOVUPS xmm13, [byte rsi + 32]
	MINPS xmm6, xmm15
	MOVAPS [byte rdx + 80], xmm11
	MOVUPS xmm14, [byte rdi + 64]
	MOVUPS xmm10, [byte rsi + 48]
	MOVAPS [byte rdx + 96], xmm6
	MOVUPS xmm11, [byte rdi + 80]
	MOVUPS xmm5, [byte rsi + 64]
	MINPS xmm2, xmm4
	ADD rdx, 112
	MOVUPS xmm6, [byte rdi + 96]
	MOVUPS xmm7, [byte rsi + 80]
	MINPS xmm8, xmm3
	MOVAPS [rdx], xmm2
	ADD rdi, 112
	MOVUPS xmm15, [byte rsi + 96]
	MINPS xmm9, xmm13
	MOVAPS [byte rdx + 16], xmm8
	SUB rcx, 28
	JAE .process_batch
	.process_batch_epilogue:
	ADD rsi, 112
	MINPS xmm12, xmm10
	MOVAPS [byte rdx + 32], xmm9
	MINPS xmm14, xmm5
	MOVAPS [byte rdx + 48], xmm12
	MINPS xmm11, xmm7
	MOVAPS [byte rdx + 64], xmm14
	MINPS xmm6, xmm15
	MOVAPS [byte rdx + 80], xmm11
	MOVAPS [byte rdx + 96], xmm6
	ADD rdx, 112
	.batch_process_finish:
	ADD rcx, 28
	JZ .return_ok
	.process_single:
	MOVSS xmm8, [rdi]
	ADD rdi, 4
	MOVSS xmm9, [rsi]
	ADD rsi, 4
	MINSS xmm8, xmm9
	MOVSS [rdx], xmm8
	ADD rdx, 4
	SUB rcx, 1
	JNZ .process_single
	.return_ok:
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
section .text.SandyBridge progbits alloc exec nowrite align=16
global _yepCore_Min_V32fV32f_V32f_SandyBridge
_yepCore_Min_V32fV32f_V32f_SandyBridge:
%else
section .text
global __yepCore_Min_V32fV32f_V32f_SandyBridge
__yepCore_Min_V32fV32f_V32f_SandyBridge:
%endif
	.ENTRY:
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 3
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 3
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 3
	JNZ .return_misaligned_pointer
	TEST rcx, rcx
	JZ .return_ok
	TEST rdx, 31
	JZ .source_z_32b_aligned
	.source_z_32b_misaligned:
	VMOVSS xmm2, [rdi]
	ADD rdi, 4
	VMOVSS xmm1, [rsi]
	ADD rsi, 4
	VMINSS xmm2, xmm2, xmm1
	VMOVSS [rdx], xmm2
	ADD rdx, 4
	SUB rcx, 1
	JZ .return_ok
	TEST rdx, 31
	JNZ .source_z_32b_misaligned
	.source_z_32b_aligned:
	SUB rcx, 56
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPS ymm2, [rdi]
	VMOVUPS ymm8, [byte rdi + 32]
	VMOVUPS ymm13, [rsi]
	VMOVUPS ymm9, [byte rdi + 64]
	VMOVUPS ymm7, [byte rsi + 32]
	VMOVUPS ymm3, [byte rdi + 96]
	VMOVUPS ymm6, [byte rsi + 64]
	VMOVUPS ymm14, [dword rdi + 128]
	VMOVUPS ymm5, [byte rsi + 96]
	VMOVUPS ymm12, [dword rdi + 160]
	VMOVUPS ymm4, [dword rsi + 128]
	VMINPS ymm2, ymm2, ymm13
	VMOVUPS ymm11, [dword rdi + 192]
	VMOVUPS ymm15, [dword rsi + 160]
	VMINPS ymm8, ymm8, ymm7
	VMOVAPS [rdx], ymm2
	ADD rdi, 224
	VMOVUPS ymm10, [dword rsi + 192]
	VMINPS ymm9, ymm9, ymm6
	VMOVAPS [byte rdx + 32], ymm8
	SUB rcx, 56
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPS ymm2, [rdi]
	ADD rsi, 224
	VMINPS ymm3, ymm3, ymm5
	VMOVAPS [byte rdx + 64], ymm9
	VMOVUPS ymm8, [byte rdi + 32]
	VMOVUPS ymm13, [rsi]
	VMINPS ymm14, ymm14, ymm4
	VMOVAPS [byte rdx + 96], ymm3
	VMOVUPS ymm9, [byte rdi + 64]
	VMOVUPS ymm7, [byte rsi + 32]
	VMINPS ymm12, ymm12, ymm15
	VMOVAPS [dword rdx + 128], ymm14
	VMOVUPS ymm3, [byte rdi + 96]
	VMOVUPS ymm6, [byte rsi + 64]
	VMINPS ymm11, ymm11, ymm10
	VMOVAPS [dword rdx + 160], ymm12
	VMOVUPS ymm14, [dword rdi + 128]
	VMOVUPS ymm5, [byte rsi + 96]
	VMOVAPS [dword rdx + 192], ymm11
	VMOVUPS ymm12, [dword rdi + 160]
	VMOVUPS ymm4, [dword rsi + 128]
	VMINPS ymm2, ymm2, ymm13
	ADD rdx, 224
	VMOVUPS ymm11, [dword rdi + 192]
	VMOVUPS ymm15, [dword rsi + 160]
	VMINPS ymm8, ymm8, ymm7
	VMOVAPS [rdx], ymm2
	ADD rdi, 224
	VMOVUPS ymm10, [dword rsi + 192]
	VMINPS ymm9, ymm9, ymm6
	VMOVAPS [byte rdx + 32], ymm8
	SUB rcx, 56
	JAE .process_batch
	.process_batch_epilogue:
	ADD rsi, 224
	VMINPS ymm3, ymm3, ymm5
	VMOVAPS [byte rdx + 64], ymm9
	VMINPS ymm14, ymm14, ymm4
	VMOVAPS [byte rdx + 96], ymm3
	VMINPS ymm12, ymm12, ymm15
	VMOVAPS [dword rdx + 128], ymm14
	VMINPS ymm11, ymm11, ymm10
	VMOVAPS [dword rdx + 160], ymm12
	VMOVAPS [dword rdx + 192], ymm11
	ADD rdx, 224
	.batch_process_finish:
	ADD rcx, 56
	JZ .return_ok
	.process_single:
	VMOVSS xmm8, [rdi]
	ADD rdi, 4
	VMOVSS xmm9, [rsi]
	ADD rsi, 4
	VMINSS xmm8, xmm8, xmm9
	VMOVSS [rdx], xmm8
	ADD rdx, 4
	SUB rcx, 1
	JNZ .process_single
	.return_ok:
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
section .text.Nehalem progbits alloc exec nowrite align=16
global _yepCore_Min_V64fV64f_V64f_Nehalem
_yepCore_Min_V64fV64f_V64f_Nehalem:
%else
section .text
global __yepCore_Min_V64fV64f_V64f_Nehalem
__yepCore_Min_V64fV64f_V64f_Nehalem:
%endif
	.ENTRY:
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 7
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 7
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 7
	JNZ .return_misaligned_pointer
	TEST rcx, rcx
	JZ .return_ok
	TEST rdx, 15
	JZ .source_z_16b_aligned
	.source_z_16b_misaligned:
	MOVSD xmm2, [rdi]
	ADD rdi, 8
	MOVSD xmm1, [rsi]
	ADD rsi, 8
	MINSD xmm2, xmm1
	MOVSD [rdx], xmm2
	ADD rdx, 8
	SUB rcx, 1
	JZ .return_ok
	TEST rdx, 15
	JNZ .source_z_16b_misaligned
	.source_z_16b_aligned:
	SUB rcx, 14
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPD xmm2, [rdi]
	MOVUPD xmm8, [byte rdi + 16]
	MOVUPD xmm4, [rsi]
	MOVUPD xmm9, [byte rdi + 32]
	MOVUPD xmm3, [byte rsi + 16]
	MOVUPD xmm12, [byte rdi + 48]
	MOVUPD xmm13, [byte rsi + 32]
	MOVUPD xmm14, [byte rdi + 64]
	MOVUPD xmm10, [byte rsi + 48]
	MOVUPD xmm11, [byte rdi + 80]
	MOVUPD xmm5, [byte rsi + 64]
	MINPD xmm2, xmm4
	MOVUPD xmm6, [byte rdi + 96]
	MOVUPD xmm7, [byte rsi + 80]
	MINPD xmm8, xmm3
	MOVAPD [rdx], xmm2
	ADD rdi, 112
	MOVUPD xmm15, [byte rsi + 96]
	MINPD xmm9, xmm13
	MOVAPD [byte rdx + 16], xmm8
	SUB rcx, 14
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPD xmm2, [rdi]
	ADD rsi, 112
	MINPD xmm12, xmm10
	MOVAPD [byte rdx + 32], xmm9
	MOVUPD xmm8, [byte rdi + 16]
	MOVUPD xmm4, [rsi]
	MINPD xmm14, xmm5
	MOVAPD [byte rdx + 48], xmm12
	MOVUPD xmm9, [byte rdi + 32]
	MOVUPD xmm3, [byte rsi + 16]
	MINPD xmm11, xmm7
	MOVAPD [byte rdx + 64], xmm14
	MOVUPD xmm12, [byte rdi + 48]
	MOVUPD xmm13, [byte rsi + 32]
	MINPD xmm6, xmm15
	MOVAPD [byte rdx + 80], xmm11
	MOVUPD xmm14, [byte rdi + 64]
	MOVUPD xmm10, [byte rsi + 48]
	MOVAPD [byte rdx + 96], xmm6
	MOVUPD xmm11, [byte rdi + 80]
	MOVUPD xmm5, [byte rsi + 64]
	MINPD xmm2, xmm4
	ADD rdx, 112
	MOVUPD xmm6, [byte rdi + 96]
	MOVUPD xmm7, [byte rsi + 80]
	MINPD xmm8, xmm3
	MOVAPD [rdx], xmm2
	ADD rdi, 112
	MOVUPD xmm15, [byte rsi + 96]
	MINPD xmm9, xmm13
	MOVAPD [byte rdx + 16], xmm8
	SUB rcx, 14
	JAE .process_batch
	.process_batch_epilogue:
	ADD rsi, 112
	MINPD xmm12, xmm10
	MOVAPD [byte rdx + 32], xmm9
	MINPD xmm14, xmm5
	MOVAPD [byte rdx + 48], xmm12
	MINPD xmm11, xmm7
	MOVAPD [byte rdx + 64], xmm14
	MINPD xmm6, xmm15
	MOVAPD [byte rdx + 80], xmm11
	MOVAPD [byte rdx + 96], xmm6
	ADD rdx, 112
	.batch_process_finish:
	ADD rcx, 14
	JZ .return_ok
	.process_single:
	MOVSD xmm8, [rdi]
	ADD rdi, 8
	MOVSD xmm9, [rsi]
	ADD rsi, 8
	MINSD xmm8, xmm9
	MOVSD [rdx], xmm8
	ADD rdx, 8
	SUB rcx, 1
	JNZ .process_single
	.return_ok:
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
section .text.SandyBridge progbits alloc exec nowrite align=16
global _yepCore_Min_V64fV64f_V64f_SandyBridge
_yepCore_Min_V64fV64f_V64f_SandyBridge:
%else
section .text
global __yepCore_Min_V64fV64f_V64f_SandyBridge
__yepCore_Min_V64fV64f_V64f_SandyBridge:
%endif
	.ENTRY:
	TEST rdi, rdi
	JZ .return_null_pointer
	TEST rdi, 7
	JNZ .return_misaligned_pointer
	TEST rsi, rsi
	JZ .return_null_pointer
	TEST rsi, 7
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 7
	JNZ .return_misaligned_pointer
	TEST rcx, rcx
	JZ .return_ok
	TEST rdx, 31
	JZ .source_z_32b_aligned
	.source_z_32b_misaligned:
	VMOVSD xmm2, [rdi]
	ADD rdi, 8
	VMOVSD xmm1, [rsi]
	ADD rsi, 8
	VMINSD xmm2, xmm2, xmm1
	VMOVSD [rdx], xmm2
	ADD rdx, 8
	SUB rcx, 1
	JZ .return_ok
	TEST rdx, 31
	JNZ .source_z_32b_misaligned
	.source_z_32b_aligned:
	SUB rcx, 28
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPD ymm2, [rdi]
	VMOVUPD ymm8, [byte rdi + 32]
	VMOVUPD ymm13, [rsi]
	VMOVUPD ymm9, [byte rdi + 64]
	VMOVUPD ymm7, [byte rsi + 32]
	VMOVUPD ymm3, [byte rdi + 96]
	VMOVUPD ymm6, [byte rsi + 64]
	VMOVUPD ymm14, [dword rdi + 128]
	VMOVUPD ymm5, [byte rsi + 96]
	VMOVUPD ymm12, [dword rdi + 160]
	VMOVUPD ymm4, [dword rsi + 128]
	VMINPD ymm2, ymm2, ymm13
	VMOVUPD ymm11, [dword rdi + 192]
	VMOVUPD ymm15, [dword rsi + 160]
	VMINPD ymm8, ymm8, ymm7
	VMOVAPD [rdx], ymm2
	ADD rdi, 224
	VMOVUPD ymm10, [dword rsi + 192]
	VMINPD ymm9, ymm9, ymm6
	VMOVAPD [byte rdx + 32], ymm8
	SUB rcx, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPD ymm2, [rdi]
	ADD rsi, 224
	VMINPD ymm3, ymm3, ymm5
	VMOVAPD [byte rdx + 64], ymm9
	VMOVUPD ymm8, [byte rdi + 32]
	VMOVUPD ymm13, [rsi]
	VMINPD ymm14, ymm14, ymm4
	VMOVAPD [byte rdx + 96], ymm3
	VMOVUPD ymm9, [byte rdi + 64]
	VMOVUPD ymm7, [byte rsi + 32]
	VMINPD ymm12, ymm12, ymm15
	VMOVAPD [dword rdx + 128], ymm14
	VMOVUPD ymm3, [byte rdi + 96]
	VMOVUPD ymm6, [byte rsi + 64]
	VMINPD ymm11, ymm11, ymm10
	VMOVAPD [dword rdx + 160], ymm12
	VMOVUPD ymm14, [dword rdi + 128]
	VMOVUPD ymm5, [byte rsi + 96]
	VMOVAPD [dword rdx + 192], ymm11
	VMOVUPD ymm12, [dword rdi + 160]
	VMOVUPD ymm4, [dword rsi + 128]
	VMINPD ymm2, ymm2, ymm13
	ADD rdx, 224
	VMOVUPD ymm11, [dword rdi + 192]
	VMOVUPD ymm15, [dword rsi + 160]
	VMINPD ymm8, ymm8, ymm7
	VMOVAPD [rdx], ymm2
	ADD rdi, 224
	VMOVUPD ymm10, [dword rsi + 192]
	VMINPD ymm9, ymm9, ymm6
	VMOVAPD [byte rdx + 32], ymm8
	SUB rcx, 28
	JAE .process_batch
	.process_batch_epilogue:
	ADD rsi, 224
	VMINPD ymm3, ymm3, ymm5
	VMOVAPD [byte rdx + 64], ymm9
	VMINPD ymm14, ymm14, ymm4
	VMOVAPD [byte rdx + 96], ymm3
	VMINPD ymm12, ymm12, ymm15
	VMOVAPD [dword rdx + 128], ymm14
	VMINPD ymm11, ymm11, ymm10
	VMOVAPD [dword rdx + 160], ymm12
	VMOVAPD [dword rdx + 192], ymm11
	ADD rdx, 224
	.batch_process_finish:
	ADD rcx, 28
	JZ .return_ok
	.process_single:
	VMOVSD xmm8, [rdi]
	ADD rdi, 8
	VMOVSD xmm9, [rsi]
	ADD rsi, 8
	VMINSD xmm8, xmm8, xmm9
	VMOVSD [rdx], xmm8
	ADD rdx, 8
	SUB rcx, 1
	JNZ .process_single
	.return_ok:
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