;                       Yeppp! library implementation
;                   This file is auto-generated by Peach-Py,
;        Portable Efficient Assembly Code-generator in Higher-level Python,
;                  part of the Yeppp! library infrastructure
; This file is part of Yeppp! library and licensed under the New BSD license.
; See LICENSE.txt for the full text of the license.

section .text$e code align=16
global _yepCore_Min_V32fV32f_V32f_Nehalem
_yepCore_Min_V32fV32f_V32f_Nehalem:
	.ENTRY:
	SUB rsp, 152
	MOVAPS [rsp], xmm13
	MOVAPS [byte rsp + 16], xmm14
	MOVAPS [byte rsp + 32], xmm11
	MOVAPS [byte rsp + 48], xmm12
	MOVAPS [byte rsp + 64], xmm6
	MOVAPS [byte rsp + 80], xmm10
	MOVAPS [byte rsp + 96], xmm9
	MOVAPS [byte rsp + 112], xmm8
	MOVAPS [dword rsp + 128], xmm7
	TEST rcx, rcx
	JZ .return_null_pointer
	TEST rcx, 3
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 3
	JNZ .return_misaligned_pointer
	TEST r8, r8
	JZ .return_null_pointer
	TEST r8, 3
	JNZ .return_misaligned_pointer
	TEST r9, r9
	JZ .return_ok
	TEST r8, 15
	JZ .source_z_16b_aligned
	.source_z_16b_misaligned:
	MOVSS xmm13, [rcx]
	ADD rcx, 4
	MOVSS xmm14, [rdx]
	ADD rdx, 4
	MINSS xmm13, xmm14
	MOVSS [r8], xmm13
	ADD r8, 4
	SUB r9, 1
	JZ .return_ok
	TEST r8, 15
	JNZ .source_z_16b_misaligned
	.source_z_16b_aligned:
	SUB r9, 28
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPS xmm13, [rcx]
	MOVUPS xmm4, [byte rcx + 16]
	MOVUPS xmm11, [rdx]
	MOVUPS xmm5, [byte rcx + 32]
	MOVUPS xmm12, [byte rdx + 16]
	MOVUPS xmm1, [byte rcx + 48]
	MOVUPS xmm0, [byte rdx + 32]
	MOVUPS xmm6, [byte rcx + 64]
	MOVUPS xmm3, [byte rdx + 48]
	MOVUPS xmm2, [byte rcx + 80]
	MOVUPS xmm10, [byte rdx + 64]
	MINPS xmm13, xmm11
	MOVUPS xmm9, [byte rcx + 96]
	MOVUPS xmm8, [byte rdx + 80]
	MINPS xmm4, xmm12
	MOVAPS [r8], xmm13
	ADD rcx, 112
	MOVUPS xmm7, [byte rdx + 96]
	MINPS xmm5, xmm0
	MOVAPS [byte r8 + 16], xmm4
	SUB r9, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPS xmm13, [rcx]
	ADD rdx, 112
	MINPS xmm1, xmm3
	MOVAPS [byte r8 + 32], xmm5
	MOVUPS xmm4, [byte rcx + 16]
	MOVUPS xmm11, [rdx]
	MINPS xmm6, xmm10
	MOVAPS [byte r8 + 48], xmm1
	MOVUPS xmm5, [byte rcx + 32]
	MOVUPS xmm12, [byte rdx + 16]
	MINPS xmm2, xmm8
	MOVAPS [byte r8 + 64], xmm6
	MOVUPS xmm1, [byte rcx + 48]
	MOVUPS xmm0, [byte rdx + 32]
	MINPS xmm9, xmm7
	MOVAPS [byte r8 + 80], xmm2
	MOVUPS xmm6, [byte rcx + 64]
	MOVUPS xmm3, [byte rdx + 48]
	MOVAPS [byte r8 + 96], xmm9
	MOVUPS xmm2, [byte rcx + 80]
	MOVUPS xmm10, [byte rdx + 64]
	MINPS xmm13, xmm11
	ADD r8, 112
	MOVUPS xmm9, [byte rcx + 96]
	MOVUPS xmm8, [byte rdx + 80]
	MINPS xmm4, xmm12
	MOVAPS [r8], xmm13
	ADD rcx, 112
	MOVUPS xmm7, [byte rdx + 96]
	MINPS xmm5, xmm0
	MOVAPS [byte r8 + 16], xmm4
	SUB r9, 28
	JAE .process_batch
	.process_batch_epilogue:
	ADD rdx, 112
	MINPS xmm1, xmm3
	MOVAPS [byte r8 + 32], xmm5
	MINPS xmm6, xmm10
	MOVAPS [byte r8 + 48], xmm1
	MINPS xmm2, xmm8
	MOVAPS [byte r8 + 64], xmm6
	MINPS xmm9, xmm7
	MOVAPS [byte r8 + 80], xmm2
	MOVAPS [byte r8 + 96], xmm9
	ADD r8, 112
	.batch_process_finish:
	ADD r9, 28
	JZ .return_ok
	.process_single:
	MOVSS xmm4, [rcx]
	ADD rcx, 4
	MOVSS xmm5, [rdx]
	ADD rdx, 4
	MINSS xmm4, xmm5
	MOVSS [r8], xmm4
	ADD r8, 4
	SUB r9, 1
	JNZ .process_single
	.return_ok:
	XOR eax, eax
	.return:
	MOVAPS xmm13, [rsp]
	MOVAPS xmm14, [byte rsp + 16]
	MOVAPS xmm11, [byte rsp + 32]
	MOVAPS xmm12, [byte rsp + 48]
	MOVAPS xmm6, [byte rsp + 64]
	MOVAPS xmm10, [byte rsp + 80]
	MOVAPS xmm9, [byte rsp + 96]
	MOVAPS xmm8, [byte rsp + 112]
	MOVAPS xmm7, [dword rsp + 128]
	ADD rsp, 152
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

section .text$f code align=16
global _yepCore_Min_V32fV32f_V32f_SandyBridge
_yepCore_Min_V32fV32f_V32f_SandyBridge:
	.ENTRY:
	SUB rsp, 152
	VMOVAPS [rsp], xmm13
	VMOVAPS [byte rsp + 16], xmm14
	VMOVAPS [byte rsp + 32], xmm8
	VMOVAPS [byte rsp + 48], xmm12
	VMOVAPS [byte rsp + 64], xmm9
	VMOVAPS [byte rsp + 80], xmm6
	VMOVAPS [byte rsp + 96], xmm10
	VMOVAPS [byte rsp + 112], xmm11
	VMOVAPS [dword rsp + 128], xmm7
	TEST rcx, rcx
	JZ .return_null_pointer
	TEST rcx, 3
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 3
	JNZ .return_misaligned_pointer
	TEST r8, r8
	JZ .return_null_pointer
	TEST r8, 3
	JNZ .return_misaligned_pointer
	TEST r9, r9
	JZ .return_ok
	TEST r8, 31
	JZ .source_z_32b_aligned
	.source_z_32b_misaligned:
	VMOVSS xmm13, [rcx]
	ADD rcx, 4
	VMOVSS xmm14, [rdx]
	ADD rdx, 4
	VMINSS xmm13, xmm13, xmm14
	VMOVSS [r8], xmm13
	ADD r8, 4
	SUB r9, 1
	JZ .return_ok
	TEST r8, 31
	JNZ .source_z_32b_misaligned
	.source_z_32b_aligned:
	SUB r9, 56
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPS ymm13, [rcx]
	VMOVUPS ymm4, [byte rcx + 32]
	VMOVUPS ymm0, [rdx]
	VMOVUPS ymm5, [byte rcx + 64]
	VMOVUPS ymm8, [byte rdx + 32]
	VMOVUPS ymm12, [byte rcx + 96]
	VMOVUPS ymm9, [byte rdx + 64]
	VMOVUPS ymm6, [dword rcx + 128]
	VMOVUPS ymm10, [byte rdx + 96]
	VMOVUPS ymm1, [dword rcx + 160]
	VMOVUPS ymm11, [dword rdx + 128]
	VMINPS ymm13, ymm13, ymm0
	VMOVUPS ymm2, [dword rcx + 192]
	VMOVUPS ymm7, [dword rdx + 160]
	VMINPS ymm4, ymm4, ymm8
	VMOVAPS [r8], ymm13
	ADD rcx, 224
	VMOVUPS ymm3, [dword rdx + 192]
	VMINPS ymm5, ymm5, ymm9
	VMOVAPS [byte r8 + 32], ymm4
	SUB r9, 56
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPS ymm13, [rcx]
	ADD rdx, 224
	VMINPS ymm12, ymm12, ymm10
	VMOVAPS [byte r8 + 64], ymm5
	VMOVUPS ymm4, [byte rcx + 32]
	VMOVUPS ymm0, [rdx]
	VMINPS ymm6, ymm6, ymm11
	VMOVAPS [byte r8 + 96], ymm12
	VMOVUPS ymm5, [byte rcx + 64]
	VMOVUPS ymm8, [byte rdx + 32]
	VMINPS ymm1, ymm1, ymm7
	VMOVAPS [dword r8 + 128], ymm6
	VMOVUPS ymm12, [byte rcx + 96]
	VMOVUPS ymm9, [byte rdx + 64]
	VMINPS ymm2, ymm2, ymm3
	VMOVAPS [dword r8 + 160], ymm1
	VMOVUPS ymm6, [dword rcx + 128]
	VMOVUPS ymm10, [byte rdx + 96]
	VMOVAPS [dword r8 + 192], ymm2
	VMOVUPS ymm1, [dword rcx + 160]
	VMOVUPS ymm11, [dword rdx + 128]
	VMINPS ymm13, ymm13, ymm0
	ADD r8, 224
	VMOVUPS ymm2, [dword rcx + 192]
	VMOVUPS ymm7, [dword rdx + 160]
	VMINPS ymm4, ymm4, ymm8
	VMOVAPS [r8], ymm13
	ADD rcx, 224
	VMOVUPS ymm3, [dword rdx + 192]
	VMINPS ymm5, ymm5, ymm9
	VMOVAPS [byte r8 + 32], ymm4
	SUB r9, 56
	JAE .process_batch
	.process_batch_epilogue:
	ADD rdx, 224
	VMINPS ymm12, ymm12, ymm10
	VMOVAPS [byte r8 + 64], ymm5
	VMINPS ymm6, ymm6, ymm11
	VMOVAPS [byte r8 + 96], ymm12
	VMINPS ymm1, ymm1, ymm7
	VMOVAPS [dword r8 + 128], ymm6
	VMINPS ymm2, ymm2, ymm3
	VMOVAPS [dword r8 + 160], ymm1
	VMOVAPS [dword r8 + 192], ymm2
	ADD r8, 224
	.batch_process_finish:
	ADD r9, 56
	JZ .return_ok
	.process_single:
	VMOVSS xmm4, [rcx]
	ADD rcx, 4
	VMOVSS xmm5, [rdx]
	ADD rdx, 4
	VMINSS xmm4, xmm4, xmm5
	VMOVSS [r8], xmm4
	ADD r8, 4
	SUB r9, 1
	JNZ .process_single
	.return_ok:
	XOR eax, eax
	.return:
	VMOVAPS xmm13, [rsp]
	VMOVAPS xmm14, [byte rsp + 16]
	VMOVAPS xmm8, [byte rsp + 32]
	VMOVAPS xmm12, [byte rsp + 48]
	VMOVAPS xmm9, [byte rsp + 64]
	VMOVAPS xmm6, [byte rsp + 80]
	VMOVAPS xmm10, [byte rsp + 96]
	VMOVAPS xmm11, [byte rsp + 112]
	VMOVAPS xmm7, [dword rsp + 128]
	ADD rsp, 152
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

section .text$e code align=16
global _yepCore_Min_V64fV64f_V64f_Nehalem
_yepCore_Min_V64fV64f_V64f_Nehalem:
	.ENTRY:
	SUB rsp, 152
	MOVAPS [rsp], xmm13
	MOVAPS [byte rsp + 16], xmm14
	MOVAPS [byte rsp + 32], xmm11
	MOVAPS [byte rsp + 48], xmm12
	MOVAPS [byte rsp + 64], xmm6
	MOVAPS [byte rsp + 80], xmm10
	MOVAPS [byte rsp + 96], xmm9
	MOVAPS [byte rsp + 112], xmm8
	MOVAPS [dword rsp + 128], xmm7
	TEST rcx, rcx
	JZ .return_null_pointer
	TEST rcx, 7
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 7
	JNZ .return_misaligned_pointer
	TEST r8, r8
	JZ .return_null_pointer
	TEST r8, 7
	JNZ .return_misaligned_pointer
	TEST r9, r9
	JZ .return_ok
	TEST r8, 15
	JZ .source_z_16b_aligned
	.source_z_16b_misaligned:
	MOVSD xmm13, [rcx]
	ADD rcx, 8
	MOVSD xmm14, [rdx]
	ADD rdx, 8
	MINSD xmm13, xmm14
	MOVSD [r8], xmm13
	ADD r8, 8
	SUB r9, 1
	JZ .return_ok
	TEST r8, 15
	JNZ .source_z_16b_misaligned
	.source_z_16b_aligned:
	SUB r9, 14
	JB .batch_process_finish
	.process_batch_prologue:
	MOVUPD xmm13, [rcx]
	MOVUPD xmm4, [byte rcx + 16]
	MOVUPD xmm11, [rdx]
	MOVUPD xmm5, [byte rcx + 32]
	MOVUPD xmm12, [byte rdx + 16]
	MOVUPD xmm1, [byte rcx + 48]
	MOVUPD xmm0, [byte rdx + 32]
	MOVUPD xmm6, [byte rcx + 64]
	MOVUPD xmm3, [byte rdx + 48]
	MOVUPD xmm2, [byte rcx + 80]
	MOVUPD xmm10, [byte rdx + 64]
	MINPD xmm13, xmm11
	MOVUPD xmm9, [byte rcx + 96]
	MOVUPD xmm8, [byte rdx + 80]
	MINPD xmm4, xmm12
	MOVAPD [r8], xmm13
	ADD rcx, 112
	MOVUPD xmm7, [byte rdx + 96]
	MINPD xmm5, xmm0
	MOVAPD [byte r8 + 16], xmm4
	SUB r9, 14
	JB .process_batch_epilogue
	align 16
	.process_batch:
	MOVUPD xmm13, [rcx]
	ADD rdx, 112
	MINPD xmm1, xmm3
	MOVAPD [byte r8 + 32], xmm5
	MOVUPD xmm4, [byte rcx + 16]
	MOVUPD xmm11, [rdx]
	MINPD xmm6, xmm10
	MOVAPD [byte r8 + 48], xmm1
	MOVUPD xmm5, [byte rcx + 32]
	MOVUPD xmm12, [byte rdx + 16]
	MINPD xmm2, xmm8
	MOVAPD [byte r8 + 64], xmm6
	MOVUPD xmm1, [byte rcx + 48]
	MOVUPD xmm0, [byte rdx + 32]
	MINPD xmm9, xmm7
	MOVAPD [byte r8 + 80], xmm2
	MOVUPD xmm6, [byte rcx + 64]
	MOVUPD xmm3, [byte rdx + 48]
	MOVAPD [byte r8 + 96], xmm9
	MOVUPD xmm2, [byte rcx + 80]
	MOVUPD xmm10, [byte rdx + 64]
	MINPD xmm13, xmm11
	ADD r8, 112
	MOVUPD xmm9, [byte rcx + 96]
	MOVUPD xmm8, [byte rdx + 80]
	MINPD xmm4, xmm12
	MOVAPD [r8], xmm13
	ADD rcx, 112
	MOVUPD xmm7, [byte rdx + 96]
	MINPD xmm5, xmm0
	MOVAPD [byte r8 + 16], xmm4
	SUB r9, 14
	JAE .process_batch
	.process_batch_epilogue:
	ADD rdx, 112
	MINPD xmm1, xmm3
	MOVAPD [byte r8 + 32], xmm5
	MINPD xmm6, xmm10
	MOVAPD [byte r8 + 48], xmm1
	MINPD xmm2, xmm8
	MOVAPD [byte r8 + 64], xmm6
	MINPD xmm9, xmm7
	MOVAPD [byte r8 + 80], xmm2
	MOVAPD [byte r8 + 96], xmm9
	ADD r8, 112
	.batch_process_finish:
	ADD r9, 14
	JZ .return_ok
	.process_single:
	MOVSD xmm4, [rcx]
	ADD rcx, 8
	MOVSD xmm5, [rdx]
	ADD rdx, 8
	MINSD xmm4, xmm5
	MOVSD [r8], xmm4
	ADD r8, 8
	SUB r9, 1
	JNZ .process_single
	.return_ok:
	XOR eax, eax
	.return:
	MOVAPS xmm13, [rsp]
	MOVAPS xmm14, [byte rsp + 16]
	MOVAPS xmm11, [byte rsp + 32]
	MOVAPS xmm12, [byte rsp + 48]
	MOVAPS xmm6, [byte rsp + 64]
	MOVAPS xmm10, [byte rsp + 80]
	MOVAPS xmm9, [byte rsp + 96]
	MOVAPS xmm8, [byte rsp + 112]
	MOVAPS xmm7, [dword rsp + 128]
	ADD rsp, 152
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return

section .text$f code align=16
global _yepCore_Min_V64fV64f_V64f_SandyBridge
_yepCore_Min_V64fV64f_V64f_SandyBridge:
	.ENTRY:
	SUB rsp, 152
	VMOVAPS [rsp], xmm13
	VMOVAPS [byte rsp + 16], xmm14
	VMOVAPS [byte rsp + 32], xmm8
	VMOVAPS [byte rsp + 48], xmm12
	VMOVAPS [byte rsp + 64], xmm9
	VMOVAPS [byte rsp + 80], xmm6
	VMOVAPS [byte rsp + 96], xmm10
	VMOVAPS [byte rsp + 112], xmm11
	VMOVAPS [dword rsp + 128], xmm7
	TEST rcx, rcx
	JZ .return_null_pointer
	TEST rcx, 7
	JNZ .return_misaligned_pointer
	TEST rdx, rdx
	JZ .return_null_pointer
	TEST rdx, 7
	JNZ .return_misaligned_pointer
	TEST r8, r8
	JZ .return_null_pointer
	TEST r8, 7
	JNZ .return_misaligned_pointer
	TEST r9, r9
	JZ .return_ok
	TEST r8, 31
	JZ .source_z_32b_aligned
	.source_z_32b_misaligned:
	VMOVSD xmm13, [rcx]
	ADD rcx, 8
	VMOVSD xmm14, [rdx]
	ADD rdx, 8
	VMINSD xmm13, xmm13, xmm14
	VMOVSD [r8], xmm13
	ADD r8, 8
	SUB r9, 1
	JZ .return_ok
	TEST r8, 31
	JNZ .source_z_32b_misaligned
	.source_z_32b_aligned:
	SUB r9, 28
	JB .batch_process_finish
	.process_batch_prologue:
	VMOVUPD ymm13, [rcx]
	VMOVUPD ymm4, [byte rcx + 32]
	VMOVUPD ymm0, [rdx]
	VMOVUPD ymm5, [byte rcx + 64]
	VMOVUPD ymm8, [byte rdx + 32]
	VMOVUPD ymm12, [byte rcx + 96]
	VMOVUPD ymm9, [byte rdx + 64]
	VMOVUPD ymm6, [dword rcx + 128]
	VMOVUPD ymm10, [byte rdx + 96]
	VMOVUPD ymm1, [dword rcx + 160]
	VMOVUPD ymm11, [dword rdx + 128]
	VMINPD ymm13, ymm13, ymm0
	VMOVUPD ymm2, [dword rcx + 192]
	VMOVUPD ymm7, [dword rdx + 160]
	VMINPD ymm4, ymm4, ymm8
	VMOVAPD [r8], ymm13
	ADD rcx, 224
	VMOVUPD ymm3, [dword rdx + 192]
	VMINPD ymm5, ymm5, ymm9
	VMOVAPD [byte r8 + 32], ymm4
	SUB r9, 28
	JB .process_batch_epilogue
	align 16
	.process_batch:
	VMOVUPD ymm13, [rcx]
	ADD rdx, 224
	VMINPD ymm12, ymm12, ymm10
	VMOVAPD [byte r8 + 64], ymm5
	VMOVUPD ymm4, [byte rcx + 32]
	VMOVUPD ymm0, [rdx]
	VMINPD ymm6, ymm6, ymm11
	VMOVAPD [byte r8 + 96], ymm12
	VMOVUPD ymm5, [byte rcx + 64]
	VMOVUPD ymm8, [byte rdx + 32]
	VMINPD ymm1, ymm1, ymm7
	VMOVAPD [dword r8 + 128], ymm6
	VMOVUPD ymm12, [byte rcx + 96]
	VMOVUPD ymm9, [byte rdx + 64]
	VMINPD ymm2, ymm2, ymm3
	VMOVAPD [dword r8 + 160], ymm1
	VMOVUPD ymm6, [dword rcx + 128]
	VMOVUPD ymm10, [byte rdx + 96]
	VMOVAPD [dword r8 + 192], ymm2
	VMOVUPD ymm1, [dword rcx + 160]
	VMOVUPD ymm11, [dword rdx + 128]
	VMINPD ymm13, ymm13, ymm0
	ADD r8, 224
	VMOVUPD ymm2, [dword rcx + 192]
	VMOVUPD ymm7, [dword rdx + 160]
	VMINPD ymm4, ymm4, ymm8
	VMOVAPD [r8], ymm13
	ADD rcx, 224
	VMOVUPD ymm3, [dword rdx + 192]
	VMINPD ymm5, ymm5, ymm9
	VMOVAPD [byte r8 + 32], ymm4
	SUB r9, 28
	JAE .process_batch
	.process_batch_epilogue:
	ADD rdx, 224
	VMINPD ymm12, ymm12, ymm10
	VMOVAPD [byte r8 + 64], ymm5
	VMINPD ymm6, ymm6, ymm11
	VMOVAPD [byte r8 + 96], ymm12
	VMINPD ymm1, ymm1, ymm7
	VMOVAPD [dword r8 + 128], ymm6
	VMINPD ymm2, ymm2, ymm3
	VMOVAPD [dword r8 + 160], ymm1
	VMOVAPD [dword r8 + 192], ymm2
	ADD r8, 224
	.batch_process_finish:
	ADD r9, 28
	JZ .return_ok
	.process_single:
	VMOVSD xmm4, [rcx]
	ADD rcx, 8
	VMOVSD xmm5, [rdx]
	ADD rdx, 8
	VMINSD xmm4, xmm4, xmm5
	VMOVSD [r8], xmm4
	ADD r8, 8
	SUB r9, 1
	JNZ .process_single
	.return_ok:
	XOR eax, eax
	.return:
	VMOVAPS xmm13, [rsp]
	VMOVAPS xmm14, [byte rsp + 16]
	VMOVAPS xmm8, [byte rsp + 32]
	VMOVAPS xmm12, [byte rsp + 48]
	VMOVAPS xmm9, [byte rsp + 64]
	VMOVAPS xmm6, [byte rsp + 80]
	VMOVAPS xmm10, [byte rsp + 96]
	VMOVAPS xmm11, [byte rsp + 112]
	VMOVAPS xmm7, [dword rsp + 128]
	ADD rsp, 152
	VZEROUPPER
	RET
	.return_null_pointer:
	MOV eax, 1
	JMP .return
	.return_misaligned_pointer:
	MOV eax, 2
	JMP .return
