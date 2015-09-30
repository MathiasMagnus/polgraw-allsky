/*
 *                       Yeppp! library implementation
 *                   This file is auto-generated by Peach-Py,
 *        Portable Efficient Assembly Code-generator in Higher-level Python,
 *                  part of the Yeppp! library infrastructure
 * This file is part of Yeppp! library and licensed under the New BSD license.
 * See LICENSE.txt for the full text of the license.
 */

#include <yepPredefines.h>
#include <yepTypes.h>
#include <yepPrivate.h>
#include <core/SumSquares.disp.h>

#if defined(YEP_MSVC_COMPATIBLE_COMPILER)
	#pragma section(".rdata$DispatchTable", read)
	#pragma section(".data$DispatchPointer", read, write)
#endif

extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Default(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
#if defined(YEP_MICROSOFT_X64_ABI)
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Nehalem(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_SandyBridge(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Bulldozer(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Haswell(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
#endif // YEP_MICROSOFT_X64_ABI
#if defined(YEP_SYSTEMV_X64_ABI)
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Nehalem(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_SandyBridge(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Bulldozer(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V32f_S32f_Haswell(const Yep32f *YEP_RESTRICT v, Yep32f *YEP_RESTRICT sumSquares, YepSize length);
#endif // YEP_SYSTEMV_X64_ABI
YEP_USE_DISPATCH_TABLE_SECTION const FunctionDescriptor<YepStatus (YEPABI*)(const Yep32f *YEP_RESTRICT, Yep32f *YEP_RESTRICT, YepSize)> _dispatchTable_yepCore_SumSquares_V32f_S32f[] = 
{
	#if defined(YEP_MICROSOFT_X64_ABI)
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Nehalem, YepIsaFeaturesDefault, YepX86SimdFeatureSSE | YepX86SimdFeatureSSE3, YepX86SystemFeatureXMM, YepCpuMicroarchitectureNehalem, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_SandyBridge, YepIsaFeaturesDefault, YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureSandyBridge, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#ifndef YEP_MACOSX_OS
			YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Bulldozer, YepIsaFeaturesDefault, YepX86SimdFeatureAVX | YepX86SimdFeatureFMA4, YepX86SystemFeatureYMM, YepCpuMicroarchitectureBulldozer, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#endif // YEP_MACOSX_OS
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Haswell, YepIsaFeaturesDefault, YepX86SimdFeatureFMA3 | YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureHaswell, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
	#endif // YEP_MICROSOFT_X64_ABI
	#if defined(YEP_SYSTEMV_X64_ABI)
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Nehalem, YepIsaFeaturesDefault, YepX86SimdFeatureSSE | YepX86SimdFeatureSSE3, YepX86SystemFeatureXMM, YepCpuMicroarchitectureNehalem, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_SandyBridge, YepIsaFeaturesDefault, YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureSandyBridge, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#ifndef YEP_MACOSX_OS
			YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Bulldozer, YepIsaFeaturesDefault, YepX86SimdFeatureAVX | YepX86SimdFeatureFMA4, YepX86SystemFeatureYMM, YepCpuMicroarchitectureBulldozer, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#endif // YEP_MACOSX_OS
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Haswell, YepIsaFeaturesDefault, YepX86SimdFeatureFMA3 | YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureHaswell, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
	#endif // YEP_SYSTEMV_X64_ABI
	YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V32f_S32f_Default, YepIsaFeaturesDefault, YepSimdFeaturesDefault, YepSystemFeaturesDefault, YepCpuMicroarchitectureUnknown, "c++", "Naive", "None")
};

extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Default(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
#if defined(YEP_MICROSOFT_X64_ABI)
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Nehalem(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_SandyBridge(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Bulldozer(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Haswell(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
#endif // YEP_MICROSOFT_X64_ABI
#if defined(YEP_SYSTEMV_X64_ABI)
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Nehalem(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_SandyBridge(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Bulldozer(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
	extern "C" YEP_LOCAL_SYMBOL YepStatus YEPABI _yepCore_SumSquares_V64f_S64f_Haswell(const Yep64f *YEP_RESTRICT v, Yep64f *YEP_RESTRICT sumSquares, YepSize length);
#endif // YEP_SYSTEMV_X64_ABI
YEP_USE_DISPATCH_TABLE_SECTION const FunctionDescriptor<YepStatus (YEPABI*)(const Yep64f *YEP_RESTRICT, Yep64f *YEP_RESTRICT, YepSize)> _dispatchTable_yepCore_SumSquares_V64f_S64f[] = 
{
	#if defined(YEP_MICROSOFT_X64_ABI)
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Nehalem, YepIsaFeaturesDefault, YepX86SimdFeatureSSE | YepX86SimdFeatureSSE2, YepX86SystemFeatureXMM, YepCpuMicroarchitectureNehalem, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_SandyBridge, YepIsaFeaturesDefault, YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureSandyBridge, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#ifndef YEP_MACOSX_OS
			YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Bulldozer, YepIsaFeaturesDefault, YepX86SimdFeatureAVX | YepX86SimdFeatureFMA4, YepX86SystemFeatureYMM, YepCpuMicroarchitectureBulldozer, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#endif // YEP_MACOSX_OS
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Haswell, YepIsaFeaturesDefault, YepX86SimdFeatureFMA3 | YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureHaswell, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
	#endif // YEP_MICROSOFT_X64_ABI
	#if defined(YEP_SYSTEMV_X64_ABI)
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Nehalem, YepIsaFeaturesDefault, YepX86SimdFeatureSSE | YepX86SimdFeatureSSE2, YepX86SystemFeatureXMM, YepCpuMicroarchitectureNehalem, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_SandyBridge, YepIsaFeaturesDefault, YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureSandyBridge, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#ifndef YEP_MACOSX_OS
			YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Bulldozer, YepIsaFeaturesDefault, YepX86SimdFeatureAVX | YepX86SimdFeatureFMA4, YepX86SystemFeatureYMM, YepCpuMicroarchitectureBulldozer, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
		#endif // YEP_MACOSX_OS
		YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Haswell, YepIsaFeaturesDefault, YepX86SimdFeatureFMA3 | YepX86SimdFeatureAVX, YepX86SystemFeatureYMM, YepCpuMicroarchitectureHaswell, "asm", YEP_NULL_POINTER, YEP_NULL_POINTER),
	#endif // YEP_SYSTEMV_X64_ABI
	YEP_DESCRIBE_FUNCTION_IMPLEMENTATION(_yepCore_SumSquares_V64f_S64f_Default, YepIsaFeaturesDefault, YepSimdFeaturesDefault, YepSystemFeaturesDefault, YepCpuMicroarchitectureUnknown, "c++", "Naive", "None")
};


YEP_USE_DISPATCH_POINTER_SECTION YepStatus (YEPABI*_yepCore_SumSquares_V32f_S32f)(const Yep32f *YEP_RESTRICT, Yep32f *YEP_RESTRICT, YepSize) = YEP_NULL_POINTER;
YEP_USE_DISPATCH_POINTER_SECTION YepStatus (YEPABI*_yepCore_SumSquares_V64f_S64f)(const Yep64f *YEP_RESTRICT, Yep64f *YEP_RESTRICT, YepSize) = YEP_NULL_POINTER;

#if defined(YEP_MSVC_COMPATIBLE_COMPILER)
	#pragma code_seg( push, ".text$DispatchFunction" )
#endif

YEP_USE_DISPATCH_FUNCTION_SECTION YepStatus YEPABI yepCore_SumSquares_V32f_S32f(const Yep32f *YEP_RESTRICT vPointer, Yep32f *YEP_RESTRICT sumSquaresPointer, YepSize length) {
	return _yepCore_SumSquares_V32f_S32f(vPointer, sumSquaresPointer, length);
}

YEP_USE_DISPATCH_FUNCTION_SECTION YepStatus YEPABI yepCore_SumSquares_V64f_S64f(const Yep64f *YEP_RESTRICT vPointer, Yep64f *YEP_RESTRICT sumSquaresPointer, YepSize length) {
	return _yepCore_SumSquares_V64f_S64f(vPointer, sumSquaresPointer, length);
}

#if defined(YEP_MSVC_COMPATIBLE_COMPILER)
	#pragma code_seg( pop )
#endif