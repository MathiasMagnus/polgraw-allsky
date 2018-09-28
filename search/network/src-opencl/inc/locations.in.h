// File configured via CMake
static const size_t kernel_path_count = ${KRN_LIST_LENGTH};
const char* kernel_paths[${KRN_LIST_LENGTH}] = { ${KRN_LIST} };
const char* kernel_inc_path = "${CMAKE_CURRENT_SOURCE_DIR}/krn";
