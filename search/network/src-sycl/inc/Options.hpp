#pragma once

// SYCL include
#include <CL/sycl.hpp>  // cl::sycl::info::device_type

// STL includes
#include <cstddef>      // std::size_t
#include <string>       // std::string


namespace cli
{
    struct options
    {
        std::size_t length, plat_id, dev_id;
        cl::sycl::info::device_type dev_type;
        bool quiet;
    };

    options parse(int argc, char** argv, const std::string banner);

    class error
    {
    public:

        error() = default;
        error(const error&) = default;
        error(error&&) = default;
        ~error() = default;

        error(std::string message) : m_message(message) {}

        const char* what() { return m_message.c_str(); }

    private:

        std::string m_message;
    };
}