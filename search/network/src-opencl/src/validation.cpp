// STL includes
#include <iostream>     // std::cout
#include <string>       // std::string
#include <fstream>      // ifstream
#include <vector>       // std::vector
#include <complex>      // std::complex
#include <iterator>     // std::istream_iterator, std::back_inserter
#include <numeric>      // std::accumulate
#include <algorithm>	// std::transform, all_of


int main(int argc, char* argv[])
{
	// Params
	using real = double;
	using complex = std::complex<real>;
	real exponent = 4,      // sensitivity to deviations
		 threshold = 1e-9;  // max allowed relative deviation of integrals

	// Test
	std::string ocl_path{ argv[1] },
		        ref_path{ argv[2] };

	std::ifstream ocl_stream{ ocl_path, std::ios::in },
		          ref_stream{ ref_path, std::ios::in };

	if (!ocl_stream.is_open() || !ref_stream.is_open())
		return -1;

	std::vector<complex> ocl_vec(std::istream_iterator<complex>{ ocl_stream },
		                         std::istream_iterator<complex>{}),
		                 ref_vec(std::istream_iterator<complex>{ ref_stream },
						         std::istream_iterator<complex>{});

	double ref_integral = // simple rectangle integral of the complex magnitudes
		std::accumulate(ref_vec.cbegin(), ref_vec.cend(),
		                (real)0,
		                [](const real& acc, const complex& val) { return acc + std::abs(val); });

	std::vector<real> deviations;

	std::transform(ocl_vec.cbegin(), ocl_vec.cend(),
		           ref_vec.cbegin(),
		           std::back_inserter(deviations),
		           [](const complex& ocl_val, const complex& ref_val)
	{
		return std::pow(std::abs(std::abs(ocl_val) - std::abs(ref_val)),
			            (real)4); // diff^4 is very sensitive to point-wise differences
	});

    double diff_integral = std::accumulate(deviations.cbegin(), deviations.cend(),
                                           (real)0);

    bool passed = diff_integral / ref_integral < threshold;

	if (passed)
		return 0;
	else
		return -1;
}
