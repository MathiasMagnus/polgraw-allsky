// STL includes
#include <iostream>     // std::cout
#include <string>       // std::string
#include <fstream>      // std::ifstream
#include <vector>       // std::vector
#include <complex>      // std::complex
#include <iterator>     // std::istream_iterator, std::back_inserter
#include <numeric>      // std::accumulate
#include <algorithm>    // std::transform
#include <array>        // std::array
#include <atomic>       // std::aromic
#include <valarray>     // std::valarray

enum member
{
    frequency,
    spindown,
    declination,
    ascension,
    signal_to_noise
};

struct signal;

namespace detail
{
    template <member M> bool compare_signals(const signal& lhs, const signal& rhs);
}

struct signal
{
    double frequency,
           spindown,
           declination,
           ascension,
           signal_to_noise;

    bool operator==(const signal& other) const
    {
        return frequency == other.frequency &&
            spindown == other.spindown &&
            declination == other.declination &&
            ascension == other.ascension &&
            signal_to_noise == other.signal_to_noise;
    };

    bool operator<(const signal& other) const
    {
        if (frequency < other.frequency) return true;
        else if (frequency > other.frequency) return false;
        else
        {
            if (spindown < other.spindown) return true;
            else if (spindown > other.spindown) return false;
            else
            {
                if (declination < other.declination) return true;
                else if (declination > other.declination) return false;
                else
                {
                    if (ascension < other.ascension) return true;
                    else if (ascension > other.ascension) return false;
                    else
                    {
                        if (signal_to_noise < other.signal_to_noise) return true;
                        else if (signal_to_noise > other.signal_to_noise) return false;
                        else
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }

    template <member M>
    static bool compare(const signal& lhs, const signal& rhs)
    {
        return detail::compare_signals<M>(lhs, rhs);
    }

    signal& operator+=(const signal& rhs)
    {
        frequency += rhs.frequency;
        spindown += rhs.spindown;
        declination += rhs.declination;
        ascension += rhs.ascension;
        signal_to_noise += rhs.signal_to_noise;

        return *this;
    }

    signal operator-(const signal& rhs) const
    {
        return signal{ frequency - rhs.frequency,
                       spindown - rhs.spindown,
                       declination - rhs.declination,
                       ascension - rhs.ascension,
                       signal_to_noise - rhs.signal_to_noise };
    }

    signal operator/(const std::size_t& rhs) const
    {
        return signal{ frequency - rhs,
                       spindown - rhs,
                       declination - rhs,
                       ascension - rhs,
                       signal_to_noise - rhs };
    }
};

namespace std
{
    signal pow(signal base, int iexp)
    {
        return signal{ std::pow(base.frequency, iexp),
                       std::pow(base.spindown, iexp),
                       std::pow(base.declination, iexp),
                       std::pow(base.ascension, iexp),
                       std::pow(base.signal_to_noise, iexp) };
    }

    template< class T >
    std::valarray<T> pow( const std::valarray<T>& base,
                          const int iexp )
    {
        //return base.apply([=](const T& val){ return std::pow(val, iexp); });

        std::valarray<T> result(base.size());

        std::transform(std::begin(base), std::end(base),
                       &(*std::begin(result)),
                       [=](const T& val){ return std::pow(val, iexp); });

        return result;
    }

    signal sqrt(const signal& val)
    {
        return signal{ std::sqrt(val.frequency),
                       std::sqrt(val.spindown),
                       std::sqrt(val.declination),
                       std::sqrt(val.ascension),
                       std::sqrt(val.signal_to_noise) };
    }
}

std::istream& operator>>(std::istream& is, signal& sig)
{
    std::array<double, 5> temp;

    std::copy_n(std::istream_iterator<double>{ is }, 5, temp.begin());

    sig = signal{ temp[0],
                  temp[1],
                  temp[2],
                  temp[3],
                  temp[4] };

    return is;
}

std::ostream& operator<<(std::ostream& os, const signal& sig)
{
    os <<
        sig.frequency << '\t' <<
        sig.spindown << '\t' <<
        sig.declination << '\t' <<
        sig.ascension << '\t' <<
        sig.signal_to_noise << std::endl;

    return os;
}

std::vector<signal> read_candidates(const std::string& filename)
{
    std::ifstream ifs{ filename, std::ios::in };

    if (!ifs.is_open()) throw std::runtime_error{filename + " not found."};

    return std::vector<signal>(std::istream_iterator<signal>{ ifs },
                               std::istream_iterator<signal>{});
}

template <> bool detail::compare_signals<member::frequency>(const signal& lhs, const signal& rhs) { return lhs.frequency < rhs.frequency; }
template <> bool detail::compare_signals<member::spindown>(const signal& lhs, const signal& rhs) { return lhs.spindown < rhs.spindown; }
template <> bool detail::compare_signals<member::declination>(const signal& lhs, const signal& rhs) { return lhs.declination < rhs.declination; }
template <> bool detail::compare_signals<member::ascension>(const signal& lhs, const signal& rhs) { return lhs.ascension < rhs.ascension; }
template <> bool detail::compare_signals<member::signal_to_noise>(const signal& lhs, const signal& rhs) { return lhs.signal_to_noise < rhs.signal_to_noise; }

/// <summary>
///     For every element in range [first1, last1) copies the nearest element from the range [first2, last2)
///     to d_first, according to the user provided norm.
/// </summary>
template <typename InputIt1, typename RandomIt2, typename OutputIt, typename Norm>
OutputIt copy_nearest(InputIt1 first1, InputIt1 last1,
    RandomIt2 first2, RandomIt2 last2,
    OutputIt d_first,
    Norm norm)
{
    while (first1 != last1)
    {
        *d_first++ =
            *std::min_element(first2, last2,
                [=](const auto& a, const auto& b)
                {
                    return norm(*first1, a) < norm(*first1, b);
                });
        first1++;
    }
    return d_first;
}

signal obtain_signal(std::string filename)
{
    std::ifstream in{filename};

    std::vector<signal> vec(std::istream_iterator<signal>{ in },
                            std::istream_iterator<signal>{});
    
    return *std::max_element(vec.cbegin(), vec.cend(), signal::compare<signal_to_noise>);
}

std::pair<std::string, std::string> data_filenames(
    std::string root,
    std::string fpo,
    std::string hemisphere
)
{
    return {{},{}};
}

int main(int, char* argv[])
{
    std::string hostname{ argv[1] };
    std::size_t count = std::atoi(argv[2]);

    std::vector<std::size_t> i(count);
    std::iota(i.begin(), i.end(), 1);

    std::valarray<signal> ref_signals(count),
                          ocl_signals(count);
    std::transform(i.cbegin(), i.cend(),
                   std::begin(ref_signals),
                   [=, name = "Ref"](const std::size_t& i)
    {
        std::stringstream path;
        path << "./" << hostname << "." << name << ".triggers.test" << i << ".1.bin";
        std::cout << "Reading reference data set " << i << std::endl;
        return obtain_signal(path.str());
    });

    std::transform(i.cbegin(), i.cend(),
                   std::begin(ocl_signals),
                   [=, name = "Ocl.Run1"](const std::size_t& i)
    {
        std::stringstream path;
        path << "./" << hostname << "." << name << ".triggers.test" << i << ".1.bin";
        std::cout << "Reading OpenCL data set " << i << std::endl;
        return obtain_signal(path.str());
    });

    std::valarray<signal> diff = ref_signals - ocl_signals;
    signal avg = diff.sum() / count,
           dev = std::sqrt(std::pow(std::valarray<signal>(diff - avg), 2).sum() / count),
           min = diff.min(),
           max = diff.max();

    std::cout << "Minimal difference:\n\n" << min << std::endl;
    std::cout << "Maximal difference:\n\n" << max << std::endl;
    std::cout << "Avarage difference:\n\n" << avg << std::endl;
    std::cout << "Std.dev difference:\n\n" << dev << std::endl;
    
    return 0;
}
