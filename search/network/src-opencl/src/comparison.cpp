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
};

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

int main(int argc, char* argv[])
{
    // Test
    std::string ocl_path{ argv[1] },
        ref_path{ argv[2] },
        out_path{ argv[3] };

    std::ifstream ocl_stream{ ocl_path, std::ios::in },
        ref_stream{ ref_path, std::ios::in };

    if (!ocl_stream.is_open() || !ref_stream.is_open())
        return -1;

    std::vector<signal> ocl_vec(std::istream_iterator<signal>{ ocl_stream },
        std::istream_iterator<signal>{}),
        ref_vec(std::istream_iterator<signal>{ ref_stream },
            std::istream_iterator<signal>{});

    auto norm =
        [freq_range = std::max_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<frequency>)->frequency - std::min_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<frequency>)->frequency,
        spin_range = std::max_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<spindown>)->spindown - std::min_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<spindown>)->spindown,
        decl_range = std::max_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<declination>)->declination - std::min_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<declination>)->declination,
        asc_range = std::max_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<ascension>)->ascension - std::min_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<ascension>)->ascension,
        snr_range = std::max_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<signal_to_noise>)->signal_to_noise - std::min_element(ref_vec.cbegin(), ref_vec.cend(), signal::compare<signal_to_noise>)->signal_to_noise](const signal& x, const signal& y)
    {
        return std::abs(x.frequency - y.frequency) / freq_range +
            std::abs(x.spindown - y.spindown) / spin_range +
            std::abs(x.declination - y.declination) / decl_range +
            std::abs(x.ascension - y.ascension) / asc_range +
            std::abs(x.signal_to_noise - y.signal_to_noise) / snr_range;
    };

    std::cout << ocl_path << " has " << ocl_vec.size() << " data points." << std::endl;
    std::cout << ref_path << " has " << ref_vec.size() << " data points." << std::endl;

    std::vector<signal> filtered;
    copy_nearest(ref_vec.cbegin(), ref_vec.cend(),
                 ocl_vec.cbegin(), ocl_vec.cend(),
                 std::back_inserter(filtered),
                 norm);

    std::vector<signal> missing;
    std::copy_if(ocl_vec.cbegin(), ocl_vec.cend(),
                 std::back_inserter(missing),
                 [&](const signal sig) { return std::find(filtered.cbegin(), filtered.cend(), sig) == filtered.cend(); });

    if (missing.size() != 0)
    {
        std::cout << missing.size() << " data points were filtered out." << std::endl;
        std::sort(missing.begin(), missing.end(), signal::compare<signal_to_noise>);
        std::cout << "Minimum SNR of missing points: " << missing[0].signal_to_noise << std::endl;
        std::size_t count = missing.size() / 10;
        std::cout << "SNR of top 10% missing points: " << std::endl;
        std::transform(missing.crbegin(), missing.crbegin() + 10,
                       std::ostream_iterator<double>{ std::cout, "\n" },
                       [](const signal& sig){ return sig.signal_to_noise; });
        std::cout << "Avarage SNR of missing points: " <<
            std::accumulate(missing.cbegin(), missing.cend(), 0.0,
                            [](const double& acc, const signal& sig){ return acc + sig.signal_to_noise; }) / missing.size() << std::endl;
    }
    else
    {
        std::cout << "No points filtered out.";
    }

    std::ofstream out_full{ out_path + ".full.bin" };
    out_full << std::scientific;
    out_full.precision(6);
    std::copy(filtered.cbegin(), filtered.cend(),
              std::ostream_iterator<signal>{ out_full });
    /*
    std::sort(ref_vec.begin(), ref_vec.end());
    std::sort(filtered.begin(), filtered.end());

    ocl_vec.erase(std::unique(ocl_vec.begin(), ocl_vec.end()),
        ocl_vec.end());
    ref_vec.erase(std::unique(ref_vec.begin(), ref_vec.end()),
        ref_vec.end());
    filtered.erase(std::unique(filtered.begin(), filtered.end()),
                   filtered.end());

    std::cout << ocl_path << " has " << ocl_vec.size() << " unique data points." << std::endl;
    std::cout << ref_path << " has " << ref_vec.size() << " unique data points." << std::endl;
    std::cout << "filtered vec has " << filtered.size() << " unique data points." << std::endl;

    std::ofstream out_filtered{ out_path + ".filtered.bin" };
    out_filtered << std::scientific;
    out_filtered.precision(6);
    std::copy(filtered.cbegin(), filtered.cend(),
              std::ostream_iterator<signal>{ out_filtered });
    */
    std::vector<double> diff;
    std::transform(ref_vec.cbegin(), ref_vec.cend(), filtered.cbegin(), std::back_inserter(diff), norm);

    auto avg_diff = std::accumulate(diff.cbegin(), diff.cend(), 0.0) / diff.size();
    auto dev_diff = std::sqrt(std::accumulate(diff.cbegin(), diff.cend(), 0.0,
        [=](const double& acc, const double& val)
        {
            return acc + std::pow(val - avg_diff, 2);
        }) / (diff.size() - 1));

    std::cout << "Avarage difference: " << avg_diff << std::endl;
    std::cout << "Standard deviation of differences: " << dev_diff << std::endl;

    std::ofstream snr_vs_diff{ out_path + ".snr_vs_diff.bin" };
    for (std::size_t i = 0; i < filtered.size(); ++i)
        snr_vs_diff << filtered[i].signal_to_noise << "\t" << diff[i] << std::endl;

    return 0;
}
