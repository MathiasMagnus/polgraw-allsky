#pragma once

// Polgraw includes
#include <AmplitudeModulationCoeff.hpp>
#include <Sky.hpp>

// SYCL includes
#include <CL/sycl.hpp>

// STL includes
#include <cmath>

template <cl::sycl::access::mode Mode, typename T, typename TT = T>
struct lazy_modulated_accessor;

template <typename T, typename TT>
struct lazy_modulated_accessor<cl::sycl::access::mode::read, T, TT>
{
    lazy_modulated_accessor(const TT& sinalfr,
                            const TT& cosalfr,
                            const TT& sindel,
                            const TT& cosdel,
                            const TT& c2d,
                            const TT& c2sd,
                            const TT& omr,
                            cl::sycl::buffer<ampl_mod_coeff<T>> coeffs)
    {}

    cl::sycl::vec<T, 2> operator[](cl::sycl::item<1> i)
    {
        
    }

    TT sinalfr_, cosalfr_, sindel_, cosdel_, c2d_, c2sd_, omr_;
    cl::sycl::accessor<ampl_mod_coeff, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer> coeffs_;
};

template <typename T, typename TT = T>
struct lazy_modulated_series
{
    lazy_modulated_series(double sinalfr,
                          double cosalfr,
                          double sindel,
                          double cosdel,
                          double c2d,
                          double c2sd,
                          double omr,
                          cl::sycl::buffer<ampl_mod_coeff<T>> coeffs)
        : sinalfr_{static_cast<TT>(sinalfr)}
        , cosalfr_{static_cast<TT>(cosalfr)}
        , sindel_{static_cast<TT>(sindel)}
        , cosdel_{static_cast<TT>(cosdel)}
        , c2d_{static_cast<TT>(c2d)}
        , c2sd_{static_cast<TT>(c2sd)}
        , omr_{static_cast<TT>(omr)}
        , coeffs_{static_cast<TT>(coeffs)}
    {}

    template <cl::sycl::access::mode Mode, cl::sycl::access::target Target>
    lazy_modulated_accessor<Mode, T> get_access(cl::sycl::handler& cgh)
    {
        return 
    }

    TT sinalfr_, cosalfr_, sindel_, cosdel_, c2d_, c2sd_, omr_;
    cl::sycl::buffer<ampl_mod_coeff<T>> coeffs_;
};

template <typename T, typename TT = T>
struct lazy_modulator
{
    lazy_modulator(ampl_mod_coeff<T> coeffs, double phir, double omr)
        : coeffs_{cl::sycl::range<1>{1}}
        , cphir_{std::cos(phir)}
        , sphir_{std::sin(phir)}
        , omr_{omr}
    {
        coeffs_.get_access<cl::sycl::access::mode::discard_write>()[0] = coeffs;
    }

    lazy_modulated_series<T, TT> operator()(sky::celestial::coord c)
    {
        const double sinal_ = std::sin(c.alpha),
                     cosal_ = std::cos(c.alpha),
                     sindel_ = std::sin(c.delta),
                     cosdel_ = std::sin(c.delta),
                     sinalfr_ = sinal_ * cphir_ - cosal_ * sphir_,
                     cosalfr_ = cosal_ * cphir_ + sinal_ * sphir_,
                     c2d_ = std::pow(cosdel_, 2),
                     c2sd_ = sindel_ * cosdel_;

        return lazy_modulated_series<T, TT>{ sinalfr_, cosalfr_, sindel_, cosdel_, c2d_, c2sd_, omr_ };
    }

    cl::sycl::buffer<ampl_mod_coeff<T>> coeffs_;
    double cphir_, sphir_, omr_;
};
