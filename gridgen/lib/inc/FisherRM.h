///////////////////////////////////////
// Name:        FisherRM.h
// Author:      Andrzej Pisarski
// Copyright:   Andrzej Pisarski
// License:     CC-BY-NC-ND
// Created:     20/05/2015
///////////////////////////////////////

#ifndef FISHERRM_H
#define FISHERRM_H

#include <vector>   // std::vector
#include <cstdint>  // std::size_t

class FisherRM
{
    public:
        FisherRM();
        FisherRM(const std::vector<double>&, const std::vector<double>&);   // vector<double>& ephemeris1, ephemeris2

        std::vector<double> get_prefisher() const;
        std::vector<double> postrmf(double) const;
        std::size_t get_ephemeris_length() const;
        unsigned int dim() const;                   // gives dimension of Fisher matrix

    protected:
        std::vector<double> m_prefisher;            // (unchanging) elements of reduced Fisher matrix
        std::size_t m_ephemeris_length;             // number of elements (data length) in ephemeris1
                                                    // (ephemeris2 should have this same length)
};

#endif // FISHERRM_H
