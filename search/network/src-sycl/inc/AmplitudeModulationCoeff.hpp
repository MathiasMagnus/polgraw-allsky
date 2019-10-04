#pragma once

template <typename T>
struct ampl_mod_coeff
{
    template <typename TT>
    ampl_mod_coeff(ampl_mod_coeff<TT> in)
     : c1{static_cast<T>(in.c1)}
     , c2{static_cast<T>(in.c2)}
     , c3{static_cast<T>(in.c3)}
     , c4{static_cast<T>(in.c4)}
     , c5{static_cast<T>(in.c5)}
     , c6{static_cast<T>(in.c6)}
     , c7{static_cast<T>(in.c7)}
     , c8{static_cast<T>(in.c8)}
     , c9{static_cast<T>(in.c9)}

    T c1, c2, c3, c4, c5, c6, c7, c8, c9;
};
