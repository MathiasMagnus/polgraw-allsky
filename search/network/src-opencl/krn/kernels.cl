// Polgraw includes
#include <floats.h.cl>       // real_t, complex_t
#include <kernels.h.cl>      // function declarations

//#pragma OPENCL EXTENSION cl_amd_printf : enable

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void modvir_kern(__global real_t* aa_d,
                          __global real_t* bb_d,
                          real_t cosalfr,
                          real_t sinalfr,
                          real_t c2d,
                          real_t c2sd,
                          __global real_t* sinmodf_d,
                          __global real_t* cosmodf_d,
                          real_t sindel,
                          real_t cosdel,
                          int Np,
                          int idet,
                          __constant Ampl_mod_coeff* amod_d)
{
  size_t idx = get_global_id(0);

  real_t c = cosalfr * cosmodf_d[idx] + sinalfr * sinmodf_d[idx];
  real_t s = sinalfr * cosmodf_d[idx] - cosalfr * sinmodf_d[idx];
  real_t c2s = 2.*c*c;
  real_t cs = c*s;

  aa_d[idx] =
    amod_d[idet].c1*(2. - c2d)*c2s +
    amod_d[idet].c2*(2. - c2d)*2.*cs +
    amod_d[idet].c3*c2sd*c +
    amod_d[idet].c4*c2sd*s -
    amod_d[idet].c1*(2. - c2d) +
    amod_d[idet].c5*c2d;
  
  bb_d[idx] =
    amod_d[idet].c6*sindel*c2s +
    amod_d[idet].c7*sindel*2.*cs +
    amod_d[idet].c8*cosdel*c +
    amod_d[idet].c9*cosdel*s -
    amod_d[idet].c6*sindel;
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void tshift_pmod_kern(real_t shft1,
                               real_t het0,
                               real3_t ns,
                               __global real_t* xDat_d,
                               __global complex_t* xa_d,
                               __global complex_t* xb_d,
                               __global real_t* shft_d,
                               __global real_t* shftf_d,
                               __global real_t* tshift_d,
                               __global real_t* aa_d,
                               __global real_t* bb_d,
                               __global real3_t* DetSSB_d,
                               real_t oms,
                               int N,
                               int nfft,
                               int interpftpad)
{
    size_t i = get_global_id(0);

    if (i < N)
    {
        real_t S = ns.x * DetSSB_d[i].x
                 + ns.y * DetSSB_d[i].y
                 + ns.z * DetSSB_d[i].z;
        shft_d[i] = S;
        shftf_d[i] = S - shft1;

        /* phase mod */
        // dlaczego - ?
        real_t phase = -het0*i - oms * S;
        real_t c = cos(phase), s = sin(phase);
        xa_d[i].x = xDat_d[i] * aa_d[i] * c;
        xa_d[i].y = xDat_d[i] * aa_d[i] * s;
        xb_d[i].x = xDat_d[i] * bb_d[i] * c;
        xb_d[i].y = xDat_d[i] * bb_d[i] * s;

        //calculate time positions for spline interpolation (recalculate instead of MEM_FENCE)
        tshift_d[i] = interpftpad * (i - /*shftf_d[i]*/(S - shft1));
    }
    else if (i < nfft)
    {
        xa_d[i].x = xa_d[i].y = xb_d[i].x = xb_d[i].y = 0.;
    }
}

/// <summary>Shifts frequencies and remove those over Nyquist.</summary>
///
__kernel void resample_postfft(__global complex_t *xa_d,
                               __global complex_t *xb_d,
                               int nfft,
                               int Ninterp,
                               int nyqst)
{
    size_t idx = get_global_id(0);

	int i = nyqst + Ninterp - nfft + idx;
    int j = nyqst + idx;

    xa_d[i].x = xa_d[j].x;
    xa_d[i].y = xa_d[j].y;
    xb_d[i].x = xb_d[j].x;
    xb_d[i].y = xb_d[j].y;

	mem_fence(CLK_GLOBAL_MEM_FENCE);

	xa_d[j].x = 0;
	xa_d[j].y = 0;
	xb_d[j].x = 0;
	xb_d[j].y = 0;
}

/// <summary>Computes sin and cos values and stores them in an array.</summary>
/// <remarks>Most likely a very bad idea. Results are used in modvir and should be computed there in place.</remarks>
///
__kernel void compute_sincosmodf(__global real_t* s,
                                 __global real_t* c,
                                 real_t omr,
                                 int N)
{
    size_t idx = get_global_id(0);

    if (idx < N)
    {
        s[idx] = sin(omr * idx);
        c[idx] = cos(omr * idx);
    }
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void computeB(__global complex_t* y,
                       __global complex_t* B,
                       int N)
{
    size_t idx = get_global_id(0);

    if (idx < N - 1)
    {
		B[idx] = 6 * (y[idx + 2] - 2 * y[idx + 1] + y[idx]);
    }
}

/// <summary>Multiplies the tridiagonal matrix specified by <c>{dl, d, du}</c> with dense vector <c>x</c>.</summary>
///
__kernel void tridiagMul(__global real_t* dl,
                         __global real_t* d,
                         __global real_t* du,
                         __global complex_t* x,
                         __global complex_t* y)
{
    size_t gid = get_global_id(0);
    size_t gsi = get_global_size(0);

	// Select 3 contributing values from x
	complex_t x1 = (gid == 0 ? (real_t)0 : x[gid - 1]),
	          x2 = x[gid],
			  x3 = (gid == gsi - 1 ? (real_t)0 : x[gid + 1]);


    y[gid] = dl[gid] * x1 +
              d[gid] * x2 +
             du[gid] * x3;
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void interpolate(__global real_t* new_x,
                          __global complex_t* new_y,
                          __global complex_t* z,
                          __global complex_t* y,
                          int N,
                          int new_N)
{
    size_t idx = get_global_id(0);
    real_t alpha = 1. / 6.;
    complex_t result;

    if (idx < new_N)
    {
        real_t x = new_x[idx];
    
        //get index of interval
        int i = floor(x);

        real_t dist1 = x - i;
        real_t dist2 = i + 1 - x;

        new_y[idx] = dist1*(z[i + 1]*alpha*(dist1*dist1 - 1) + y[i + 1]) + dist2*(z[i]*alpha*(dist2*dist2 - 1) + y[i]);
    }
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void phase_mod_1(__global complex_t* xa,
                          __global complex_t* xb,
                          __global complex_t* xar,
                          __global complex_t* xbr,
                          real_t het1,
                          real_t sgnlt1,
                          __global real_t* shft,
                          int N)
{
    size_t idx = get_global_id(0);

    if (idx < N)
    {
	    // _tmp1[0][i]
		//        |     aux->t2[i]
		//        |         |
		//        |         |    (double)(2*i)*ifo[n].sig.shft[i]
		//        |         |           |
		real_t tmp10i = idx*idx + 2*idx*shft[idx];

		real_t phase = idx*het1 + sgnlt1*tmp10i;
		complex_t exph = cbuild(cos(phase), -sin(phase));

		xa[idx] = cmulcc(xar[idx], exph);
		xb[idx] = cmulcc(xbr[idx], exph);
    }
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void phase_mod_2(__global complex_t* xa,
                          __global complex_t* xb,
                          __global complex_t* xar,
                          __global complex_t* xbr,
                          real_t het1,
                          real_t sgnlt1,
                          __global real_t* shft,
                          int N)
{
    size_t idx = get_global_id(0);

    if (idx < N)
    {
		// _tmp1[n][i]
		//        |     aux->t2[i]
		//        |         |
		//        |         |    (double)(2*i)*ifo[n].sig.shft[i]
		//        |         |           |
		real_t tmp1ni = idx*idx + 2*idx*shft[idx];

		real_t phase = idx*het1 + sgnlt1*tmp1ni;
		complex_t exph = cbuild(cos(phase), -sin(phase));

		xa[idx] += cmulcc(xar[idx], exph);
		xb[idx] += cmulcc(xbr[idx], exph);
    }
}

/// <summary>Compute F-statistics.</summary>
/// 
__kernel void compute_Fstat(__global complex_t* xa,
                            __global complex_t* xb,
                            __global real_t* F,
                            __constant real_t* maa_d,
                            __constant real_t* mbb_d,
                            int N)
{
  size_t i = get_global_id(0);

  complex_t xai = xa[i],
            xbi = xb[i];
  real_t maa = maa_d[0],
         mbb = mbb_d[0];

  F[i] = (pown(creal(xai), 2) + pown(cimag(xai), 2)) / maa +
         (pown(creal(xbi), 2) + pown(cimag(xbi), 2)) / mbb;
}

/// <summary>Compute F-statistics.</summary>
/// <precondition>lsi less than or equal to nav</precondition>
/// <precondition>lsi be a divisor of nav</precondition>
/// <precondition>lsi be an integer power of 2</precondition>
/// <precondition>nav be a divisor of gsi</precondition>
/// 
__kernel void normalize_Fstat_wg_reduce(__global real_t* F,
                                        __local real_t* shared,
                                        unsigned int nav)
{
    size_t lid = get_local_id(0),
           lsi = get_local_size(0),
           grp = get_group_id(0),
		   off = get_global_offset(0);
    
    // Load all of nav into local memory
    {   
        event_t copy = async_work_group_copy(shared,
                                             (F + off) + grp * nav,
                                             nav,
                                             0);   
        wait_group_events(1, &copy);
    }

    // Pass responsible for handling nav potentially having more elements than the work-group has threads
    //
    // ASSERT: lsi <= nav
    // ASSERT: nav % lsi == 0
    if (lsi != nav)
    {
        real_t mu = 0;

        for (size_t p = 0; p < nav; p += lsi) mu += shared[p + lid];

        shared[lid] = mu;
    }

    // Divide and conquer inside the work-group
    //
    // ASSERT: lsi is a power of 2
    for (size_t mid = lsi / 2; mid != 0; mid = mid >> 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < mid) shared[lid] += shared[mid + lid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    real_t mu = shared[0] / (2 * nav);

    // Normalize in global
    //
    // ASSERT: gsi % nav == 0
    for (size_t p = 0; p < nav; p += lsi)
    {
        F[off + grp * nav + p + lid] /= mu;

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
