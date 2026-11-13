// Human ventricular myocyte model of the O'Hara-Rudy family (ORd 2011, the
// base of ToR-ORd), with three cell types (endo / mid / epi) and two
// ToR-ORd-style extensions: L-type Ca current split between the junctional
// subspace (80 %) and the cytosol (20 %), and Ca-activated / background
// chloride currents.
//
// This file is shared by the OpenCL kernel (tor_ord.cl) and by the CPU
// reference in main.cpp, so both run exactly the same equations.
//
// It is a benchmark model with the structure and cost of ToR-ORd (40 state
// variables + V, ~15 currents, very different stiffness along the action
// potential); it is NOT the published ToR-ORd calibration. For scientific
// use, replace this file by the official ToR-ORd equations.
//
// Units: mV, ms, uA/uF, mM.

#ifndef TORORD_LIKE_H
#define TORORD_LIKE_H

#ifdef __OPENCL_VERSION__
    // Precision chosen at build time with -DREAL=float|double.
    #ifndef REAL
    #define REAL float
    #endif
    #if defined(USE_DOUBLE)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    typedef REAL real;
    #define TORORD_FN
#else
    #include <cmath>
    #ifndef REAL
    #define REAL double
    #endif
    typedef REAL real;
    #define TORORD_FN inline
    using std::exp;
    using std::fabs;
    using std::log;
    using std::pow;
    using std::sqrt;
#endif

// ---------------------------------------------------------------------------
// State variables (V is kept outside, in its own field)
// ---------------------------------------------------------------------------
#define S_NAI    0
#define S_NASS   1
#define S_KI     2
#define S_KSS    3
#define S_CAI    4
#define S_CASS   5
#define S_CANSR  6
#define S_CAJSR  7
#define S_M      8
#define S_HF     9
#define S_HS     10
#define S_J      11
#define S_HSP    12
#define S_JP     13
#define S_ML     14
#define S_HL     15
#define S_HLP    16
#define S_A      17
#define S_IF     18
#define S_IS     19
#define S_AP     20
#define S_IFP    21
#define S_ISP    22
#define S_D      23
#define S_FF     24
#define S_FS     25
#define S_FCAF   26
#define S_FCAS   27
#define S_JCA    28
#define S_NCA    29
#define S_FFP    30
#define S_FCAFP  31
#define S_XRF    32
#define S_XRS    33
#define S_XS1    34
#define S_XS2    35
#define S_XK1    36
#define S_JRELNP 37
#define S_JRELP  38
#define S_CAMKT  39
#define TORORD_NSTATES 40

#define CELL_ENDO 0
#define CELL_EPI  1
#define CELL_MID  2

#define TORORD_V0 (-87.5)

// Initial values (ORd defaults; a few beats settle them).
TORORD_FN void torord_initial_state(real* s) {
    s[S_NAI] = 7.0;     s[S_NASS] = 7.0;   s[S_KI] = 145.0;   s[S_KSS] = 145.0;
    s[S_CAI] = 1.0e-4;  s[S_CASS] = 1.0e-4; s[S_CANSR] = 1.2; s[S_CAJSR] = 1.2;
    s[S_M] = 0.0;   s[S_HF] = 1.0;  s[S_HS] = 1.0;  s[S_J] = 1.0;  s[S_HSP] = 1.0;  s[S_JP] = 1.0;
    s[S_ML] = 0.0;  s[S_HL] = 1.0;  s[S_HLP] = 1.0;
    s[S_A] = 0.0;   s[S_IF] = 1.0;  s[S_IS] = 1.0;  s[S_AP] = 0.0; s[S_IFP] = 1.0; s[S_ISP] = 1.0;
    s[S_D] = 0.0;   s[S_FF] = 1.0;  s[S_FS] = 1.0;  s[S_FCAF] = 1.0; s[S_FCAS] = 1.0;
    s[S_JCA] = 1.0; s[S_NCA] = 0.0; s[S_FFP] = 1.0; s[S_FCAFP] = 1.0;
    s[S_XRF] = 0.0; s[S_XRS] = 0.0; s[S_XS1] = 0.0; s[S_XS2] = 0.0; s[S_XK1] = 1.0;
    s[S_JRELNP] = 0.0; s[S_JRELP] = 0.0; s[S_CAMKT] = 0.0;
}

// Gate states use Rush-Larsen (exponential) integration; the rest use
// forward Euler.
TORORD_FN int torord_is_gate(int k) {
    return (k >= S_M && k <= S_XK1 && k != S_NCA) || k == S_JRELNP || k == S_JRELP;
}

// v - x with protection against the 0/0 of the GHK flux at v = 0.
TORORD_FN real ghk_v(real v) { return (fabs(v) < (real)1.0e-4) ? (real)1.0e-4 : v; }

// Computes the ionic current (dv = -Iion - Istim) and, for every state,
// either (inf, tau) for gates or the time derivative (in rate[]).
TORORD_FN real torord_rates(real v, const real* s, int celltype, real istim, real* inf, real* tau, real* rate) {
    // Physical constants and geometry
    const real nao = 140.0, cao = 1.8, ko = 5.4;
    const real cli = 24.0, clo = 150.0;
    const real R = 8314.0, T = 310.0, F = 96485.0;
    const real L = 0.01, rad = 0.0011;
    const real vcell = 1000.0 * 3.14 * rad * rad * L;
    const real Ageo = 2.0 * 3.14 * rad * rad + 2.0 * 3.14 * rad * L;
    const real Acap = 2.0 * Ageo;
    const real vmyo = 0.68 * vcell, vnsr = 0.0552 * vcell, vjsr = 0.0048 * vcell, vss = 0.02 * vcell;
    const real RTF = R * T / F;

    const real nai = s[S_NAI], nass = s[S_NASS], ki = s[S_KI], kss = s[S_KSS];
    const real cai = s[S_CAI], cass = s[S_CASS], cansr = s[S_CANSR], cajsr = s[S_CAJSR];

    // Cell-type scalings
    const int epi = (celltype == CELL_EPI), mid = (celltype == CELL_MID);

    // CaMK
    const real KmCaMK = 0.15, aCaMK = 0.05, bCaMK = 0.00068, CaMKo = 0.05, KmCaM = 0.0015;
    const real CaMKb = CaMKo * (1.0 - s[S_CAMKT]) / (1.0 + KmCaM / cass);
    const real CaMKa = CaMKb + s[S_CAMKT];
    rate[S_CAMKT] = aCaMK * CaMKb * (CaMKb + s[S_CAMKT]) - bCaMK * s[S_CAMKT];
    const real fCaMKp = 1.0 / (1.0 + KmCaMK / CaMKa);

    // Reversal potentials
    const real ENa = RTF * log(nao / nai);
    const real EK = RTF * log(ko / ki);
    const real PKNa = 0.01833;
    const real EKs = RTF * log((ko + PKNa * nao) / (ki + PKNa * nai));
    const real ECl = RTF * log(cli / clo);
    const real vg = ghk_v(v);
    const real vffrt = vg * F * F / (R * T);
    const real vfrt = vg * F / (R * T);

    // INa
    inf[S_M] = 1.0 / (1.0 + exp(-(v + 39.57) / 9.871));
    tau[S_M] = 1.0 / (6.765 * exp((v + 11.64) / 34.77) + 8.552 * exp(-(v + 77.42) / 5.955));
    const real hss = 1.0 / (1.0 + exp((v + 82.90) / 6.086));
    inf[S_HF] = hss;
    tau[S_HF] = 1.0 / (1.432e-5 * exp(-(v + 1.196) / 6.285) + 6.149 * exp((v + 0.5096) / 20.27));
    inf[S_HS] = hss;
    tau[S_HS] = 1.0 / (0.009794 * exp(-(v + 17.95) / 28.05) + 0.3343 * exp((v + 5.730) / 56.66));
    inf[S_J] = hss;
    tau[S_J] = 2.038 + 1.0 / (0.02136 * exp(-(v + 100.6) / 8.281) + 0.3052 * exp((v + 0.9941) / 38.45));
    inf[S_HSP] = 1.0 / (1.0 + exp((v + 89.1) / 6.086));
    tau[S_HSP] = 3.0 * tau[S_HS];
    inf[S_JP] = hss;
    tau[S_JP] = 1.46 * tau[S_J];
    const real Ahf = 0.99, Ahs = 1.0 - Ahf;
    const real h = Ahf * s[S_HF] + Ahs * s[S_HS];
    const real hp = Ahf * s[S_HF] + Ahs * s[S_HSP];
    const real GNa = 75.0;
    const real m3 = s[S_M] * s[S_M] * s[S_M];
    const real INa = GNa * (v - ENa) * m3 * ((1.0 - fCaMKp) * h * s[S_J] + fCaMKp * hp * s[S_JP]);

    // INaL
    inf[S_ML] = 1.0 / (1.0 + exp(-(v + 42.85) / 5.264));
    tau[S_ML] = tau[S_M];
    inf[S_HL] = 1.0 / (1.0 + exp((v + 87.61) / 7.488));
    tau[S_HL] = 200.0;
    inf[S_HLP] = 1.0 / (1.0 + exp((v + 93.81) / 7.488));
    tau[S_HLP] = 3.0 * 200.0;
    const real GNaL = epi ? 0.0075 * 0.6 : 0.0075;
    const real INaL = GNaL * (v - ENa) * s[S_ML] * ((1.0 - fCaMKp) * s[S_HL] + fCaMKp * s[S_HLP]);

    // Ito
    inf[S_A] = 1.0 / (1.0 + exp(-(v - 14.34) / 14.82));
    tau[S_A] = 1.0515 / (1.0 / (1.2089 * (1.0 + exp(-(v - 18.4099) / 29.3814))) +
                         3.5 / (1.0 + exp((v + 100.0) / 29.3814)));
    const real iss = 1.0 / (1.0 + exp((v + 43.94) / 5.711));
    const real delta_epi = epi ? 1.0 - (0.95 / (1.0 + exp((v + 70.0) / 5.0))) : 1.0;
    const real tiF = delta_epi * (4.562 + 1.0 / (0.3933 * exp(-(v + 100.0) / 100.0) + 0.08004 * exp((v + 50.0) / 16.59)));
    const real tiS = delta_epi * (23.62 + 1.0 / (0.001416 * exp(-(v + 96.52) / 59.05) + 1.780e-8 * exp((v + 114.1) / 8.079)));
    inf[S_IF] = iss; tau[S_IF] = tiF;
    inf[S_IS] = iss; tau[S_IS] = tiS;
    const real AiF = 1.0 / (1.0 + exp((v - 213.6) / 151.2)), AiS = 1.0 - AiF;
    inf[S_AP] = 1.0 / (1.0 + exp(-(v - 24.34) / 14.82));
    tau[S_AP] = tau[S_A];
    const real dti_develop = 1.354 + 1.0e-4 / (exp((v - 167.4) / 15.89) + exp(-(v - 12.23) / 0.2154));
    const real dti_recover = 1.0 - 0.5 / (1.0 + exp((v + 70.0) / 20.0));
    inf[S_IFP] = iss; tau[S_IFP] = dti_develop * dti_recover * tiF;
    inf[S_ISP] = iss; tau[S_ISP] = dti_develop * dti_recover * tiS;
    const real i_gate = AiF * s[S_IF] + AiS * s[S_IS];
    const real ip = AiF * s[S_IFP] + AiS * s[S_ISP];
    const real Gto = epi ? 0.02 * 4.0 : (mid ? 0.02 * 4.0 : 0.02);
    const real Ito = Gto * (v - EK) * ((1.0 - fCaMKp) * s[S_A] * i_gate + fCaMKp * s[S_AP] * ip);

    // ICaL (gates)
    inf[S_D] = 1.0 / (1.0 + exp(-(v + 3.940) / 4.230));
    tau[S_D] = 0.6 + 1.0 / (exp(-0.05 * (v + 6.0)) + exp(0.09 * (v + 14.0)));
    const real fss = 1.0 / (1.0 + exp((v + 19.58) / 3.696));
    inf[S_FF] = fss;
    tau[S_FF] = 7.0 + 1.0 / (0.0045 * exp(-(v + 20.0) / 10.0) + 0.0045 * exp((v + 20.0) / 10.0));
    inf[S_FS] = fss;
    tau[S_FS] = 1000.0 + 1.0 / (0.000035 * exp(-(v + 5.0) / 4.0) + 0.000035 * exp((v + 5.0) / 6.0));
    const real Aff = 0.6, Afs = 1.0 - Aff;
    const real f = Aff * s[S_FF] + Afs * s[S_FS];
    inf[S_FCAF] = fss;
    tau[S_FCAF] = 7.0 + 1.0 / (0.04 * exp(-(v - 4.0) / 7.0) + 0.04 * exp((v - 4.0) / 7.0));
    inf[S_FCAS] = fss;
    tau[S_FCAS] = 100.0 + 1.0 / (0.00012 * exp(-v / 3.0) + 0.00012 * exp(v / 7.0));
    const real Afcaf = 0.3 + 0.6 / (1.0 + exp((v - 10.0) / 10.0)), Afcas = 1.0 - Afcaf;
    const real fca = Afcaf * s[S_FCAF] + Afcas * s[S_FCAS];
    inf[S_JCA] = fss; tau[S_JCA] = 75.0;
    inf[S_FFP] = fss; tau[S_FFP] = 2.5 * tau[S_FF];
    const real fp = Aff * s[S_FFP] + Afs * s[S_FS];
    inf[S_FCAFP] = fss; tau[S_FCAFP] = 2.5 * tau[S_FCAF];
    const real fcap = Afcaf * s[S_FCAFP] + Afcas * s[S_FCAS];
    const real Kmn = 0.002, k2n = 1000.0;
    const real km2n = s[S_JCA] * 1.0;
    const real kmn_c = 1.0 + Kmn / cass;
    const real anca = 1.0 / (k2n / km2n + kmn_c * kmn_c * kmn_c * kmn_c);
    rate[S_NCA] = anca * k2n - s[S_NCA] * km2n;

    // ICaL fluxes: 80 % into the subspace, 20 % into the cytosol.
    const real e2 = exp(2.0 * vfrt), e1 = exp(vfrt);
    const real PhiCaL_ss = 4.0 * vffrt * (cass * e2 - 0.341 * cao) / (e2 - 1.0);
    const real PhiCaL_i = 4.0 * vffrt * (cai * e2 - 0.341 * cao) / (e2 - 1.0);
    const real PhiCaNa_ss = vffrt * (0.75 * nass * e1 - 0.75 * nao) / (e1 - 1.0);
    const real PhiCaNa_i = vffrt * (0.75 * nai * e1 - 0.75 * nao) / (e1 - 1.0);
    const real PhiCaK_ss = vffrt * (0.75 * kss * e1 - 0.75 * ko) / (e1 - 1.0);
    const real PhiCaK_i = vffrt * (0.75 * ki * e1 - 0.75 * ko) / (e1 - 1.0);
    const real PCa = (epi ? 1.2 : (mid ? 2.5 : 1.0)) * 0.0001;
    const real PCap = 1.1 * PCa;
    const real open_np = s[S_D] * (f * (1.0 - s[S_NCA]) + s[S_JCA] * fca * s[S_NCA]);
    const real open_p = s[S_D] * (fp * (1.0 - s[S_NCA]) + s[S_JCA] * fcap * s[S_NCA]);
    const real gCa = (1.0 - fCaMKp) * PCa * open_np + fCaMKp * PCap * open_p;
    const real fss_frac = 0.8;
    const real ICaL_ss = fss_frac * gCa * PhiCaL_ss;
    const real ICaL_i = (1.0 - fss_frac) * gCa * PhiCaL_i;
    const real ICaNa_ss = fss_frac * 0.00125 * gCa * PhiCaNa_ss;
    const real ICaNa_i = (1.0 - fss_frac) * 0.00125 * gCa * PhiCaNa_i;
    const real ICaK_ss = fss_frac * 3.574e-4 * gCa * PhiCaK_ss;
    const real ICaK_i = (1.0 - fss_frac) * 3.574e-4 * gCa * PhiCaK_i;
    const real ICaL = ICaL_ss + ICaL_i;

    // IKr
    const real xrss = 1.0 / (1.0 + exp(-(v + 8.337) / 6.789));
    inf[S_XRF] = xrss;
    tau[S_XRF] = 12.98 + 1.0 / (0.3652 * exp((v - 31.66) / 3.869) + 4.123e-5 * exp(-(v - 47.78) / 20.38));
    inf[S_XRS] = xrss;
    tau[S_XRS] = 1.865 + 1.0 / (0.06629 * exp((v - 34.70) / 7.355) + 1.128e-5 * exp(-(v - 29.74) / 25.94));
    const real Axrf = 1.0 / (1.0 + exp((v + 54.81) / 38.21)), Axrs = 1.0 - Axrf;
    const real xr = Axrf * s[S_XRF] + Axrs * s[S_XRS];
    const real rkr = 1.0 / (1.0 + exp((v + 55.0) / 75.0)) * 1.0 / (1.0 + exp((v - 10.0) / 30.0));
    const real GKr = (epi ? 1.3 : (mid ? 0.8 : 1.0)) * 0.046;
    const real IKr = GKr * sqrt(ko / 5.4) * xr * rkr * (v - EK);

    // IKs
    const real xs1ss = 1.0 / (1.0 + exp(-(v + 11.60) / 8.932));
    inf[S_XS1] = xs1ss;
    tau[S_XS1] = 817.3 + 1.0 / (2.326e-4 * exp((v + 48.28) / 17.80) + 0.001292 * exp(-(v + 210.0) / 230.0));
    inf[S_XS2] = xs1ss;
    tau[S_XS2] = 1.0 / (0.01 * exp((v - 50.0) / 20.0) + 0.0193 * exp(-(v + 66.54) / 31.0));
    const real KsCa = 1.0 + 0.6 / (1.0 + pow((real)3.8e-5 / cai, (real)1.4));
    const real GKs = (epi ? 1.4 : 1.0) * 0.0034;
    const real IKs = GKs * KsCa * s[S_XS1] * s[S_XS2] * (v - EKs);

    // IK1
    inf[S_XK1] = 1.0 / (1.0 + exp(-(v + 2.5538 * ko + 144.59) / (1.5692 * ko + 3.8115)));
    tau[S_XK1] = 122.2 / (exp(-(v + 127.2) / 20.36) + exp((v + 236.8) / 69.33));
    const real rk1 = 1.0 / (1.0 + exp((v + 105.8 - 2.6 * ko) / 9.493));
    const real GK1 = (epi ? 1.2 : (mid ? 1.3 : 1.0)) * 0.1908;
    const real IK1 = GK1 * sqrt(ko) * rk1 * s[S_XK1] * (v - EK);

    // INaCa (cytosol and subspace)
    const real kna1 = 15.0, kna2 = 5.0, kna3 = 88.12, kasymm = 12.5;
    const real wna = 6.0e4, wca = 6.0e4, wnaca = 5.0e3, kcaon = 1.5e6, kcaoff = 5.0e3;
    const real qna = 0.5224, qca = 0.1670;
    const real hca = exp(qca * v / RTF), hna = exp(qna * v / RTF);
    const real Gncx = (epi ? 1.1 : (mid ? 1.4 : 1.0)) * 0.0008;
    const real KmCaAct = 150.0e-6;
    real INaCa_i = 0.0, INaCa_ss = 0.0;
    for (int compartment = 0; compartment < 2; ++compartment) {
        const real na = compartment == 0 ? nai : nass;
        const real ca = compartment == 0 ? cai : cass;
        const real h1 = 1.0 + na / kna3 * (1.0 + hna);
        const real h2 = (na * hna) / (kna3 * h1);
        const real h3 = 1.0 / h1;
        const real h4 = 1.0 + na / kna1 * (1.0 + na / kna2);
        const real h5 = na * na / (h4 * kna1 * kna2);
        const real h6 = 1.0 / h4;
        const real h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna);
        const real h8 = nao / (kna3 * hna * h7);
        const real h9 = 1.0 / h7;
        const real h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2);
        const real h11 = nao * nao / (h10 * kna1 * kna2);
        const real h12 = 1.0 / h10;
        const real k1 = h12 * cao * kcaon, k2 = kcaoff;
        const real k3p = h9 * wca, k3pp = h8 * wnaca, k3 = k3p + k3pp;
        const real k4p = h3 * wca / hca, k4pp = h2 * wnaca, k4 = k4p + k4pp;
        const real k5 = kcaoff, k6 = h6 * ca * kcaon;
        const real k7 = h5 * h2 * wna, k8 = h8 * h11 * wna;
        const real x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3);
        const real x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8);
        const real x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3);
        const real x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8);
        const real sum = x1 + x2 + x3 + x4;
        const real E1 = x1 / sum, E2 = x2 / sum, E3 = x3 / sum, E4 = x4 / sum;
        const real act = KmCaAct / ca;
        const real allo = 1.0 / (1.0 + act * act);
        const real JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp;
        const real JncxCa = E2 * k2 - E1 * k1;
        const real current = Gncx * allo * (JncxNa + 2.0 * JncxCa);
        if (compartment == 0) INaCa_i = 0.8 * current;
        else INaCa_ss = 0.2 * current;
    }

    // INaK
    const real k1p = 949.5, k1m = 182.4, k2p = 687.2, k2m = 39.4;
    const real k3pk = 1899.0, k3m = 79300.0, k4pk = 639.0, k4m = 40.0;
    const real Knai = 9.073 * exp((-0.1550) * v / (3.0 * RTF));
    const real Knao = 27.78 * exp((1.0 + 0.1550) * v / (3.0 * RTF));
    const real Kki = 0.5, Kko = 0.3582, MgADP = 0.05, MgATP = 9.8, Kmgatp = 1.698e-7;
    const real H = 1.0e-7, eP = 4.2, Khp = 1.698e-7, Knap = 224.0, Kxkur = 292.0;
    const real P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur);
    const real ni = nai / Knai, no = nao / Knao, kin = ki / Kki, kon = ko / Kko;
    const real ni3 = ni * ni * ni, no3 = no * no * no;
    const real di = (1.0 + ni) * (1.0 + ni) * (1.0 + ni) + (1.0 + kin) * (1.0 + kin) - 1.0;
    const real dout = (1.0 + no) * (1.0 + no) * (1.0 + no) + (1.0 + kon) * (1.0 + kon) - 1.0;
    const real a1 = k1p * ni3 / di, b1 = k1m * MgADP;
    const real a2 = k2p, b2 = k2m * no3 / dout;
    const real a3 = k3pk * kon * kon / dout, b3 = k3m * P * H / (1.0 + MgATP / Kmgatp);
    const real a4 = k4pk * MgATP / Kmgatp / (1.0 + MgATP / Kmgatp), b4 = k4m * kin * kin / di;
    const real y1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2;
    const real y2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4;
    const real y3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1;
    const real y4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1;
    const real ysum = y1 + y2 + y3 + y4;
    const real JnakNa = 3.0 * (y1 / ysum * a3 - y2 / ysum * b3);
    const real JnakK = 2.0 * (y4 / ysum * b1 - y3 / ysum * a1);
    const real Pnak = (epi ? 0.9 : (mid ? 0.7 : 1.0)) * 30.0;
    const real INaK = Pnak * (JnakNa + JnakK);

    // Background currents, pump, chloride
    const real xkb = 1.0 / (1.0 + exp(-(v - 14.48) / 18.34));
    const real IKb = (epi ? 0.6 : 1.0) * 0.003 * xkb * (v - EK);
    const real INab = 3.75e-10 * vffrt * (nai * e1 - nao) / (e1 - 1.0);
    const real ICab = 2.5e-8 * 4.0 * vffrt * (cai * e2 - 0.341 * cao) / (e2 - 1.0);
    const real IpCa = 0.0005 * cai / (0.0005 + cai);
    const real GClCa = 0.2843, KdClCa = 0.1;
    const real IClCa = 0.2 * GClCa / (1.0 + KdClCa / cass) * (v - ECl) +
                       0.8 * GClCa / (1.0 + KdClCa / cai) * (v - ECl);
    const real IClb = 1.98e-3 * (v - ECl);

    // Diffusion between compartments
    const real JdiffNa = (nass - nai) / 2.0;
    const real JdiffK = (kss - ki) / 2.0;
    const real Jdiff = (cass - cai) / 0.2;

    // SR release (driven by the subspace ICaL)
    const real bt = 4.75, a_rel = 0.5 * bt;
    const real sr8 = pow((real)1.5 / cajsr, (real)8.0);
    const real rel_scale = mid ? 1.7 : 1.0;
    inf[S_JRELNP] = rel_scale * a_rel * (-ICaL_ss) / (1.0 + sr8);
    real tau_rel = bt / (1.0 + 0.0123 / cajsr);
    tau[S_JRELNP] = tau_rel < 0.001 ? (real)0.001 : tau_rel;
    const real btp = 1.25 * bt, a_relp = 0.5 * btp;
    inf[S_JRELP] = rel_scale * a_relp * (-ICaL_ss) / (1.0 + sr8);
    real tau_relp = btp / (1.0 + 0.0123 / cajsr);
    tau[S_JRELP] = tau_relp < 0.001 ? (real)0.001 : tau_relp;
    const real Jrel = (1.0 - fCaMKp) * s[S_JRELNP] + fCaMKp * s[S_JRELP];

    // SR uptake and transfer
    const real up_scale = epi ? 1.3 : 1.0;
    const real Jupnp = up_scale * 0.004375 * cai / (cai + 0.00092);
    const real Jupp = up_scale * 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017);
    const real Jleak = 0.0039375 * cansr / 15.0;
    const real Jup = (1.0 - fCaMKp) * Jupnp + fCaMKp * Jupp - Jleak;
    const real Jtr = (cansr - cajsr) / 100.0;

    // Concentrations
    rate[S_NAI] = -(INa + INaL + 3.0 * INaCa_i + ICaNa_i + 3.0 * INaK + INab) * Acap / (F * vmyo) +
                  JdiffNa * vss / vmyo;
    rate[S_NASS] = -(ICaNa_ss + 3.0 * INaCa_ss) * Acap / (F * vss) - JdiffNa;
    rate[S_KI] = -(Ito + IKr + IKs + IK1 + IKb + ICaK_i + istim - 2.0 * INaK) * Acap / (F * vmyo) +
                 JdiffK * vss / vmyo;
    rate[S_KSS] = -(ICaK_ss) * Acap / (F * vss) - JdiffK;

    const real cmdnmax = epi ? 0.05 * 1.3 : 0.05;
    const real kmcmdn = 0.00238, trpnmax = 0.07, kmtrpn = 0.0005;
    const real BSRmax = 0.047, KmBSR = 0.00087, BSLmax = 1.124, KmBSL = 0.0087;
    const real csqnmax = 10.0, kmcsqn = 0.8;
    const real Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn / ((kmcmdn + cai) * (kmcmdn + cai)) +
                             trpnmax * kmtrpn / ((kmtrpn + cai) * (kmtrpn + cai)));
    rate[S_CAI] = Bcai * (-(ICaL_i + IpCa + ICab - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo) -
                          Jup * vnsr / vmyo + Jdiff * vss / vmyo);
    const real Bcass = 1.0 / (1.0 + BSRmax * KmBSR / ((KmBSR + cass) * (KmBSR + cass)) +
                              BSLmax * KmBSL / ((KmBSL + cass) * (KmBSL + cass)));
    rate[S_CASS] = Bcass * (-(ICaL_ss - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss) + Jrel * vjsr / vss - Jdiff);
    rate[S_CANSR] = Jup - Jtr * vjsr / vnsr;
    const real Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / ((kmcsqn + cajsr) * (kmcsqn + cajsr)));
    rate[S_CAJSR] = Bcajsr * (Jtr - Jrel);

    // Total ionic current (uA/uF)
    return INa + INaL + Ito + ICaL + ICaNa_ss + ICaNa_i + ICaK_ss + ICaK_i + IKr + IKs + IK1 +
           INaCa_i + INaCa_ss + INaK + INab + IKb + IpCa + ICab + IClCa + IClb;
}

// Advances one cell by `dt`, split into equal substeps. The number of
// substeps follows the stiffness of the current phase of the action
// potential, limiting per substep both the change of V (`dv_max`, mV) and the
// relative change of cytosolic Ca2+ (`ca_tol`):
//   upstroke          ~ hundreds of substeps (dV/dt ~ 250 mV/ms)
//   Ca2+ release      ~ tens (Ca_i rises ~10x in a few ms)
//   plateau/recovery  a few
//   rest              `min_substeps`
// Returns the number of substeps used.
TORORD_FN int torord_advance(real* v, real* s, int celltype, real istim, real dt, real dv_max, real ca_tol,
                            int min_substeps, int max_substeps) {
    real inf[TORORD_NSTATES], tau[TORORD_NSTATES], rate[TORORD_NSTATES];

    const real dvdt = -(torord_rates(*v, s, celltype, istim, inf, tau, rate) + istim);
    const real by_v = fabs(dvdt) * dt / dv_max;
    const real by_ca = fabs(rate[S_CAI]) * dt / (ca_tol * s[S_CAI]);
    int n = (int)ceil(by_v > by_ca ? by_v : by_ca);
    if (n < min_substeps) n = min_substeps;
    if (n > max_substeps) n = max_substeps;
    const real h = dt / (real)n;

    for (int step = 0; step < n; ++step) {
        const real iion = torord_rates(*v, s, celltype, istim, inf, tau, rate);
        for (int k = 0; k < TORORD_NSTATES; ++k) {
            if (torord_is_gate(k)) s[k] = inf[k] + (s[k] - inf[k]) * exp(-h / tau[k]);
            else s[k] += h * rate[k];
        }
        *v += -h * (iion + istim);
    }
    return n;
}

#endif // TORORD_LIKE_H
