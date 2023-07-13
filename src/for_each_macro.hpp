#pragma once

/*
 * The PP_NARG macro evaluates to the number of arguments that have been
 * passed to it.
 *
 * Laurent Deniau, "__VA_NARG__," 17 January 2006, <comp.std.c> (29 November
 * 2007).
 */
#define PP_NARG(...) PP_NARG_(__VA_ARGS__, PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)

// see
// https://stackoverflow.com/questions/6707148/foreach-macro-on-macros-arguments

#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,  \
                 _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,   \
                 _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38,   \
                 _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50,   \
                 _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62,   \
                 _63, N, ...)                                                  \
  N

#define PP_RSEQ_N()                                                            \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45,  \
      44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27,  \
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,   \
      8, 7, 6, 5, 4, 3, 2, 1, 0

/* need extra level to force extra eval */
#define Paste(a, b) a##b
#define XPASTE(a, b) Paste(a, b)

/* APPLYXn variadic X-Macro by M Joshua Ryan      */
/* Free for all uses. Don't be a jerk.            */
/* I got bored after typing 15 of these.          */
/* You could keep going upto 64 (PPNARG's limit). */
#define APPLYX1(a) X(a)
#define APPLYX2(a, b) X(a) X(b)
#define APPLYX3(a, b, c) X(a) X(b) X(c)
#define APPLYX4(a, b, c, d) X(a) X(b) X(c) X(d)
#define APPLYX5(a, b, c, d, e) X(a) X(b) X(c) X(d) X(e)
#define APPLYX6(a, b, c, d, e, f) X(a) X(b) X(c) X(d) X(e) X(f)
#define APPLYX7(a, b, c, d, e, f, g) X(a) X(b) X(c) X(d) X(e) X(f) X(g)
#define APPLYX8(a, b, c, d, e, f, g, h) X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h)
#define APPLYX9(a, b, c, d, e, f, g, h, i)                                     \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i)
#define APPLYX10(a, b, c, d, e, f, g, h, i, j)                                 \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j)
#define APPLYX11(a, b, c, d, e, f, g, h, i, j, k)                              \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k)
#define APPLYX12(a, b, c, d, e, f, g, h, i, j, k, l)                           \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l)
#define APPLYX13(a, b, c, d, e, f, g, h, i, j, k, l, m)                        \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m)
#define APPLYX14(a, b, c, d, e, f, g, h, i, j, k, l, m, n)                     \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m) X(n)
#define APPLYX15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)                  \
  X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m) X(n) X(o)
#define APPLYX16(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,  \
                 s15, s16)                                                     \
  X(s1)                                                                        \
  X(s2)                                                                        \
  X(s3)                                                                        \
  X(s4)                                                                        \
  X(s5) X(s6) X(s7) X(s8) X(s9) X(s10) X(s11) X(s12) X(s13) X(s14) X(s15) X(s16)

#define APPLYX17(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,  \
                 s15, s16, s17)                                                \
  X(s1)                                                                        \
  X(s2)                                                                        \
  X(s3)                                                                        \
  X(s4)                                                                        \
  X(s5)                                                                        \
  X(s6)                                                                        \
  X(s7) X(s8) X(s9) X(s10) X(s11) X(s12) X(s13) X(s14) X(s15) X(s16) X(s17)

#define APPLYX_(M, ...) M(__VA_ARGS__)
#define APPLYXn(...) APPLYX_(XPASTE(APPLYX, PP_NARG(__VA_ARGS__)), __VA_ARGS__)

// Example usage:

// #define INOUTARGS_dbrrt                                                        \
//   do_optimization, cost_jump, best_cost_prune_factor, cost_weight, cost_bound, \
//       ao_rrt_rebuild_tree, ao_rrt

// #define X(a) loader.set(VAR_WITH_NAME(a));
//     APPLYXn(INOUTARGS_dbrrt);
// #undef X
