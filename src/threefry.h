/* moprobit: MCMC Estimation of Multivariate Ordinal Probit Models
 * Copyright 2019 Michael J. Culbertson <mculbertson@edanalytics.org>
 *
 *  moprobit is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  moprobit is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  To read a copy of the GNU General Public License, see
 *  <http://www.gnu.org/licenses/>.
 */

/* Threefry2x64_20 algorithm
 *
 * Counter-based pseudo-random number generator
 *   taking a key of 128-bits (two 64-bit words) and a counter of 128-bits
 *   and producing a uniformly distributed 128-bit unsigned integer
 *
 * John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw (2011)
 * "Parallel random numbers: As easy as 1, 2, 3"
 * SC'11: Proceedings of the 2011 International Conference for High
 *   Performance Computing, Networking, Storage and Anaysis
 * Article No. 16
 * doi:10.1145/2063384.2063405
 *
 * Niels Ferguson, Stefan Lucks, Bruce Schneier, Doug Whiting, Mihir Bellare
 *   Tadayoshi Kohno, Jon Callas, and Jesse Walker (2010)
 * "The Skein hash function family" (version 1.3)
 * https://www.schneier.com/academic/paperfiles/skein1.3.pdf
 *
 * The Threefry2x64 algorithm is a simplification of the Threefish algorithm
 * used in the Skein hash function family. Threefry2x64 uses only 128 bits,
 * and thus needs no permutation. It also skips the "tweak". The counter is
 * taken as the "plaintext" to initialize the algorithm.
 *
 * Let k[0] and k[1] be the two 64-bit words of the key,
 *     p[0] and p[1] be the two 64-bit words of the counter,
 *     x[0] and x[1] be the two 64-bit words of the output.
 *
 * The key is first extended by an additional 64-bit word:
 *   k[2] = 0x1BD11BDAA9FC1A22 ^ k[0] ^ k[1]
 *
 * The output is initialized by the sum of the counter and key:
 *   x[0] = p[0] + k[0]
 *   x[1] = x[1] + k[1]
 *
 * The algorithm proceeds in a series of "mix" operations, consisting of
 *   an add, a rotation, and an exclusive-or:
 *   x[0] += x[1]
 *   Rotate x[1] left by R_i bits, where i is the round number
 *   x[1] ^= x[0]
 * After every four rounds, a rotation of the extended key is added,
 *   along with an integer:
 *   x[0] += k[(i/4) mod 3]
 *   x[1] += k[(i/4+1) mod 3] + i/4
 * So, k[1] and k[2] are added after the 4th round,
 *     k[2] and k[0] after the 8th round,
 *     k[0] and k[1] after the 12th round, etc.
 *
 * The number of bits to rotate in each round, from Salmon et al.:
 *   R_i = 16, 42, 12, 31, 16, 32, 24, 21
 * where these values are recycled after 8 rounds.
 *
 * Salmon et al. indicated that 13 rounds are sufficient to pass BigCrush,
 * and recommend 20 rounds for "safety margin".
 *
 * To convert an unsigned 64-bit integer to a double in [0, 1),
 *   mask to the 53 most significant digits and multiply by 2^-64:
 *   1./((double)(0xffffffffffffffff)+1.) * (x ^ 0x7ff)
 * To exclude 0, add half of the initial factor after the multiply.
 *
 */

#ifndef MOPROBIT_THREEFRY_H
#define MOPROBIT_THREEFRY_H

#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <Rcpp.h>

#define MOPROBIT_DIST_RAND_EXCLUDE_0 1

namespace moprobit { namespace dist {

class Key
{
  static const uint64_t SKEIN_PARITY_KEY = 0x1BD11BDAA9FC1A22;
  
public:
  
  const uint64_t k0, k1, kEXT;
  
  Key() : k0( time(NULL) ), k1( getpid() ), kEXT(SKEIN_PARITY_KEY ^ k0 ^ k1) {}
  Key(uint64_t seed) : k0(seed), k1(0), kEXT(SKEIN_PARITY_KEY ^ seed) {}
  Key(uint64_t a, uint64_t b) : k0(a), k1(b), kEXT(SKEIN_PARITY_KEY ^ a ^ b) {}
  
  static Key from_R()
  {
    const uint64_t k0 = ((uint64_t)(R::runif(0, 1) * 4294967296) << 32) | (uint64_t)(R::runif(0, 1) * 4294967296);
    const uint64_t k1 = ((uint64_t)(R::runif(0, 1) * 4294967296) << 32) | (uint64_t)(R::runif(0, 1) * 4294967296);
    return Key(k0, k1);
  }

  static Key from_stream(std::istream &is)
  {
    uint64_t a, b;
    const std::ios_base::fmtflags fmt = is.flags();
    is >> std::hex >> a >> b;
    is.flags(fmt);
    return Key(a, b);
  }
  
};

class Counter
{
public:
  
  uint64_t i0, i1;
  
  Counter() : i0(0), i1(0) {}
  Counter(uint64_t context) : i0(context), i1(0) {}
  Counter(uint64_t context, uint64_t i) : i0(context), i1(i) {}

  Counter& operator++() { ++i1; return *this; } // prefix ++
  Counter& jump() { ++i0; i1 = 0; return *this; } // increment context
  
};

inline std::ostream& operator<< (std::ostream& os, const Key &k)
{
  const std::ios_base::fmtflags fmt = os.flags();
  const char fillChar = os.fill('0');
  os << std::hex << std::setw(16) << k.k0 << " " << std::setw(16) << k.k1;
  os.flags(fmt);
  os.fill(fillChar);
  return os;
}

inline std::ostream& operator<< (std::ostream& os, const Counter &n)
{
  const std::ios_base::fmtflags fmt = os.flags();
  const char fillChar = os.fill('0');
  os << std::hex << std::setw(16) << n.i0 << " " << std::setw(16) << n.i1;
  os.flags(fmt);
  os.fill(fillChar);
  return os;
}

inline std::istream& operator>> (std::istream& is, Counter &n)
{
  const std::ios_base::fmtflags fmt = is.flags();
  is >> std::hex >> n.i0 >> n.i1;
  is.flags(fmt);
  return is;
}

inline bool operator<  (const Counter &n, const uint64_t rhs) { return n.i1 <  rhs; }
inline bool operator<= (const Counter &n, const uint64_t rhs) { return n.i1 <= rhs; }
inline bool operator== (const Counter &n, const uint64_t rhs) { return n.i1 == rhs; }
inline bool operator>= (const Counter &n, const uint64_t rhs) { return n.i1 >= rhs; }
inline bool operator>  (const Counter &n, const uint64_t rhs) { return n.i1 >  rhs; }

inline bool operator<  (const uint64_t lhs, const Counter &n) { return lhs <  n.i1; }
inline bool operator<= (const uint64_t lhs, const Counter &n) { return lhs <= n.i1; }
inline bool operator== (const uint64_t lhs, const Counter &n) { return lhs == n.i1; }
inline bool operator>= (const uint64_t lhs, const Counter &n) { return lhs >= n.i1; }
inline bool operator>  (const uint64_t lhs, const Counter &n) { return lhs >  n.i1; }


inline uint64_t ROTATE(uint64_t x, int b) { return ((x) << (b)) | ((x) >> (64-(b))); }

inline Counter threefry2x64_20(const Counter &n, const Key &k)
{
  // Rotation schedule
  static const int R1 = 16;
  static const int R2 = 42;
  static const int R3 = 12;
  static const int R4 = 31;
  static const int R5 = 16;
  static const int R6 = 32;
  static const int R7 = 24;
  static const int R8 = 21;

  /* Initialize output */
  Counter x(n.i0 + k.k0, n.i1 + k.k1);

  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R1) ^ x.i0;  /* Round  1 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R2) ^ x.i0;  /* Round  2 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R3) ^ x.i0;  /* Round  3 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R4) ^ x.i0;  /* Round  4 */
  x.i0 += k.k1; x.i1 += k.kEXT + 1;
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R5) ^ x.i0;  /* Round  5 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R6) ^ x.i0;  /* Round  6 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R7) ^ x.i0;  /* Round  7 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R8) ^ x.i0;  /* Round  8 */
  x.i0 += k.kEXT; x.i1 += k.k0 + 2;
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R1) ^ x.i0;  /* Round  9 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R2) ^ x.i0;  /* Round 10 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R3) ^ x.i0;  /* Round 11 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R4) ^ x.i0;  /* Round 12 */
  x.i0 += k.k0; x.i1 += k.k1 + 3;
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R5) ^ x.i0;  /* Round 13 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R6) ^ x.i0;  /* Round 14 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R7) ^ x.i0;  /* Round 15 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R8) ^ x.i0;  /* Round 16 */
  x.i0 += k.k1; x.i1 += k.kEXT + 4;
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R1) ^ x.i0;  /* Round 17 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R2) ^ x.i0;  /* Round 18 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R3) ^ x.i0;  /* Round 19 */
  x.i0 += x.i1; x.i1 = ROTATE(x.i1, R4) ^ x.i0;  /* Round 20 */
  x.i0 += k.kEXT; x.i1 += k.k0 + 5;
  
  return x;
}

inline void threefry2x64_20(const Counter &n, const Key &k, Counter &x)
{
  x = threefry2x64_20(n, k);
}

inline double U64_TO_D01(const uint64_t x)
{
  static const double D2_64 = (1.0/((double)(0xffffffffffffffff)+1.));
  static const double D2_65 = (0.5/((double)(0xffffffffffffffff)+1.));
#ifdef MOPROBIT_DIST_RAND_EXCLUDE_0
  return D2_64 * ((x) ^ 0x7ff) + D2_65;
#else
  return D2_64 * ((x) ^ 0x7ff) );
#endif
}


} }

#endif
