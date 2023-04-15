/*
 * g++ test/dgemmt.cpp -DMKL_ILP64 -I/usr/include/mkl -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o dgemmt-test-mkl
 * g++ test/dgemmt.cpp -DOPENBLAS -lopenblas -Lbuild/lib -I. -Ibuild -o dgemmt-test-openblas
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#ifdef OPENBLAS
#include <common.h>
extern "C" void BLASFUNC(dgemmt)(const char *, const char *, const char *,
                                 const blasint *, const blasint *,
                                 const double *, const double *,
                                 const blasint *, const double *,
                                 const blasint *, const double *, double *,
                                 const blasint *);
namespace blas {
void dgemm(const char *transA, const char *transB, const blasint *m,
           const blasint *n, const blasint *k, const double *alpha,
           const double *A, const blasint *ldA, const double *B,
           const blasint *ldB, const double *beta, double *C,
           const blasint *ldC) {
    ::BLASFUNC(dgemm)(const_cast<char *>(transA), const_cast<char *>(transB),
                      const_cast<blasint *>(m), const_cast<blasint *>(n),
                      const_cast<blasint *>(k), const_cast<double *>(alpha),
                      const_cast<double *>(A), const_cast<blasint *>(ldA),
                      const_cast<double *>(B), const_cast<blasint *>(ldB),
                      const_cast<double *>(beta), C,
                      const_cast<blasint *>(ldC));
}
void dgemmt(const char *uplo, const char *transA, const char *transB,
            const blasint *n, const blasint *k, const double *alpha,
            const double *A, const blasint *ldA, const double *B,
            const blasint *ldB, const double *beta, double *C,
            const blasint *ldC) {
    ::BLASFUNC(dgemmt)(uplo, transA, transB, n, k, alpha, A, ldA, B, ldB, beta,
                       C, ldC);
}
} // namespace blas
#else
#include <mkl_blas.h>
typedef MKL_INT blasint;
namespace blas {
using ::dgemm;
using ::dgemmt;
} // namespace blas
#endif

struct Matrix {
    std::vector<double> storage;
    blasint rows, cols;

    Matrix(blasint r, blasint c) : storage(r * c), rows(r), cols(c) {}

    double &operator()(blasint r, blasint c) { return storage[c * rows + r]; }
    const double &operator()(blasint r, blasint c) const {
        return storage[c * rows + r];
    }
    template <class F>
    void generate(const F &f) {
        std::generate(storage.begin(), storage.end(), f);
    }
    double *data() { return storage.data(); }
    const double *data() const { return storage.data(); }
};

template <class Rng>
blasint do_test(char uplo, char transA, char transB, blasint n, blasint k,
                blasint ldA, blasint ldB, blasint ldC, Rng &&rng) {
    std::cout << uplo << transA << transB << ' ' << n << ' ' << k << ' ' << ldA
              << ' ' << ldB << ' ' << ldC << std::endl;
    assert(uplo == 'U' || uplo == 'L');
    assert(transA == 'N' || transA == 'T');
    assert(transB == 'N' || transB == 'T');
    assert(n > 0);
    assert(k > 0);
    blasint m = n;
    auto alloc_mat = [](char trans, blasint ld, blasint r, blasint c) {
        if (trans == 'N') {
            assert(ld == 0 || ld >= r);
            ld = std::max(ld, r);
            return Matrix(ld, c);
        } else {
            assert(ld == 0 || ld >= c);
            ld = std::max(ld, c);
            return Matrix(ld, r);
        }
    };
    auto A = alloc_mat(transA, ldA, m, k);
    auto B = alloc_mat(transB, ldB, k, n);
    assert(ldC == 0 || ldC >= m);
    ldC = std::max(ldC, m);
    auto C = Matrix(ldC, n);

    A.generate(rng);
    B.generate(rng);
    C.generate(rng);
    double alpha = rng();
    double beta = rng();

    Matrix R1 = C;
    blas::dgemm(&transA, &transB, &m, &n, &k,                 //
                &alpha, A.data(), &A.rows, B.data(), &B.rows, //
                &beta, R1.data(), &R1.rows);

    Matrix R2 = C;
    blas::dgemmt(&uplo, &transA, &transB, &n, &k,              //
                 &alpha, A.data(), &A.rows, B.data(), &B.rows, //
                 &beta, R2.data(), &R2.rows);

    const double tolerance = 1e-12;
    blasint failures = 0;
    for (blasint c = 0; c < n; ++c) {
        blasint r_min = uplo == 'U' ? 0 : c;
        blasint r_max = uplo == 'U' ? c + 1 : m;
        for (blasint r = r_min; r < r_max; ++r) {
            if (std::abs(R1(r, c) - R2(r, c)) > tolerance) {
                std::cerr << '(' << r << ", " << c << ") " << R1(r, c)
                          << " != " << R2(r, c) << std::endl;
                if (++failures > 10) {
                    std::cout << "too many failures, giving up" << std::endl;
                    return failures;
                };
            }
        }
    }
    std::cout << (failures == 0 ? "pass" : "fail") << std::endl;
    return failures;
}

int main() {
    std::default_random_engine rng(12345);
    std::uniform_real_distribution<double> uniform(0, 1);
    auto uniform_random = [&] { return uniform(rng); };

    blasint failures = 0;
    for (blasint s : {1, 20}) {
        for (char transA : {'N', 'T'}) {
            for (char transB : {'N', 'T'}) {
                for (char uplo : {'U', 'L'}) {
                    failures += do_test(uplo, transA, transB, 13 * s, 17 * s, 0,
                                        0, 0, uniform_random);
                    failures += do_test(uplo, transA, transB, 13 * s, 11 * s,
                                        19 * s, 23 * s, 53 * s, uniform_random);
                }
            }
        }
    }
    std::cout << (failures == 0 ? "all passed" : "some failed") << std::endl;
    return failures;
}
