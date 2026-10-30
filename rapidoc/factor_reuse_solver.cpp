#include "factor_reuse_solver.h"
#include <cstdlib>

#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

extern "C"
{
#include "blasfeo_d_aux.h"
#include "blasfeo_d_blasfeo_api.h"
#include "blasfeo_stdlib.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <iomanip>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using namespace fatrop;

thread_local std::string last_error;

// Activated only by frs_solve_ablation. Nested preparation scopes pause the
// same clock once; production entry points retain their inclusive timers.
struct AblationSolveTimer
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point started = Clock::now();
    Clock::time_point paused_at;
    Clock::duration excluded = Clock::duration::zero();
    int pause_depth = 0;
    bool stopped = false;
    double call_seconds = 0.0;
    double solve_seconds = 0.0;

    void pause()
    {
        if (!stopped && pause_depth++ == 0)
            paused_at = Clock::now();
    }

    void resume()
    {
        if (!stopped && --pause_depth == 0)
            excluded += Clock::now() - paused_at;
    }

    void stop()
    {
        if (stopped)
            return;
        const auto ended = Clock::now();
        if (pause_depth > 0)
            excluded += ended - paused_at;
        call_seconds = std::chrono::duration<double>(ended - started).count();
        solve_seconds = std::max(0.0,
            std::chrono::duration<double>(ended - started - excluded).count());
        stopped = true;
    }
};

thread_local AblationSolveTimer *ablation_timer = nullptr;

struct ExcludeAblationPreparation
{
    AblationSolveTimer *timer = ablation_timer;
    ExcludeAblationPreparation() { if (timer) timer->pause(); }
    ~ExcludeAblationPreparation() { if (timer) timer->resume(); }
};

struct BindAblationTimer
{
    explicit BindAblationTimer(AblationSolveTimer &timer) { ablation_timer = &timer; }
    ~BindAblationTimer() { ablation_timer = nullptr; }
};

template <class T>
void resize_solve_workspace(std::vector<T> &values, std::size_t size)
{
    if (values.size() != size)
    {
        ExcludeAblationPreparation excluded;
        values.resize(size);
    }
}

int fail(int status, const std::string &message)
{
    last_error = message;
    return status;
}

void clear_error()
{
    last_error.clear();
}

double dot(const std::vector<double> &a, const std::vector<double> &b)
{
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (; i + 15 < a.size(); i += 16)
    {
        sum0 = _mm256_add_pd(
            sum0, _mm256_mul_pd(_mm256_loadu_pd(a.data() + i),
                                 _mm256_loadu_pd(b.data() + i)));
        sum1 = _mm256_add_pd(
            sum1, _mm256_mul_pd(_mm256_loadu_pd(a.data() + i + 4),
                                 _mm256_loadu_pd(b.data() + i + 4)));
        sum2 = _mm256_add_pd(
            sum2, _mm256_mul_pd(_mm256_loadu_pd(a.data() + i + 8),
                                 _mm256_loadu_pd(b.data() + i + 8)));
        sum3 = _mm256_add_pd(
            sum3, _mm256_mul_pd(_mm256_loadu_pd(a.data() + i + 12),
                                 _mm256_loadu_pd(b.data() + i + 12)));
    }
    sum0 = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                         _mm256_add_pd(sum2, sum3));
    const __m128d pair = _mm_add_pd(
        _mm256_castpd256_pd128(sum0), _mm256_extractf128_pd(sum0, 1));
    const __m128d total = _mm_hadd_pd(pair, pair);
    double value = _mm_cvtsd_f64(total);
    for (; i < a.size(); ++i)
        value += a[i] * b[i];
    return value;
#else
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        value += a[i] * b[i];
    return value;
#endif
}

double norm(const std::vector<double> &a)
{
    return std::sqrt(std::max(0.0, dot(a, a)));
}

void axpy(double alpha, const std::vector<double> &x,
          std::vector<double> &y)
{
    std::size_t i = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
    const __m256d coefficient = _mm256_set1_pd(alpha);
    for (; i + 3 < x.size(); i += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x.data() + i);
        const __m256d yv = _mm256_loadu_pd(y.data() + i);
        _mm256_storeu_pd(y.data() + i,
                         _mm256_add_pd(yv, _mm256_mul_pd(coefficient, xv)));
    }
#endif
    for (; i < x.size(); ++i)
        y[i] += alpha * x[i];
}

void axpy_raw(double alpha, const double *source, double *destination,
              int count)
{
    int index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
    const __m256d coefficient = _mm256_set1_pd(alpha);
    for (; index + 3 < count; index += 4)
    {
        const __m256d source_values = _mm256_loadu_pd(source + index);
        const __m256d destination_values =
            _mm256_loadu_pd(destination + index);
        _mm256_storeu_pd(
            destination + index,
            _mm256_add_pd(destination_values,
                          _mm256_mul_pd(coefficient, source_values)));
    }
#endif
    for (; index < count; ++index)
        destination[index] += alpha * source[index];
}

void apply_compressed_update(
    int rows, int columns, int rank, const double *weights,
    const double *compressed_solution, const double *compressed_image,
    double *solution, double *residual, double *errors)
{
    std::fill_n(errors, columns, 0.0);
    for (int row = 0; row < rows; ++row)
    {
        const double *solution_direction = compressed_solution +
            static_cast<std::size_t>(row) * rank;
        const double *image_direction = compressed_image +
            static_cast<std::size_t>(row) * rank;
        double *solution_row = solution +
            static_cast<std::size_t>(row) * columns;
        double *residual_row = residual +
            static_cast<std::size_t>(row) * columns;

        int column = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
        for (; column + 3 < columns; column += 4)
        {
            __m256d solution_update = _mm256_setzero_pd();
            __m256d residual_update = _mm256_setzero_pd();
            for (int direction = 0; direction < rank; ++direction)
            {
                const __m256d weight = _mm256_loadu_pd(
                    weights + static_cast<std::size_t>(direction) * columns +
                    column);
                solution_update = _mm256_add_pd(
                    solution_update,
                    _mm256_mul_pd(
                        _mm256_set1_pd(solution_direction[direction]), weight));
                residual_update = _mm256_add_pd(
                    residual_update,
                    _mm256_mul_pd(
                        _mm256_set1_pd(image_direction[direction]), weight));
            }
            const __m256d old_solution =
                _mm256_loadu_pd(solution_row + column);
            const __m256d updated_residual = _mm256_sub_pd(
                _mm256_loadu_pd(residual_row + column), residual_update);
            const __m256d old_errors = _mm256_loadu_pd(errors + column);
            _mm256_storeu_pd(
                solution_row + column,
                _mm256_add_pd(old_solution, solution_update));
            _mm256_storeu_pd(residual_row + column, updated_residual);
            _mm256_storeu_pd(
                errors + column,
                _mm256_add_pd(
                    old_errors,
                    _mm256_mul_pd(updated_residual, updated_residual)));
        }
#endif
        for (; column < columns; ++column)
        {
            double solution_update = 0.0;
            double residual_update = 0.0;
            for (int direction = 0; direction < rank; ++direction)
            {
                const double weight = weights[
                    static_cast<std::size_t>(direction) * columns + column];
                solution_update += solution_direction[direction] * weight;
                residual_update += image_direction[direction] * weight;
            }
            solution_row[column] += solution_update;
            residual_row[column] -= residual_update;
            errors[column] += residual_row[column] * residual_row[column];
        }
    }
}

void scale(double alpha, std::vector<double> &values)
{
    std::size_t i = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
    const __m256d coefficient = _mm256_set1_pd(alpha);
    for (; i + 3 < values.size(); i += 4)
    {
        const __m256d value = _mm256_loadu_pd(values.data() + i);
        _mm256_storeu_pd(values.data() + i,
                         _mm256_mul_pd(coefficient, value));
    }
#endif
    for (; i < values.size(); ++i)
        values[i] *= alpha;
}

std::vector<Index> controls(int horizon, int nu)
{
    std::vector<Index> value(horizon + 1, static_cast<Index>(nu));
    value.back() = 0;
    return value;
}

std::vector<Index> initial_constraints(int horizon, int nx)
{
    std::vector<Index> value(horizon + 1, 0);
    value.front() = static_cast<Index>(nx);
    return value;
}

class PackedMatrix
{
public:
    PackedMatrix() = default;

    PackedMatrix(int rows, int columns)
    {
        reset(rows, columns);
    }

    ~PackedMatrix()
    {
        release();
    }

    PackedMatrix(const PackedMatrix &) = delete;
    PackedMatrix &operator=(const PackedMatrix &) = delete;

    PackedMatrix(PackedMatrix &&other) noexcept
    {
        move_from(other);
    }

    PackedMatrix &operator=(PackedMatrix &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(other);
        }
        return *this;
    }

    void reset(int rows, int columns)
    {
        if (rows < 0 || columns < 0)
            throw std::invalid_argument("negative BLASFEO matrix dimension");
        if (rows == rows_ && columns == columns_ && storage_ != nullptr)
            return;

        ExcludeAblationPreparation excluded;
        release();
        rows_ = rows;
        columns_ = columns;
        const int allocated_rows = std::max(1, rows);
        const int allocated_columns = std::max(1, columns);
        const std::size_t bytes =
            blasfeo_memsize_dmat(allocated_rows, allocated_columns);
        blasfeo_malloc_align(&storage_, bytes);
        if (storage_ == nullptr)
            throw std::bad_alloc();
        blasfeo_create_dmat(allocated_rows, allocated_columns, &matrix_,
                            storage_);
    }

    void pack_row_major(const double *values)
    {
        if (rows_ == 0 || columns_ == 0)
            return;
        blasfeo_pack_tran_dmat(columns_, rows_, const_cast<double *>(values),
                               columns_, &matrix_, 0, 0);
    }

    void unpack_row_major(double *values)
    {
        if (rows_ == 0 || columns_ == 0)
            return;
        blasfeo_unpack_tran_dmat(rows_, columns_, &matrix_, 0, 0, values,
                                 columns_);
    }

    blasfeo_dmat *matrix()
    {
        return &matrix_;
    }

    blasfeo_dmat *matrix() const
    {
        return const_cast<blasfeo_dmat *>(&matrix_);
    }

private:
    void release() noexcept
    {
        if (storage_ != nullptr)
            blasfeo_free_align(storage_);
        storage_ = nullptr;
        rows_ = 0;
        columns_ = 0;
        matrix_ = {};
    }

    void move_from(PackedMatrix &other) noexcept
    {
        rows_ = other.rows_;
        columns_ = other.columns_;
        storage_ = other.storage_;
        matrix_ = other.matrix_;
        other.rows_ = 0;
        other.columns_ = 0;
        other.storage_ = nullptr;
        other.matrix_ = {};
    }

    int rows_ = 0;
    int columns_ = 0;
    void *storage_ = nullptr;
    blasfeo_dmat matrix_{};
};

using ProfileClock = std::chrono::steady_clock;
constexpr bool kDetailedResidualProfile = false;

double profile_seconds(ProfileClock::time_point started)
{
    return std::chrono::duration<double>(ProfileClock::now() - started).count();
}

bool detailed_sweep_profile_enabled()
{
    static const bool enabled = [] {
        const char *value = std::getenv("FRS_PROFILE_SWEEP");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

template <typename Operation>
void profile_sweep_kernel(bool enabled, double &seconds, int &calls,
                          Operation &&operation)
{
    if (!enabled)
    {
        operation();
        return;
    }
    const auto started = ProfileClock::now();
    operation();
    seconds += profile_seconds(started);
    ++calls;
}

struct TimingProfile
{
    double rhs_pack = 0.0;
    double backward = 0.0;
    double trsm = 0.0;
    double forward = 0.0;
    double output_unpack = 0.0;
    double arnoldi = 0.0;
    double residual = 0.0;
    double residual_pack = 0.0;
    double residual_hx = 0.0;
    double residual_at_lambda = 0.0;
    double residual_ax = 0.0;
    double residual_unpack = 0.0;
    double residual_subtract_norm = 0.0;
    double backward_gemm = 0.0;
    double backward_copy = 0.0;
    double backward_scaled_copy = 0.0;
    double backward_trsm_kernel = 0.0;
    int backward_gemm_calls = 0;
    int backward_copy_calls = 0;
    int backward_scaled_copy_calls = 0;
    int backward_trsm_calls = 0;
    double forward_gemm = 0.0;
    double forward_copy = 0.0;
    double forward_scaled_copy = 0.0;
    double forward_trsm_kernel = 0.0;
    int forward_gemm_calls = 0;
    int forward_copy_calls = 0;
    int forward_scaled_copy_calls = 0;
    int forward_trsm_calls = 0;
    double setup = 0.0;
    double difference_matvec = 0.0;
    double projection = 0.0;
    double input_copy = 0.0;
    double workspace = 0.0;
    double control = 0.0;
    double final_update = 0.0;
    std::vector<double> residual_history;
};

struct FactorWorkspace
{
    explicit FactorWorkspace(int horizon)
        : affine(horizon + 1), eliminated(horizon)
    {
    }

    PackedMatrix rhs;
    std::vector<PackedMatrix> affine;
    std::vector<PackedMatrix> eliminated;
    PackedMatrix propagated;
    PackedMatrix control_rhs;
    PackedMatrix control;
    PackedMatrix state;
    PackedMatrix next_state;
    PackedMatrix multiplier;
    PackedMatrix initial_multiplier;
    PackedMatrix solution;
    bool solution_is_packed = false;
};

const PackedMatrix *packed_solution(const FactorWorkspace &workspace)
{
    return workspace.solution_is_packed ? &workspace.solution : nullptr;
}

void store_profile(FrsStats &stats, const TimingProfile &profile)
{
    // Stop before statistics serialization and destruction of local operators.
    if (ablation_timer)
        ablation_timer->stop();
    stats.rhs_pack_seconds = profile.rhs_pack;
    stats.backward_seconds = profile.backward;
    stats.trsm_seconds = profile.trsm;
    stats.forward_seconds = profile.forward;
    stats.output_unpack_seconds = profile.output_unpack;
    stats.arnoldi_seconds = profile.arnoldi;
    stats.residual_seconds = profile.residual;
    stats.residual_pack_seconds = profile.residual_pack;
    stats.residual_hx_seconds = profile.residual_hx;
    stats.residual_at_lambda_seconds = profile.residual_at_lambda;
    stats.residual_ax_seconds = profile.residual_ax;
    stats.residual_unpack_seconds = profile.residual_unpack;
    stats.residual_subtract_norm_seconds = profile.residual_subtract_norm;
    stats.backward_gemm_seconds = profile.backward_gemm;
    stats.backward_copy_seconds = profile.backward_copy;
    stats.backward_scaled_copy_seconds = profile.backward_scaled_copy;
    stats.backward_trsm_kernel_seconds = profile.backward_trsm_kernel;
    stats.backward_gemm_calls = profile.backward_gemm_calls;
    stats.backward_copy_calls = profile.backward_copy_calls;
    stats.backward_scaled_copy_calls = profile.backward_scaled_copy_calls;
    stats.backward_trsm_calls = profile.backward_trsm_calls;
    stats.forward_gemm_seconds = profile.forward_gemm;
    stats.forward_copy_seconds = profile.forward_copy;
    stats.forward_scaled_copy_seconds = profile.forward_scaled_copy;
    stats.forward_trsm_kernel_seconds = profile.forward_trsm_kernel;
    stats.forward_gemm_calls = profile.forward_gemm_calls;
    stats.forward_copy_calls = profile.forward_copy_calls;
    stats.forward_scaled_copy_calls = profile.forward_scaled_copy_calls;
    stats.forward_trsm_calls = profile.forward_trsm_calls;
    stats.setup_seconds = profile.setup;
    stats.difference_matvec_seconds = profile.difference_matvec;
    stats.projection_seconds = profile.projection;
    stats.input_copy_seconds = profile.input_copy;
    stats.workspace_seconds = profile.workspace;
    stats.control_seconds = profile.control;
    stats.final_update_seconds = profile.final_update;
    stats.residual_history_count = static_cast<int>(std::min<std::size_t>(
        profile.residual_history.size(), FRS_MAX_RESIDUAL_HISTORY));
    std::fill_n(stats.residual_history, FRS_MAX_RESIDUAL_HISTORY, 0.0);
    std::copy_n(profile.residual_history.begin(),
                stats.residual_history_count, stats.residual_history);
}

/* Diagonalize a small symmetric stage update. All eigenpairs are retained:
   this is a numerical factor-update mechanism, not a low-rank truncation. */
void symmetric_jacobi_eigendecomposition(
    const std::vector<double> &input, int dimension,
    std::vector<double> &eigenvalues, std::vector<double> &eigenvectors)
{
    if (dimension <= 0 ||
        input.size() != static_cast<std::size_t>(dimension) * dimension)
        throw std::invalid_argument("invalid symmetric update matrix");

    std::vector<double> matrix = input;
    eigenvectors.assign(static_cast<std::size_t>(dimension) * dimension, 0.0);
    for (int index = 0; index < dimension; ++index)
        eigenvectors[static_cast<std::size_t>(index) * dimension + index] = 1.0;

    const double epsilon = std::numeric_limits<double>::epsilon();
    for (int sweep = 0; sweep < 64; ++sweep)
    {
        bool changed = false;
        for (int p = 0; p < dimension; ++p)
        {
            for (int q = p + 1; q < dimension; ++q)
            {
                const double app = matrix[static_cast<std::size_t>(p) * dimension + p];
                const double aqq = matrix[static_cast<std::size_t>(q) * dimension + q];
                const double apq = matrix[static_cast<std::size_t>(p) * dimension + q];
                const double threshold = 32.0 * epsilon *
                    (std::abs(app) + std::abs(aqq) + 1.0);
                if (std::abs(apq) <= threshold)
                    continue;

                changed = true;
                const double tau = (aqq - app) / (2.0 * apq);
                const double tangent = std::copysign(
                    1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;

                for (int row = 0; row < dimension; ++row)
                {
                    if (row == p || row == q)
                        continue;
                    const double arp = matrix[static_cast<std::size_t>(row) * dimension + p];
                    const double arq = matrix[static_cast<std::size_t>(row) * dimension + q];
                    const double new_rp = cosine * arp - sine * arq;
                    const double new_rq = sine * arp + cosine * arq;
                    matrix[static_cast<std::size_t>(row) * dimension + p] = new_rp;
                    matrix[static_cast<std::size_t>(p) * dimension + row] = new_rp;
                    matrix[static_cast<std::size_t>(row) * dimension + q] = new_rq;
                    matrix[static_cast<std::size_t>(q) * dimension + row] = new_rq;
                }
                matrix[static_cast<std::size_t>(p) * dimension + p] = app - tangent * apq;
                matrix[static_cast<std::size_t>(q) * dimension + q] = aqq + tangent * apq;
                matrix[static_cast<std::size_t>(p) * dimension + q] = 0.0;
                matrix[static_cast<std::size_t>(q) * dimension + p] = 0.0;

                for (int row = 0; row < dimension; ++row)
                {
                    const double vrp = eigenvectors[
                        static_cast<std::size_t>(row) * dimension + p];
                    const double vrq = eigenvectors[
                        static_cast<std::size_t>(row) * dimension + q];
                    eigenvectors[static_cast<std::size_t>(row) * dimension + p] =
                        cosine * vrp - sine * vrq;
                    eigenvectors[static_cast<std::size_t>(row) * dimension + q] =
                        sine * vrp + cosine * vrq;
                }
            }
        }
        if (!changed)
            break;
    }

    double diagonal_scale = 1.0;
    double off_diagonal_maximum = 0.0;
    for (int row = 0; row < dimension; ++row)
    {
        diagonal_scale = std::max(diagonal_scale, std::abs(
            matrix[static_cast<std::size_t>(row) * dimension + row]));
        for (int column = row + 1; column < dimension; ++column)
            off_diagonal_maximum = std::max(off_diagonal_maximum, std::abs(
                matrix[static_cast<std::size_t>(row) * dimension + column]));
    }
    if (off_diagonal_maximum > 1.0e-12 * diagonal_scale)
        throw std::runtime_error("signed stage update eigensolve did not converge");

    eigenvalues.resize(dimension);
    for (int index = 0; index < dimension; ++index)
        eigenvalues[index] = matrix[
            static_cast<std::size_t>(index) * dimension + index];
}

void apply_signed_cholesky_rank_one(
    std::vector<double> &lower, int dimension,
    std::vector<double> update, double sign)
{
    for (int column = 0; column < dimension; ++column)
    {
        const double diagonal = lower[
            static_cast<std::size_t>(column) * dimension + column];
        const double radicand = diagonal * diagonal +
                                sign * update[column] * update[column];
        if (!(diagonal > 0.0) || !(radicand > 0.0) || !std::isfinite(radicand))
            throw std::runtime_error(
                "target control block is not positive definite during factor update");

        const double replacement = std::sqrt(radicand);
        const double cosine = replacement / diagonal;
        const double sine = update[column] / diagonal;
        lower[static_cast<std::size_t>(column) * dimension + column] = replacement;

        for (int row = column + 1; row < dimension; ++row)
        {
            const std::size_t offset =
                static_cast<std::size_t>(row) * dimension + column;
            const double updated = (lower[offset] +
                                    sign * sine * update[row]) / cosine;
            lower[offset] = updated;
            update[row] = cosine * update[row] - sine * updated;
        }
    }
}

std::vector<double> exact_signed_control_factor_update(
    const double *cached_lower, const std::vector<double> &target_block,
    int dimension)
{
    std::vector<double> lower(static_cast<std::size_t>(dimension) * dimension, 0.0);
    std::vector<double> difference(target_block);
    for (int row = 0; row < dimension; ++row)
        for (int column = 0; column <= row; ++column)
            lower[static_cast<std::size_t>(row) * dimension + column] =
                cached_lower[static_cast<std::size_t>(row) * dimension + column];

    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            double forward_entry = 0.0;
            for (int inner = 0; inner <= std::min(row, column); ++inner)
                forward_entry +=
                    lower[static_cast<std::size_t>(row) * dimension + inner] *
                    lower[static_cast<std::size_t>(column) * dimension + inner];
            difference[static_cast<std::size_t>(row) * dimension + column] -=
                forward_entry;
        }
    }

    for (int row = 0; row < dimension; ++row)
        for (int column = row + 1; column < dimension; ++column)
        {
            const double symmetric = 0.5 * (
                difference[static_cast<std::size_t>(row) * dimension + column] +
                difference[static_cast<std::size_t>(column) * dimension + row]);
            difference[static_cast<std::size_t>(row) * dimension + column] = symmetric;
            difference[static_cast<std::size_t>(column) * dimension + row] = symmetric;
        }

    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    symmetric_jacobi_eigendecomposition(
        difference, dimension, eigenvalues, eigenvectors);

    /* Positive updates are applied first. During the subsequent downdates,
       every intermediate matrix equals the positive-definite target plus the
       remaining positive semidefinite downdates. */
    for (int pass = 0; pass < 2; ++pass)
    {
        for (int eigen = 0; eigen < dimension; ++eigen)
        {
            const double value = eigenvalues[eigen];
            if ((pass == 0 && value <= 0.0) ||
                (pass == 1 && value >= 0.0) || value == 0.0)
                continue;
            std::vector<double> direction(dimension);
            const double magnitude = std::sqrt(std::abs(value));
            for (int row = 0; row < dimension; ++row)
                direction[row] = magnitude * eigenvectors[
                    static_cast<std::size_t>(row) * dimension + eigen];
            apply_signed_cholesky_rank_one(
                lower, dimension, std::move(direction), value > 0.0 ? 1.0 : -1.0);
        }
    }
    return lower;
}

/* Cached F_F and the structured backsolve that defines P_F^{-1}. This class
   is a factorization object, not a stored preconditioner matrix P_F. */
struct FreshFactorizationContext
{
    FreshFactorizationContext(int horizon, int nx, int nu)
        : dims(horizon + 1, controls(horizon, nu),
               std::vector<Index>(horizon + 1, static_cast<Index>(nx)),
               initial_constraints(horizon, nx),
               std::vector<Index>(horizon + 1, 0)),
          info(dims), jacobian(dims), hess(dims), solver(info),
          d_x(info.number_of_primal_variables),
          d_s(info.number_of_slack_variables),
          f(info.number_of_primal_variables), g(info.number_of_eq_constraints),
          x(info.number_of_primal_variables), mult(info.number_of_eq_constraints)
    {
    }

    ProblemDims<OcpType> dims;
    ProblemInfo<OcpType> info;
    Jacobian<OcpType> jacobian;
    Hessian<OcpType> hess;
    AugSystemSolver<OcpType> solver;
    VecRealAllocated d_x;
    VecRealAllocated d_s;
    VecRealAllocated f;
    VecRealAllocated g;
    VecRealAllocated x;
    VecRealAllocated mult;
};

class CachedForwardFactorization
{
public:
    double fresh_numeric_factorization_seconds = 0.0;

    CachedForwardFactorization(int horizon, int nx, int nu, const double *hessian,
                 const double *terminal_hessian, const double *a_blocks,
                 const double *cached_value_hessians = nullptr,
                 const double *cached_control_cholesky = nullptr,
                 const double *cached_cross_factors = nullptr)
        : horizon(horizon), nx(nx), nu(nu), stage_dim(nx + nu),
          n_primal(horizon * (nx + nu) + nx),
          n_dual((horizon + 1) * nx), dimension(n_primal + n_dual),
          created_by_fresh_factorization(cached_value_hessians == nullptr)
    {
        const bool reuse_factors = cached_value_hessians != nullptr;
        if (reuse_factors != (cached_control_cholesky != nullptr) ||
            reuse_factors != (cached_cross_factors != nullptr))
            throw std::invalid_argument(
                "cached FATROP factors must be supplied together");

        if (!reuse_factors)
        {
            fresh_context = std::make_unique<FreshFactorizationContext>(
                horizon, nx, nu);
            const auto fresh_numeric_started = std::chrono::steady_clock::now();
            auto &info = fresh_context->info;
            auto &jacobian = fresh_context->jacobian;
            auto &hess = fresh_context->hess;
            auto &solver = fresh_context->solver;
            auto &d_x = fresh_context->d_x;
            auto &d_s = fresh_context->d_s;
            auto &f = fresh_context->f;
            auto &g = fresh_context->g;
            auto &x = fresh_context->x;
            auto &mult = fresh_context->mult;

            for (int k = 0; k < horizon; ++k)
            {
                for (int row = 0; row < stage_dim; ++row)
                {
                    const int external_row = row < nu ? nx + row : row - nu;
                    for (int column = 0; column < stage_dim; ++column)
                    {
                        const int external_column =
                            column < nu ? nx + column : column - nu;
                        hess.RSQrqt[k](row, column) =
                            hessian[(k * stage_dim + external_row) *
                                        stage_dim +
                                    external_column];
                    }
                }

                for (int row = 0; row < nx; ++row)
                {
                    for (int j = 0; j < nu; ++j)
                        jacobian.BAbt[k](j, row) =
                            -a_blocks[index_a(k, row, nx + j)];
                    for (int i = 0; i < nx; ++i)
                        jacobian.BAbt[k](nu + i, row) =
                            -a_blocks[index_a(k, row, i)];
                }
            }

            for (int row = 0; row < nx; ++row)
            {
                for (int column = 0; column < nx; ++column)
                    hess.RSQrqt[horizon](row, column) =
                        terminal_hessian[row * nx + column];
                jacobian.Gg_eqt[0](nu + row, row) = -1.0;
            }

            d_x = 0.0;
            d_s = 0.0;
            f = 0.0;
            g = 0.0;
            x = 0.0;
            mult = 0.0;
            if (solver.solve(info, jacobian, hess, d_x, d_s, f, g, x,
                             mult) !=
                LinsolReturnFlag::SUCCESS)
                throw std::runtime_error("FATROP factorization failed");
            fresh_numeric_factorization_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - fresh_numeric_started)
                    .count();
        }

        std::vector<double> value_hessians(
            static_cast<std::size_t>(horizon + 1) * nx * nx);
        std::vector<double> control_cholesky(std::max<std::size_t>(
            1, static_cast<std::size_t>(horizon) * nu * nu));
        std::vector<double> cross_factors(std::max<std::size_t>(
            1, static_cast<std::size_t>(horizon) * nx * nu));
        if (reuse_factors)
        {
            std::copy(cached_value_hessians,
                      cached_value_hessians + value_hessians.size(),
                      value_hessians.begin());
            std::copy(cached_control_cholesky,
                      cached_control_cholesky + control_cholesky.size(),
                      control_cholesky.begin());
            std::copy(cached_cross_factors,
                      cached_cross_factors + cross_factors.size(),
                      cross_factors.begin());
        }
        else if (!fresh_context->solver.export_rhs_factors(
                     fresh_context->info, value_hessians.data(),
                     control_cholesky.data(), cross_factors.data()))
            throw std::runtime_error(
                "FATROP factors are not compatible with matrix RHS solves");

        control_cholesky_values = control_cholesky;

        packed_first_hessian.reset(nx, stage_dim);
        packed_first_hessian.pack_row_major(hessian);

        packed_dynamics.reserve(horizon);
        packed_value_hessians.reserve(horizon + 1);
        packed_control_cholesky.reserve(horizon);
        packed_cross_factors.reserve(horizon);

        for (int stage = 0; stage < horizon; ++stage)
        {
            packed_dynamics.emplace_back(nx, stage_dim);
            packed_dynamics.back().pack_row_major(
                a_blocks + static_cast<std::size_t>(stage) * nx * stage_dim);

            packed_control_cholesky.emplace_back(nu, nu);
            packed_cross_factors.emplace_back(nx, nu);
            if (nu > 0)
            {
                packed_control_cholesky.back().pack_row_major(
                    control_cholesky.data() +
                    static_cast<std::size_t>(stage) * nu * nu);
                packed_cross_factors.back().pack_row_major(
                    cross_factors.data() +
                    static_cast<std::size_t>(stage) * nx * nu);
            }
        }

        for (int stage = 0; stage <= horizon; ++stage)
        {
            packed_value_hessians.emplace_back(nx, nx);
            packed_value_hessians.back().pack_row_major(
                value_hessians.data() +
                static_cast<std::size_t>(stage) * nx * nx);
        }

    }

    CachedForwardFactorization(int horizon, int nx, int nu,
                               FrsCachedSolve cached_solve,
                               void *cached_solve_context)
        : horizon(horizon), nx(nx), nu(nu), stage_dim(nx + nu),
          n_primal(horizon * (nx + nu) + nx),
          n_dual((horizon + 1) * nx), dimension(n_primal + n_dual),
          created_by_fresh_factorization(false),
          native_cached_solve(cached_solve),
          native_cached_solve_context(cached_solve_context)
    {
        if (native_cached_solve == nullptr ||
            native_cached_solve_context == nullptr)
            throw std::invalid_argument("native cached solve is null");
    }

    const std::vector<double> &cached_control_factors() const
    {
        return control_cholesky_values;
    }

    void prepare_workspace(FactorWorkspace &workspace) const
    {
        (void)workspace;
    }

    bool solve(const std::vector<double> &rhs, int columns,
               std::vector<double> &output, TimingProfile &timing,
               FactorWorkspace &workspace) const
    {
        workspace.solution_is_packed = false;
        if (native_cached_solve != nullptr)
        {
            if (columns <= 0 ||
                rhs.size() != static_cast<std::size_t>(dimension) * columns)
                return false;
            resize_solve_workspace(output, rhs.size());
            const auto started = ProfileClock::now();
            const int status = native_cached_solve(
                native_cached_solve_context, columns, rhs.data(),
                output.data());
            timing.backward += profile_seconds(started);
            if (status != 0)
                return false;
            return true;
        }

        auto &rhs_workspace = workspace.rhs;
        auto &affine_workspace = workspace.affine;
        auto &eliminated_workspace = workspace.eliminated;
        auto &propagated_workspace = workspace.propagated;
        auto &control_rhs_workspace = workspace.control_rhs;
        auto &control_workspace = workspace.control;
        auto &state_workspace = workspace.state;
        auto &next_state_workspace = workspace.next_state;
        auto &multiplier_workspace = workspace.multiplier;
        auto &initial_multiplier_workspace = workspace.initial_multiplier;
        auto &solution_workspace = workspace.solution;

        if (columns <= 0 ||
            rhs.size() != static_cast<std::size_t>(dimension) * columns)
            return false;

        const std::size_t output_size =
            static_cast<std::size_t>(dimension) * columns;
        if (output.size() != output_size)
            resize_solve_workspace(output, output_size);
        const bool profile_sweep = detailed_sweep_profile_enabled();
        solution_workspace.reset(dimension, columns);
        rhs_workspace.reset(dimension, columns);
        const auto pack_started = ProfileClock::now();
        rhs_workspace.pack_row_major(rhs.data());
        timing.rhs_pack += profile_seconds(pack_started);
        propagated_workspace.reset(nx, columns);
        control_rhs_workspace.reset(nu, columns);
        control_workspace.reset(nu, columns);
        state_workspace.reset(nx, columns);
        next_state_workspace.reset(nx, columns);
        multiplier_workspace.reset(nx, columns);
        initial_multiplier_workspace.reset(nx, columns);
        for (PackedMatrix &matrix : affine_workspace)
            matrix.reset(nx, columns);
        for (PackedMatrix &matrix : eliminated_workspace)
            matrix.reset(nu, columns);

        const int terminal = horizon * stage_dim;
        const auto backward_started = ProfileClock::now();
        double backward_trsm = 0.0;
        profile_sweep_kernel(
            profile_sweep, timing.backward_scaled_copy,
            timing.backward_scaled_copy_calls, [&] {
                blasfeo_dgecpsc(
                    nx, columns, -1.0, rhs_workspace.matrix(), terminal, 0,
                    affine_workspace[horizon].matrix(), 0, 0);
            });
        for (int stage = horizon - 1; stage >= 0; --stage)
        {
            const int primal = stage * stage_dim;
            profile_sweep_kernel(
                profile_sweep, timing.backward_gemm,
                timing.backward_gemm_calls, [&] {
                    blasfeo_dgemm_nn(
                        nx, columns, nx, 1.0,
                        packed_value_hessians[stage + 1].matrix(), 0, 0,
                        rhs_workspace.matrix(),
                        n_primal + (stage + 1) * nx, 0, 1.0,
                        affine_workspace[stage + 1].matrix(), 0, 0,
                        propagated_workspace.matrix(), 0, 0);
                });

            if (nu > 0)
            {
                profile_sweep_kernel(
                    profile_sweep, timing.backward_gemm,
                    timing.backward_gemm_calls, [&] {
                        blasfeo_dgemm_tn(
                            nu, columns, nx, -1.0,
                            packed_dynamics[stage].matrix(), 0, nx,
                            propagated_workspace.matrix(), 0, 0, -1.0,
                            rhs_workspace.matrix(), primal + nx, 0,
                            control_rhs_workspace.matrix(), 0, 0);
                    });
                const auto trsm_started = ProfileClock::now();
                blasfeo_dtrsm_llnn(
                    nu, columns, 1.0,
                    packed_control_cholesky[stage].matrix(), 0, 0,
                    control_rhs_workspace.matrix(), 0, 0,
                    eliminated_workspace[stage].matrix(), 0, 0);
                const double trsm_elapsed = profile_seconds(trsm_started);
                backward_trsm += trsm_elapsed;
                if (profile_sweep)
                {
                    timing.backward_trsm_kernel += trsm_elapsed;
                    ++timing.backward_trsm_calls;
                }
            }

            profile_sweep_kernel(
                profile_sweep, timing.backward_gemm,
                timing.backward_gemm_calls, [&] {
                    blasfeo_dgemm_tn(
                        nx, columns, nx, -1.0,
                        packed_dynamics[stage].matrix(), 0, 0,
                        propagated_workspace.matrix(), 0, 0, -1.0,
                        rhs_workspace.matrix(), primal, 0,
                        affine_workspace[stage].matrix(), 0, 0);
                });
            if (nu > 0)
                profile_sweep_kernel(
                    profile_sweep, timing.backward_gemm,
                    timing.backward_gemm_calls, [&] {
                        blasfeo_dgemm_nn(
                            nx, columns, nu, -1.0,
                            packed_cross_factors[stage].matrix(), 0, 0,
                            eliminated_workspace[stage].matrix(), 0, 0, 1.0,
                            affine_workspace[stage].matrix(), 0, 0,
                            affine_workspace[stage].matrix(), 0, 0);
                    });
        }
        timing.backward += profile_seconds(backward_started) - backward_trsm;
        timing.trsm += backward_trsm;

        const auto forward_started = ProfileClock::now();
        double forward_trsm = 0.0;
        double output_unpack = 0.0;
        profile_sweep_kernel(
            profile_sweep, timing.forward_copy, timing.forward_copy_calls,
            [&] {
                blasfeo_dgecp(nx, columns, rhs_workspace.matrix(), n_primal,
                              0, state_workspace.matrix(), 0, 0);
            });

        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            if (nu > 0)
            {
                profile_sweep_kernel(
                    profile_sweep, timing.forward_gemm,
                    timing.forward_gemm_calls, [&] {
                        blasfeo_dgemm_tn(
                            nu, columns, nx, -1.0,
                            packed_cross_factors[stage].matrix(), 0, 0,
                            state_workspace.matrix(), 0, 0, -1.0,
                            eliminated_workspace[stage].matrix(), 0, 0,
                            control_rhs_workspace.matrix(), 0, 0);
                    });
                const auto trsm_started = ProfileClock::now();
                blasfeo_dtrsm_lltn(
                    nu, columns, 1.0,
                    packed_control_cholesky[stage].matrix(), 0, 0,
                    control_rhs_workspace.matrix(), 0, 0,
                    control_workspace.matrix(), 0, 0);
                const double trsm_elapsed = profile_seconds(trsm_started);
                forward_trsm += trsm_elapsed;
                if (profile_sweep)
                {
                    timing.forward_trsm_kernel += trsm_elapsed;
                    ++timing.forward_trsm_calls;
                }
            }

            profile_sweep_kernel(
                profile_sweep, timing.forward_copy,
                timing.forward_copy_calls, [&] {
                    blasfeo_dgecp(nx, columns, state_workspace.matrix(), 0, 0,
                                  solution_workspace.matrix(), primal, 0);
                });
            if (nu > 0)
                profile_sweep_kernel(
                    profile_sweep, timing.forward_copy,
                    timing.forward_copy_calls, [&] {
                        blasfeo_dgecp(
                            nu, columns, control_workspace.matrix(), 0, 0,
                            solution_workspace.matrix(), primal + nx, 0);
                    });

            profile_sweep_kernel(
                profile_sweep, timing.forward_copy,
                timing.forward_copy_calls, [&] {
                    blasfeo_dgecp(nx, columns, rhs_workspace.matrix(),
                                  n_primal + (stage + 1) * nx, 0,
                                  next_state_workspace.matrix(), 0, 0);
                });
            profile_sweep_kernel(
                profile_sweep, timing.forward_gemm,
                timing.forward_gemm_calls, [&] {
                    blasfeo_dgemm_nn(
                        nx, columns, nx, -1.0,
                        packed_dynamics[stage].matrix(), 0, 0,
                        state_workspace.matrix(), 0, 0, 1.0,
                        next_state_workspace.matrix(), 0, 0,
                        next_state_workspace.matrix(), 0, 0);
                });
            if (nu > 0)
                profile_sweep_kernel(
                    profile_sweep, timing.forward_gemm,
                    timing.forward_gemm_calls, [&] {
                        blasfeo_dgemm_nn(
                            nx, columns, nu, -1.0,
                            packed_dynamics[stage].matrix(), 0, nx,
                            control_workspace.matrix(), 0, 0, 1.0,
                            next_state_workspace.matrix(), 0, 0,
                            next_state_workspace.matrix(), 0, 0);
                    });

            profile_sweep_kernel(
                profile_sweep, timing.forward_gemm,
                timing.forward_gemm_calls, [&] {
                    blasfeo_dgemm_nn(
                        nx, columns, nx, -1.0,
                        packed_value_hessians[stage + 1].matrix(), 0, 0,
                        next_state_workspace.matrix(), 0, 0, -1.0,
                        affine_workspace[stage + 1].matrix(), 0, 0,
                        multiplier_workspace.matrix(), 0, 0);
                });
            profile_sweep_kernel(
                profile_sweep, timing.forward_copy,
                timing.forward_copy_calls, [&] {
                    blasfeo_dgecp(
                        nx, columns, multiplier_workspace.matrix(), 0, 0,
                        solution_workspace.matrix(),
                        n_primal + (stage + 1) * nx, 0);
                });

            if (stage == 0)
            {
                profile_sweep_kernel(
                    profile_sweep, timing.forward_copy,
                    timing.forward_copy_calls, [&] {
                        blasfeo_dgecp(
                            nx, columns, rhs_workspace.matrix(), 0, 0,
                            initial_multiplier_workspace.matrix(), 0, 0);
                    });
                profile_sweep_kernel(
                    profile_sweep, timing.forward_gemm,
                    timing.forward_gemm_calls, [&] {
                        blasfeo_dgemm_nn(
                            nx, columns, nx, -1.0,
                            packed_first_hessian.matrix(), 0, 0,
                            state_workspace.matrix(), 0, 0, 1.0,
                            initial_multiplier_workspace.matrix(), 0, 0,
                            initial_multiplier_workspace.matrix(), 0, 0);
                    });
                if (nu > 0)
                    profile_sweep_kernel(
                        profile_sweep, timing.forward_gemm,
                        timing.forward_gemm_calls, [&] {
                            blasfeo_dgemm_nn(
                                nx, columns, nu, -1.0,
                                packed_first_hessian.matrix(), 0, nx,
                                control_workspace.matrix(), 0, 0, 1.0,
                                initial_multiplier_workspace.matrix(), 0, 0,
                                initial_multiplier_workspace.matrix(), 0, 0);
                        });
                profile_sweep_kernel(
                    profile_sweep, timing.forward_gemm,
                    timing.forward_gemm_calls, [&] {
                        blasfeo_dgemm_tn(
                            nx, columns, nx, -1.0,
                            packed_dynamics[0].matrix(), 0, 0,
                            multiplier_workspace.matrix(), 0, 0, 1.0,
                            initial_multiplier_workspace.matrix(), 0, 0,
                            initial_multiplier_workspace.matrix(), 0, 0);
                    });
                profile_sweep_kernel(
                    profile_sweep, timing.forward_copy,
                    timing.forward_copy_calls, [&] {
                        blasfeo_dgecp(
                            nx, columns,
                            initial_multiplier_workspace.matrix(), 0, 0,
                            solution_workspace.matrix(), n_primal, 0);
                    });
            }

            std::swap(state_workspace, next_state_workspace);
        }

        profile_sweep_kernel(
            profile_sweep, timing.forward_copy, timing.forward_copy_calls,
            [&] {
                blasfeo_dgecp(nx, columns, state_workspace.matrix(), 0, 0,
                              solution_workspace.matrix(), terminal, 0);
            });
        const auto output_unpack_started = ProfileClock::now();
        solution_workspace.unpack_row_major(output.data());
        output_unpack += profile_seconds(output_unpack_started);
        const double forward_total = profile_seconds(forward_started);
        timing.forward += forward_total - forward_trsm - output_unpack;
        timing.trsm += forward_trsm;
        timing.output_unpack += output_unpack;
        workspace.solution_is_packed = true;
        return true;
    }

    FrsCachedSolve native_cached_solve = nullptr;
    void *native_cached_solve_context = nullptr;

    bool export_rhs_factors(double *value_hessians,
                            double *control_cholesky,
                            double *cross_factors) const
    {
        return created_by_fresh_factorization && fresh_context != nullptr &&
               fresh_context->solver.export_rhs_factors(
                   fresh_context->info, value_hessians, control_cholesky,
                   cross_factors);
    }

    const int horizon;
    const int nx;
    const int nu;
    const int stage_dim;
    const int n_primal;
    const int n_dual;
    const int dimension;

private:
    const bool created_by_fresh_factorization;

    std::size_t index_a(int stage, int row, int column) const
    {
        return static_cast<std::size_t>((stage * nx + row) * stage_dim + column);
    }

    std::unique_ptr<FreshFactorizationContext> fresh_context;
    PackedMatrix packed_first_hessian;
    std::vector<PackedMatrix> packed_dynamics;
    std::vector<PackedMatrix> packed_value_hessians;
    std::vector<PackedMatrix> packed_control_cholesky;
    std::vector<PackedMatrix> packed_cross_factors;
    std::vector<double> control_cholesky_values;
};

class BatchedKktOperator
{
public:
    BatchedKktOperator(int horizon, int nx, int nu, const double *hessian,
                       const double *terminal_hessian,
                       const double *a_blocks, bool prepare_packed = true)
        : horizon(horizon), nx(nx), nu(nu), stage_dim(nx + nu),
          n_primal(horizon * (nx + nu) + nx),
          dimension(n_primal + (horizon + 1) * nx),
          packed_terminal_hessian(nx, nx)
    {
        ExcludeAblationPreparation excluded;
        if (!prepare_packed)
            return;
        packed_hessians.reserve(horizon);
        packed_dynamics.reserve(horizon);
        for (int stage = 0; stage < horizon; ++stage)
        {
            packed_hessians.emplace_back(stage_dim, stage_dim);
            packed_hessians.back().pack_row_major(
                hessian + static_cast<std::size_t>(stage) * stage_dim *
                              stage_dim);
            packed_dynamics.emplace_back(nx, stage_dim);
            packed_dynamics.back().pack_row_major(
                a_blocks + static_cast<std::size_t>(stage) * nx * stage_dim);
        }
        packed_terminal_hessian.pack_row_major(terminal_hessian);
    }

    bool apply(const std::vector<double> &input, int columns,
               std::vector<double> &output,
               const PackedMatrix *packed_input = nullptr,
               TimingProfile *profile = nullptr,
               const std::vector<double> *rhs_for_residual = nullptr,
               std::vector<double> *residual_squared_norms = nullptr,
               std::vector<double> *rhs_squared_norms = nullptr)
    {
        if (columns <= 0 ||
            input.size() != static_cast<std::size_t>(dimension) * columns)
            return false;
        const bool fused_residual = rhs_for_residual != nullptr;
        if (fused_residual &&
            (rhs_for_residual->size() != input.size() ||
             residual_squared_norms == nullptr ||
             rhs_squared_norms == nullptr))
            return false;
        if (fused_residual)
        {
            ExcludeAblationPreparation excluded;
            residual_squared_norms->assign(columns, 0.0);
            rhs_squared_norms->assign(columns, 0.0);
        }

        ProfileClock::time_point component_started;
        if (profile != nullptr)
            component_started = ProfileClock::now();
        prepare(columns, packed_input == nullptr, false);
        if (packed_input == nullptr)
            input_workspace.pack_row_major(input.data());
        blasfeo_dmat *input_matrix = packed_input == nullptr
            ? input_workspace.matrix()
            : packed_input->matrix();
        resize_solve_workspace(output, input.size());
        if (profile != nullptr)
            profile->residual_pack += profile_seconds(component_started);

        auto emit_block = [&](int block_rows, int row_offset) {
            if (!fused_residual)
            {
                unpack_block(
                    block_rows, columns,
                    output.data() +
                        static_cast<std::size_t>(row_offset) * columns);
                return;
            }

            if (profile != nullptr)
                component_started = ProfileClock::now();
            unpack_block(block_rows, columns, block_buffer.data());
            if (profile != nullptr)
                profile->residual_unpack +=
                    profile_seconds(component_started);

            if (profile != nullptr)
                component_started = ProfileClock::now();
            for (int row = 0; row < block_rows; ++row)
                for (int column = 0; column < columns; ++column)
                {
                    const std::size_t local =
                        static_cast<std::size_t>(row) * columns + column;
                    const std::size_t global =
                        static_cast<std::size_t>(row_offset + row) * columns +
                        column;
                    const double rhs_value = (*rhs_for_residual)[global];
                    const double residual = rhs_value - block_buffer[local];
                    output[global] = residual;
                    (*residual_squared_norms)[column] += residual * residual;
                    (*rhs_squared_norms)[column] += rhs_value * rhs_value;
                }
            if (profile != nullptr)
                profile->residual_subtract_norm +=
                    profile_seconds(component_started);
        };

        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            const int current_multiplier = n_primal + stage * nx;
            const int next_multiplier = n_primal + (stage + 1) * nx;
            if (profile != nullptr)
                component_started = ProfileClock::now();
            blasfeo_dgemm_nn(
                stage_dim, columns, stage_dim, 1.0,
                packed_hessians[stage].matrix(), 0, 0,
                input_matrix, primal, 0, 0.0,
                block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0,
                0);
            if (profile != nullptr)
                profile->residual_hx += profile_seconds(component_started);
            if (profile != nullptr)
                component_started = ProfileClock::now();
            blasfeo_dgead(nx, columns, 1.0, input_matrix,
                          current_multiplier, 0, block_workspace.matrix(), 0,
                          0);
            blasfeo_dgemm_tn(
                stage_dim, columns, nx, 1.0,
                packed_dynamics[stage].matrix(), 0, 0,
                input_matrix, next_multiplier, 0, 1.0,
                block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0,
                0);
            if (profile != nullptr)
                profile->residual_at_lambda +=
                    profile_seconds(component_started);
            emit_block(stage_dim, primal);
        }

        const int terminal = horizon * stage_dim;
        const int terminal_multiplier = n_primal + horizon * nx;
        if (profile != nullptr)
            component_started = ProfileClock::now();
        blasfeo_dgemm_nn(
            nx, columns, nx, 1.0, packed_terminal_hessian.matrix(), 0, 0,
            input_matrix, terminal, 0, 0.0,
            block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0, 0);
        if (profile != nullptr)
            profile->residual_hx += profile_seconds(component_started);
        if (profile != nullptr)
            component_started = ProfileClock::now();
        blasfeo_dgead(nx, columns, 1.0, input_matrix,
                      terminal_multiplier, 0, block_workspace.matrix(), 0, 0);
        if (profile != nullptr)
            profile->residual_at_lambda += profile_seconds(component_started);
        emit_block(nx, terminal);

        if (profile != nullptr)
            component_started = ProfileClock::now();
        blasfeo_dgecp(nx, columns, input_matrix, 0, 0,
                      block_workspace.matrix(), 0, 0);
        if (profile != nullptr)
            profile->residual_ax += profile_seconds(component_started);
        emit_block(nx, n_primal);

        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            const int next_primal =
                stage + 1 < horizon ? (stage + 1) * stage_dim : terminal;
            if (profile != nullptr)
                component_started = ProfileClock::now();
            blasfeo_dgemm_nn(
                nx, columns, stage_dim, 1.0,
                packed_dynamics[stage].matrix(), 0, 0,
                input_matrix, primal, 0, 1.0,
                input_matrix, next_primal, 0,
                block_workspace.matrix(), 0, 0);
            if (profile != nullptr)
                profile->residual_ax += profile_seconds(component_started);
            emit_block(nx, n_primal + (stage + 1) * nx);
        }
        return true;
    }

    double relative_residual(const std::vector<double> &rhs,
                             const std::vector<double> &solution,
                             int columns)
    {
        if (columns <= 0 || rhs.size() != solution.size() ||
            rhs.size() != static_cast<std::size_t>(dimension) * columns)
            return std::numeric_limits<double>::infinity();

        prepare(columns, true, true);
        rhs_workspace.pack_row_major(rhs.data());
        input_workspace.pack_row_major(solution.data());
        double residual_squared = 0.0;

        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            const int current_multiplier = n_primal + stage * nx;
            const int next_multiplier = n_primal + (stage + 1) * nx;
            blasfeo_dgecp(stage_dim, columns, rhs_workspace.matrix(), primal, 0,
                          block_workspace.matrix(), 0, 0);
            blasfeo_dgemm_nn(
                stage_dim, columns, stage_dim, -1.0,
                packed_hessians[stage].matrix(), 0, 0,
                input_workspace.matrix(), primal, 0, 1.0,
                block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0,
                0);
            blasfeo_dgead(nx, columns, -1.0, input_workspace.matrix(),
                          current_multiplier, 0, block_workspace.matrix(), 0,
                          0);
            blasfeo_dgemm_tn(
                stage_dim, columns, nx, -1.0,
                packed_dynamics[stage].matrix(), 0, 0,
                input_workspace.matrix(), next_multiplier, 0, 1.0,
                block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0,
                0);
            residual_squared += block_squared_norm(stage_dim, columns);
        }

        const int terminal = horizon * stage_dim;
        const int terminal_multiplier = n_primal + horizon * nx;
        blasfeo_dgecp(nx, columns, rhs_workspace.matrix(), terminal, 0,
                      block_workspace.matrix(), 0, 0);
        blasfeo_dgemm_nn(
            nx, columns, nx, -1.0, packed_terminal_hessian.matrix(), 0, 0,
            input_workspace.matrix(), terminal, 0, 1.0,
            block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0, 0);
        blasfeo_dgead(nx, columns, -1.0, input_workspace.matrix(),
                      terminal_multiplier, 0, block_workspace.matrix(), 0, 0);
        residual_squared += block_squared_norm(nx, columns);

        blasfeo_dgecp(nx, columns, rhs_workspace.matrix(), n_primal, 0,
                      block_workspace.matrix(), 0, 0);
        blasfeo_dgead(nx, columns, -1.0, input_workspace.matrix(), 0, 0,
                      block_workspace.matrix(), 0, 0);
        residual_squared += block_squared_norm(nx, columns);

        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            const int next_primal =
                stage + 1 < horizon ? (stage + 1) * stage_dim : terminal;
            const int dual = n_primal + (stage + 1) * nx;
            blasfeo_dgecp(nx, columns, rhs_workspace.matrix(), dual, 0,
                          block_workspace.matrix(), 0, 0);
            blasfeo_dgead(nx, columns, -1.0, input_workspace.matrix(),
                          next_primal, 0, block_workspace.matrix(), 0, 0);
            blasfeo_dgemm_nn(
                nx, columns, stage_dim, -1.0,
                packed_dynamics[stage].matrix(), 0, 0,
                input_workspace.matrix(), primal, 0, 1.0,
                block_workspace.matrix(), 0, 0, block_workspace.matrix(), 0,
                0);
            residual_squared += block_squared_norm(nx, columns);
        }

        return std::sqrt(std::max(0.0, residual_squared)) /
               std::max(norm(rhs), 1e-300);
    }

private:
    void prepare(int columns, bool need_input, bool need_rhs)
    {
        if (prepared_block_columns == columns &&
            (!need_input || prepared_input_columns == columns) &&
            (!need_rhs || prepared_rhs_columns == columns))
            return;
        ExcludeAblationPreparation excluded;
        if (prepared_block_columns != columns)
        {
            block_workspace.reset(stage_dim, columns);
            block_buffer.resize(static_cast<std::size_t>(stage_dim) * columns);
            prepared_block_columns = columns;
        }
        if (need_input && prepared_input_columns != columns)
        {
            input_workspace.reset(dimension, columns);
            prepared_input_columns = columns;
        }
        if (need_rhs && prepared_rhs_columns != columns)
        {
            rhs_workspace.reset(dimension, columns);
            prepared_rhs_columns = columns;
        }
    }

    void unpack_block(int rows, int columns, double *output)
    {
        blasfeo_unpack_tran_dmat(rows, columns, block_workspace.matrix(), 0, 0,
                                 output, columns);
    }

    double block_squared_norm(int rows, int columns)
    {
        unpack_block(rows, columns, block_buffer.data());
        double value = 0.0;
        const std::size_t size = static_cast<std::size_t>(rows) * columns;
        for (std::size_t index = 0; index < size; ++index)
            value += block_buffer[index] * block_buffer[index];
        return value;
    }

    const int horizon;
    const int nx;
    const int nu;
    const int stage_dim;
    const int n_primal;
    const int dimension;
    std::vector<PackedMatrix> packed_hessians;
    std::vector<PackedMatrix> packed_dynamics;
    PackedMatrix packed_terminal_hessian;
    PackedMatrix input_workspace;
    PackedMatrix rhs_workspace;
    PackedMatrix block_workspace;
    std::vector<double> block_buffer;
    int prepared_block_columns = 0;
    int prepared_input_columns = 0;
    int prepared_rhs_columns = 0;
};

class ExactDifferenceOperator
{
public:
    ExactDifferenceOperator(int horizon, int nx, int nu,
                            const double *stage_hessians,
                            const double *terminal_hessian,
                            const double *dynamics)
        : dimension(horizon * (nx + nu) + nx + (horizon + 1) * nx)
    {
        const int stage_dim = nx + nu;
        const int n_primal = horizon * stage_dim + nx;
        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dim;
            const int multiplier = n_primal + (stage + 1) * nx;
            const double *hessian = stage_hessians +
                static_cast<std::size_t>(stage) * stage_dim * stage_dim;
            for (int row = 0; row < stage_dim; ++row)
                for (int column = 0; column < stage_dim; ++column)
                {
                    const double value = hessian[
                        static_cast<std::size_t>(row) * stage_dim + column];
                    if (value != 0.0)
                        entries.push_back({primal + row, primal + column, value});
                }

            const double *dynamics_block = dynamics +
                static_cast<std::size_t>(stage) * nx * stage_dim;
            for (int row = 0; row < nx; ++row)
                for (int column = 0; column < stage_dim; ++column)
                {
                    const double value = dynamics_block[
                        static_cast<std::size_t>(row) * stage_dim + column];
                    if (value == 0.0)
                        continue;
                    entries.push_back(
                        {multiplier + row, primal + column, value});
                    entries.push_back(
                        {primal + column, multiplier + row, value});
                }
        }

        const int terminal = horizon * stage_dim;
        for (int row = 0; row < nx; ++row)
            for (int column = 0; column < nx; ++column)
            {
                const double value = terminal_hessian[
                    static_cast<std::size_t>(row) * nx + column];
                if (value != 0.0)
                    entries.push_back({terminal + row, terminal + column, value});
            }

        std::stable_sort(
            entries.begin(), entries.end(),
            [](const Entry &left, const Entry &right) {
                return left.destination < right.destination;
            });
        row_offsets.assign(static_cast<std::size_t>(dimension) + 1, 0);
        for (const Entry &entry : entries)
            ++row_offsets[static_cast<std::size_t>(entry.destination) + 1];
        for (int row = 0; row < dimension; ++row)
            row_offsets[static_cast<std::size_t>(row) + 1] +=
                row_offsets[row];
    }

    bool apply(const std::vector<double> &input, int columns,
               std::vector<double> &output,
               const PackedMatrix * = nullptr,
               const std::vector<double> *rhs_for_norms = nullptr,
               std::vector<double> *output_squared_norms = nullptr,
               std::vector<double> *rhs_squared_norms = nullptr,
               double output_scale = 1.0)
    {
        if (columns <= 0 ||
            input.size() != static_cast<std::size_t>(dimension) * columns)
            return false;
        const bool collect_norms = rhs_for_norms != nullptr;
        if (collect_norms &&
            (rhs_for_norms->size() != input.size() ||
             output_squared_norms == nullptr || rhs_squared_norms == nullptr))
            return false;
        if (collect_norms)
        {
            output_squared_norms->assign(columns, 0.0);
            rhs_squared_norms->assign(columns, 0.0);
        }

        output.resize(input.size());
        for (int row = 0; row < dimension; ++row)
        {
            double *target = output.data() +
                static_cast<std::size_t>(row) * columns;
            std::fill_n(target, columns, 0.0);
            for (std::size_t index = row_offsets[row];
                 index < row_offsets[static_cast<std::size_t>(row) + 1];
                 ++index)
            {
                const Entry &entry = entries[index];
                axpy_raw(
                    entry.value,
                    input.data() +
                        static_cast<std::size_t>(entry.source) * columns,
                    target, columns);
            }

            if (collect_norms)
            {
                const double *rhs_row = rhs_for_norms->data() +
                    static_cast<std::size_t>(row) * columns;
                for (int column = 0; column < columns; ++column)
                {
                    target[column] *= output_scale;
                    (*output_squared_norms)[column] +=
                        target[column] * target[column];
                    (*rhs_squared_norms)[column] +=
                        rhs_row[column] * rhs_row[column];
                }
            }
            else if (output_scale != 1.0)
                for (int column = 0; column < columns; ++column)
                    target[column] *= output_scale;
        }
        return true;
    }

private:
    struct Entry
    {
        int destination;
        int source;
        double value;
    };

    int dimension;
    std::vector<Entry> entries;
    std::vector<std::size_t> row_offsets;
};

bool apply_right_preconditioned(
    BatchedKktOperator &target_operator,
    ExactDifferenceOperator *difference_operator,
    const std::vector<double> &input,
    const std::vector<double> &preconditioned, int columns,
    std::vector<double> &output, const PackedMatrix *packed_preconditioned)
{
    if (difference_operator == nullptr)
        return target_operator.apply(preconditioned, columns, output,
                                     packed_preconditioned);
    if (!difference_operator->apply(preconditioned, columns, output,
                                    packed_preconditioned) ||
        output.size() != input.size())
        return false;
    for (std::size_t index = 0; index < output.size(); ++index)
        output[index] += input[index];
    return true;
}

struct GmresWorkspace
{
    void prepare(std::size_t size, int max_iterations, int columns)
    {
        ExcludeAblationPreparation excluded;
        basis.resize(static_cast<std::size_t>(max_iterations) + 1);
        preconditioned.resize(max_iterations);
        for (auto &vector : basis)
            vector.resize(size);
        for (auto &vector : preconditioned)
            vector.resize(size);
        h.assign(static_cast<std::size_t>(max_iterations + 1) *
                     max_iterations * columns,
                 0.0);
        c.assign(static_cast<std::size_t>(max_iterations) * columns, 0.0);
        s.assign(static_cast<std::size_t>(max_iterations) * columns, 0.0);
        g.assign(static_cast<std::size_t>(max_iterations + 1) * columns,
                 0.0);
        y.assign(static_cast<std::size_t>(max_iterations) * columns, 0.0);
        beta.assign(columns, 0.0);
        coefficient.assign(columns, 0.0);
        used.assign(columns, 0);
        done.assign(columns, 0);
    }

    void prepare_global_lazy(std::size_t size, int max_iterations)
    {
        basis.resize(static_cast<std::size_t>(max_iterations) + 1);
        preconditioned.resize(max_iterations);
        basis[0].resize(size);
        h.assign(static_cast<std::size_t>(max_iterations + 1) *
                     max_iterations,
                 0.0);
        c.assign(max_iterations, 0.0);
        s.assign(max_iterations, 0.0);
        g.assign(static_cast<std::size_t>(max_iterations) + 1, 0.0);
        y.assign(max_iterations, 0.0);
        beta.assign(1, 0.0);
        coefficient.assign(1, 0.0);
        used.assign(1, 0);
        done.assign(1, 0);
    }

    std::vector<std::vector<double>> basis;
    std::vector<std::vector<double>> preconditioned;
    std::vector<double> h;
    std::vector<double> c;
    std::vector<double> s;
    std::vector<double> g;
    std::vector<double> y;
    std::vector<double> beta;
    std::vector<double> coefficient;
    std::vector<int> used;
    std::vector<unsigned char> done;
    std::vector<double> difference_scratch;
    std::vector<double> global_h_raw;
    std::vector<double> global_y;
    std::vector<double> active_rhs;
    std::vector<double> active_solution;
    std::vector<double> active_residual;
    std::vector<double> initial_active_residual;
    std::vector<double> reconstructed_residual;
    std::vector<double> correction_rhs;
    std::vector<double> correction_solution;
    std::vector<double> residual_scales;
    std::vector<double> rhs_denominators;
    std::vector<double> column_errors;
    std::vector<double> column_residual_norms;
    std::vector<double> column_rhs_denominators;
    std::vector<int> active_columns;
    std::vector<int> failed_columns;
    std::vector<double> compressed_remainder;
    std::vector<double> compressed_basis;
    std::vector<double> compressed_coefficients;
    std::vector<double> compressed_weights;
    std::vector<double> compressed_rhs;
    std::vector<double> compressed_solution;
    std::vector<double> compressed_image;
    std::vector<double> compressed_norms;
    std::vector<double> compressed_candidate;
    std::vector<double> compressed_errors;
};

struct RecycleSpace
{
    int rows = 0;
    int rank = 0;
    std::vector<double> u;
    std::vector<double> z;
    std::vector<double> c;
    std::vector<double> gamma;
};

struct ForwardData
{
    ForwardData(int horizon, int nx, int nu, const double *h_forward,
                const double *h_forward_terminal, const double *a_blocks)
        : h_forward(h_forward,
                    h_forward + static_cast<std::size_t>(horizon) *
                                    (nx + nu) * (nx + nu)),
          h_forward_terminal(h_forward_terminal,
                             h_forward_terminal +
                                 static_cast<std::size_t>(nx) * nx),
          a_blocks(a_blocks,
                   a_blocks + static_cast<std::size_t>(horizon) * nx *
                                  (nx + nu))
    {
    }

    ForwardData(int horizon, int nx, int nu)
        : h_forward(static_cast<std::size_t>(horizon) * (nx + nu) *
                        (nx + nu),
                    0.0),
          h_forward_terminal(static_cast<std::size_t>(nx) * nx, 0.0),
          a_blocks(static_cast<std::size_t>(horizon) * nx * (nx + nu),
                   0.0),
          has_forward_matrix(false)
    {
    }

    std::vector<double> h_forward;
    std::vector<double> h_forward_terminal;
    std::vector<double> a_blocks;
    bool has_forward_matrix = true;
};

class PreparedSolver
{
public:
    PreparedSolver(int horizon, int nx, int nu, const double *h_forward,
                   const double *h_forward_terminal, const double *a_blocks)
        : data(std::make_shared<ForwardData>(
              horizon, nx, nu, h_forward, h_forward_terminal, a_blocks)),
          factorization(std::make_shared<CachedForwardFactorization>(
              horizon, nx, nu, data->h_forward.data(),
              data->h_forward_terminal.data(), data->a_blocks.data())),
          factor_workspace(horizon)
    {
        factorization->prepare_workspace(factor_workspace);
    }

    PreparedSolver(int horizon, int nx, int nu,
                   FrsCachedSolve cached_solve, void *context)
        : data(std::make_shared<ForwardData>(horizon, nx, nu)),
          factorization(std::make_shared<CachedForwardFactorization>(
              horizon, nx, nu, cached_solve, context)),
          factor_workspace(horizon)
    {
        factorization->prepare_workspace(factor_workspace);
    }

    PreparedSolver(int horizon, int nx, int nu, const double *h_forward,
                   const double *h_forward_terminal, const double *a_blocks,
                   const double *value_hessians,
                   const double *control_cholesky,
                   const double *cross_factors)
        : data(std::make_shared<ForwardData>(
              horizon, nx, nu, h_forward, h_forward_terminal, a_blocks)),
          factorization(std::make_shared<CachedForwardFactorization>(
              horizon, nx, nu, data->h_forward.data(),
              data->h_forward_terminal.data(), data->a_blocks.data(),
              value_hessians, control_cholesky, cross_factors)),
          factor_workspace(horizon)
    {
        factorization->prepare_workspace(factor_workspace);
    }

    PreparedSolver(std::shared_ptr<ForwardData> data,
                   std::shared_ptr<CachedForwardFactorization> factor)
        : data(std::move(data)), factorization(std::move(factor)),
          factor_workspace(factorization->horizon)
    {
        factorization->prepare_workspace(factor_workspace);
    }

    std::shared_ptr<ForwardData> data;
    std::shared_ptr<CachedForwardFactorization> factorization;
    FactorWorkspace factor_workspace;
    GmresWorkspace gmres_workspace;
    std::vector<double> all_rhs;
    std::vector<double> all_solution;
    std::vector<double> target_hessian_workspace;
    std::vector<double> target_terminal_hessian_workspace;
    std::vector<double> target_dynamics_workspace;
    std::vector<double> initial_residual;
    std::vector<double> rhs_chunk;
    std::vector<double> solution_chunk;
    std::vector<double> residual_chunk;
    std::vector<double> h_difference;
    std::vector<double> terminal_difference;
    std::vector<double> a_difference;
};

class PersistentExecutor
{
public:
    explicit PersistentExecutor(int capacity) : capacity_(capacity)
    {
        for (int worker = 1; worker < capacity_; ++worker)
            threads_.emplace_back([this, worker]() { worker_loop(worker); });
    }

    ~PersistentExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto &thread : threads_)
            thread.join();
    }

    int capacity() const { return capacity_; }

    void run(int active, const std::function<void(int)> &job)
    {
        std::lock_guard<std::mutex> run_lock(run_mutex_);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            active_ = active;
            remaining_ = active - 1;
            job_ = job;
            ++generation_;
        }
        start_cv_.notify_all();
        job(0);
        if (active > 1)
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            done_cv_.wait(lock, [this]() { return remaining_ == 0; });
            job_ = {};
        }
    }

private:
    void worker_loop(int worker)
    {
        std::size_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(state_mutex_);
        while (true)
        {
            start_cv_.wait(lock, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_)
                return;
            observed_generation = generation_;
            const int active = active_;
            const auto job = job_;
            lock.unlock();
            if (worker < active)
                job(worker);
            lock.lock();
            if (worker < active && --remaining_ == 0)
                done_cv_.notify_one();
        }
    }

    int capacity_;
    std::vector<std::thread> threads_;
    std::mutex run_mutex_;
    std::mutex state_mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::function<void(int)> job_;
    std::size_t generation_ = 0;
    int active_ = 0;
    int remaining_ = 0;
    bool stopping_ = false;
};

thread_local std::unique_ptr<PersistentExecutor> parallel_executor;

bool gmres(const CachedForwardFactorization &factor, FactorWorkspace &factor_workspace,
           BatchedKktOperator &target_operator,
           ExactDifferenceOperator *difference_operator,
           const std::vector<double> &rhs, int columns, double tolerance,
           int max_iterations, TimingProfile &timing,
           std::vector<double> &solution, double &iterations, double &residual,
           GmresWorkspace &workspace, bool use_preconditioner = true)
{
    if (columns <= 0 || rhs.size() % static_cast<std::size_t>(columns) != 0)
        return false;
    const std::size_t rows = rhs.size() / columns;
    {
        ExcludeAblationPreparation excluded;
        workspace.prepare(rhs.size(), max_iterations, columns);
        solution.reserve(rhs.size());
        timing.residual_history.reserve(static_cast<std::size_t>(max_iterations) + 1);
    }
    auto &basis = workspace.basis;
    auto &preconditioned_basis = workspace.preconditioned;
    auto &h = workspace.h;
    auto &c = workspace.c;
    auto &s = workspace.s;
    auto &g = workspace.g;
    auto &y = workspace.y;
    auto &beta = workspace.beta;
    auto &coefficient = workspace.coefficient;
    auto &used = workspace.used;
    auto &done = workspace.done;
    auto &difference_scratch = workspace.difference_scratch;
    std::copy(rhs.begin(), rhs.end(), basis[0].begin());
    for (std::size_t row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            beta[column] = std::hypot(
                beta[column], basis[0][row * columns + column]);

    int remaining = 0;
    std::vector<double> latest_estimate;
    resize_solve_workspace(latest_estimate, columns);
    for (int column = 0; column < columns; ++column)
    {
        g[column] = beta[column];
        if (beta[column] == 0.0)
        {
            done[column] = 1;
            continue;
        }
        ++remaining;
        const double inverse = 1.0 / beta[column];
        for (std::size_t row = 0; row < rows; ++row)
            basis[0][row * columns + column] *= inverse;
    }

    auto H = [&](int row, int iteration, int column) -> double & {
        return h[(static_cast<std::size_t>(row) * max_iterations +
                  iteration) * columns + column];
    };
    auto C = [&](int iteration, int column) -> double & {
        return c[static_cast<std::size_t>(iteration) * columns + column];
    };
    auto S = [&](int iteration, int column) -> double & {
        return s[static_cast<std::size_t>(iteration) * columns + column];
    };
    auto G = [&](int row, int column) -> double & {
        return g[static_cast<std::size_t>(row) * columns + column];
    };
    auto Y = [&](int row, int column) -> double & {
        return y[static_cast<std::size_t>(row) * columns + column];
    };
    for (int column = 0; column < columns; ++column)
        if (beta[column] > 0.0)
            latest_estimate[column] = 1.0;
    const auto record_residual = [&](int iteration, double estimate) {
        const std::size_t index = static_cast<std::size_t>(iteration);
        if (timing.residual_history.size() <= index)
            timing.residual_history.resize(index + 1, 0.0);
        timing.residual_history[index] =
            std::max(timing.residual_history[index], estimate);
    };
    record_residual(0, remaining > 0 ? 1.0 : 0.0);

    for (int k = 0; k < max_iterations && remaining > 0; ++k)
    {
        auto &recovered = preconditioned_basis[k];
        auto &candidate = basis[k + 1];
        const auto difference_started = ProfileClock::now();
        if (use_preconditioner)
        {
            if (!factor.solve(basis[k], columns, recovered, timing,
                              factor_workspace))
                return false;
            if (!apply_right_preconditioned(
                    target_operator, difference_operator, basis[k], recovered,
                    columns, candidate, packed_solution(factor_workspace)))
                return false;
        }
        else if (!target_operator.apply(
                     basis[k], columns, candidate,
                     packed_solution(factor_workspace)))
        {
            return false;
        }
        timing.difference_matvec += profile_seconds(difference_started);
        const auto arnoldi_started = ProfileClock::now();

        for (std::size_t row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                if (done[column])
                    candidate[row * columns + column] = 0.0;

        // Two-pass modified Gram-Schmidt keeps the independent Krylov
        // recurrences stable for strongly active bounded-LQR directions.
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int j = 0; j <= k; ++j)
            {
                std::fill(coefficient.begin(), coefficient.end(), 0.0);
                for (std::size_t row = 0; row < rows; ++row)
                    for (int column = 0; column < columns; ++column)
                        if (!done[column])
                            coefficient[column] +=
                                basis[j][row * columns + column] *
                                candidate[row * columns + column];
                for (int column = 0; column < columns; ++column)
                    if (!done[column])
                        H(j, k, column) += coefficient[column];
                for (std::size_t row = 0; row < rows; ++row)
                    for (int column = 0; column < columns; ++column)
                        if (!done[column])
                            candidate[row * columns + column] -=
                                coefficient[column] *
                                basis[j][row * columns + column];
            }
        }

        std::fill(coefficient.begin(), coefficient.end(), 0.0);
        for (std::size_t row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                if (!done[column])
                    coefficient[column] = std::hypot(
                        coefficient[column],
                        candidate[row * columns + column]);
        for (int column = 0; column < columns; ++column)
        {
            if (done[column])
                continue;
            H(k + 1, k, column) = coefficient[column];
            if (coefficient[column] > 1e-15)
            {
                const double inverse = 1.0 / coefficient[column];
                for (std::size_t row = 0; row < rows; ++row)
                    candidate[row * columns + column] *= inverse;
            }
        }

        for (int j = 0; j < k; ++j)
            for (int column = 0; column < columns; ++column)
            {
                if (done[column])
                    continue;
                const double top =
                    C(j, column) * H(j, k, column) +
                    S(j, column) * H(j + 1, k, column);
                H(j + 1, k, column) =
                    -S(j, column) * H(j, k, column) +
                    C(j, column) * H(j + 1, k, column);
                H(j, k, column) = top;
            }

        for (int column = 0; column < columns; ++column)
        {
            if (done[column])
                continue;
            const bool breakdown = H(k + 1, k, column) <= 1e-15;
            const double rho =
                std::hypot(H(k, k, column), H(k + 1, k, column));
            C(k, column) = rho == 0.0 ? 1.0 : H(k, k, column) / rho;
            S(k, column) =
                rho == 0.0 ? 0.0 : H(k + 1, k, column) / rho;
            H(k, k, column) = rho;
            H(k + 1, k, column) = 0.0;
            G(k + 1, column) = -S(k, column) * G(k, column);
            G(k, column) *= C(k, column);
            used[column] = k + 1;
            const double estimate =
                std::abs(G(k + 1, column)) / beta[column];
            latest_estimate[column] = estimate;
            if (estimate <= tolerance || breakdown)
            {
                done[column] = 1;
                --remaining;
                for (std::size_t row = 0; row < rows; ++row)
                    candidate[row * columns + column] = 0.0;
            }
        }
        record_residual(
            k + 1,
            *std::max_element(latest_estimate.begin(), latest_estimate.end()));
        timing.arnoldi += profile_seconds(arnoldi_started);
    }

    for (int column = 0; column < columns; ++column)
    {
        for (int row = used[column] - 1; row >= 0; --row)
        {
            double value = G(row, column);
            for (int inner = row + 1; inner < used[column]; ++inner)
                value -= H(row, inner, column) * Y(inner, column);
            if (std::abs(H(row, row, column)) < 1e-30)
                return false;
            Y(row, column) = value / H(row, row, column);
        }
    }

    const auto reduced_rhs_started = ProfileClock::now();
    solution.assign(rhs.size(), 0.0);
    for (int column = 0; column < columns; ++column)
        for (int j = 0; j < used[column]; ++j)
            for (std::size_t row = 0; row < rows; ++row)
            {
                const auto &search_direction = use_preconditioner
                    ? preconditioned_basis[j]
                    : basis[j];
                solution[row * columns + column] +=
                    Y(j, column) *
                    search_direction[row * columns + column];
            }
    timing.arnoldi += profile_seconds(reduced_rhs_started);

    iterations = 0.0;
    residual = 0.0;
    bool converged = true;
    for (int column = 0; column < columns; ++column)
    {
        iterations += used[column];
        const double estimate = beta[column] == 0.0
            ? 0.0
            : std::abs(G(used[column], column)) / beta[column];
        residual = std::max(residual, estimate);
        converged = converged && std::isfinite(estimate) &&
                    estimate <= tolerance;
    }
    iterations /= columns;
    return converged;
}

double contiguous_dot(const std::vector<double> &left,
                      const std::vector<double> &right)
{
    const std::size_t size = left.size();
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    std::size_t index = 0;
    for (; index + 4 <= size; index += 4)
    {
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }
    double sum = (sum0 + sum1) + (sum2 + sum3);
    for (; index < size; ++index)
        sum += left[index] * right[index];
    return sum;
}

double contiguous_squared_norm(const std::vector<double> &values)
{
    return contiguous_dot(values, values);
}

double column_norm(const std::vector<double> &matrix, int rows, int columns,
                   int column)
{
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    int row = 0;
    for (; row + 4 <= rows; row += 4)
    {
        const double value0 = matrix[
            static_cast<std::size_t>(row) * columns + column];
        const double value1 = matrix[
            static_cast<std::size_t>(row + 1) * columns + column];
        const double value2 = matrix[
            static_cast<std::size_t>(row + 2) * columns + column];
        const double value3 = matrix[
            static_cast<std::size_t>(row + 3) * columns + column];
        sum0 += value0 * value0;
        sum1 += value1 * value1;
        sum2 += value2 * value2;
        sum3 += value3 * value3;
    }
    double squared = (sum0 + sum1) + (sum2 + sum3);
    for (; row < rows; ++row)
    {
        const double value = matrix[
            static_cast<std::size_t>(row) * columns + column];
        squared += value * value;
    }
    if (!std::isfinite(squared))
        return std::numeric_limits<double>::infinity();
    return std::sqrt(std::max(0.0, squared));
}

double column_relative_residuals(
    const std::vector<double> &residual_vector,
    const std::vector<double> &rhs, int columns,
    std::vector<double> &errors)
{
    const int rows = static_cast<int>(rhs.size()) / columns;
    resize_solve_workspace(errors, columns);
    double maximum = 0.0;
    for (int column = 0; column < columns; ++column)
    {
        const double rhs_norm = column_norm(rhs, rows, columns, column);
        const double denominator = rhs_norm > 1e-14 ? rhs_norm : 1.0;
        errors[column] =
            column_norm(residual_vector, rows, columns, column) / denominator;
        maximum = std::max(maximum, errors[column]);
    }
    return maximum;
}

bool explicit_column_residuals(
    BatchedKktOperator &target_operator, const std::vector<double> &rhs,
    const std::vector<double> &solution, int columns,
    std::vector<double> &residual_vector, std::vector<double> &errors,
    double &maximum, TimingProfile &timing,
    std::vector<double> *residual_norms = nullptr,
    std::vector<double> *rhs_denominators = nullptr,
    std::vector<int> *active_columns = nullptr,
    double active_tolerance = std::numeric_limits<double>::infinity(),
    const PackedMatrix *packed_solution = nullptr)
{
    const auto started = ProfileClock::now();
    std::vector<double> rhs_squared_norms;
    resize_solve_workspace(rhs_squared_norms, columns);
    TimingProfile *detail_profile =
        kDetailedResidualProfile ? &timing : nullptr;
    if (!target_operator.apply(
            solution, columns, residual_vector, packed_solution,
            detail_profile, &rhs, &errors, &rhs_squared_norms))
        return false;
    {
        ExcludeAblationPreparation excluded;
        if (residual_norms != nullptr)
            residual_norms->assign(columns, 0.0);
        if (rhs_denominators != nullptr)
            rhs_denominators->assign(columns, 0.0);
        if (active_columns != nullptr)
        {
            active_columns->clear();
            active_columns->reserve(columns);
        }
    }
    ProfileClock::time_point finalize_started;
    if (detail_profile != nullptr)
        finalize_started = ProfileClock::now();
    maximum = 0.0;
    for (int column = 0; column < columns; ++column)
    {
        const double residual_norm = std::sqrt(errors[column]);
        const double rhs_norm = std::sqrt(rhs_squared_norms[column]);
        const double denominator = rhs_norm > 1e-14 ? rhs_norm : 1.0;
        if (residual_norms != nullptr)
            (*residual_norms)[column] = residual_norm;
        if (rhs_denominators != nullptr)
            (*rhs_denominators)[column] = denominator;
        errors[column] = residual_norm / denominator;
        if (active_columns != nullptr && errors[column] > active_tolerance)
            active_columns->push_back(column);
        maximum = std::max(maximum, errors[column]);
    }
    if (detail_profile != nullptr)
        timing.residual_subtract_norm += profile_seconds(finalize_started);
    timing.residual += profile_seconds(started);
    return std::isfinite(maximum);
}

bool build_recycle_space(
    const CachedForwardFactorization &factor, FactorWorkspace &factor_workspace,
    BatchedKktOperator &target_operator,
    ExactDifferenceOperator *difference_operator,
    const GmresWorkspace &workspace,
    int rows, int active, int global_steps, int requested_rank,
    TimingProfile &timing, RecycleSpace &recycle)
{
    if (requested_rank <= 0 || global_steps <= 0 || active <= 0)
        return true;

    const auto projection_started = ProfileClock::now();
    std::vector<std::vector<double>> u_columns;
    u_columns.reserve(requested_rank);
    for (int local = 0;
         local < active && static_cast<int>(u_columns.size()) < requested_rank;
         ++local)
    {
        for (int basis_index = 0;
             basis_index <= global_steps &&
             static_cast<int>(u_columns.size()) < requested_rank;
             ++basis_index)
        {
            std::vector<double> candidate(rows);
            for (int row = 0; row < rows; ++row)
                candidate[row] = workspace.basis[basis_index][
                    static_cast<std::size_t>(row) * active + local];
            const double original_norm = std::sqrt(std::max(
                0.0, contiguous_squared_norm(candidate)));
            if (!(original_norm > 0.0))
                continue;
            for (int pass = 0; pass < 2; ++pass)
                for (const auto &column : u_columns)
                    axpy_raw(-contiguous_dot(column, candidate),
                             column.data(), candidate.data(), rows);
            const double norm = std::sqrt(std::max(
                0.0, contiguous_squared_norm(candidate)));
            if (norm <= 1e-12 * original_norm)
                continue;
            const double inverse = 1.0 / norm;
            for (double &value : candidate)
                value *= inverse;
            u_columns.push_back(std::move(candidate));
        }
    }
    timing.projection += profile_seconds(projection_started);
    if (u_columns.empty())
        return true;

    int rank = static_cast<int>(u_columns.size());
    recycle.u.resize(static_cast<std::size_t>(rows) * rank);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < rank; ++column)
            recycle.u[static_cast<std::size_t>(row) * rank + column] =
                u_columns[column][row];

    if (!factor.solve(recycle.u, rank, recycle.z, timing, factor_workspace))
        return false;
    const auto operator_started = ProfileClock::now();
    if (!apply_right_preconditioned(
            target_operator, difference_operator, recycle.u, recycle.z, rank,
            recycle.c, packed_solution(factor_workspace)))
        return false;
    timing.difference_matvec += profile_seconds(operator_started);

    const auto orthogonalize_started = ProfileClock::now();
    std::vector<std::vector<double>> kept_u;
    std::vector<std::vector<double>> kept_z;
    std::vector<std::vector<double>> kept_c;
    kept_u.reserve(rank);
    kept_z.reserve(rank);
    kept_c.reserve(rank);
    for (int column = 0; column < rank; ++column)
    {
        std::vector<double> u_column(rows);
        std::vector<double> z_column(rows);
        std::vector<double> c_column(rows);
        for (int row = 0; row < rows; ++row)
        {
            const std::size_t index =
                static_cast<std::size_t>(row) * rank + column;
            u_column[row] = recycle.u[index];
            z_column[row] = recycle.z[index];
            c_column[row] = recycle.c[index];
        }
        const double original_norm = std::sqrt(std::max(
            0.0, contiguous_squared_norm(c_column)));
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t kept = 0; kept < kept_c.size(); ++kept)
            {
                const double coefficient =
                    contiguous_dot(kept_c[kept], c_column);
                axpy_raw(-coefficient, kept_c[kept].data(),
                         c_column.data(), rows);
                axpy_raw(-coefficient, kept_u[kept].data(),
                         u_column.data(), rows);
                axpy_raw(-coefficient, kept_z[kept].data(),
                         z_column.data(), rows);
            }
        const double norm = std::sqrt(std::max(
            0.0, contiguous_squared_norm(c_column)));
        if (!(norm > 1e-12 * std::max(1.0, original_norm)))
            continue;
        const double inverse = 1.0 / norm;
        for (int row = 0; row < rows; ++row)
        {
            c_column[row] *= inverse;
            u_column[row] *= inverse;
            z_column[row] *= inverse;
        }
        kept_c.push_back(std::move(c_column));
        kept_u.push_back(std::move(u_column));
        kept_z.push_back(std::move(z_column));
    }

    recycle.rows = rows;
    recycle.rank = static_cast<int>(kept_c.size());
    recycle.u.resize(static_cast<std::size_t>(rows) * recycle.rank);
    recycle.z.resize(static_cast<std::size_t>(rows) * recycle.rank);
    recycle.c.resize(static_cast<std::size_t>(rows) * recycle.rank);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < recycle.rank; ++column)
        {
            const std::size_t index =
                static_cast<std::size_t>(row) * recycle.rank + column;
            recycle.u[index] = kept_u[column][row];
            recycle.z[index] = kept_z[column][row];
            recycle.c[index] = kept_c[column][row];
        }
    timing.projection += profile_seconds(orthogonalize_started);
    return recycle.rank > 0;
}

bool apply_recycle_projection(
    RecycleSpace &recycle, std::vector<double> &residual,
    std::vector<double> &solution, int columns, TimingProfile &timing)
{
    if (recycle.rank <= 0)
        return true;
    const auto started = ProfileClock::now();
    const int rows = recycle.rows;
    const int rank = recycle.rank;
    recycle.gamma.assign(static_cast<std::size_t>(rank) * columns, 0.0);
    for (int direction = 0; direction < rank; ++direction)
        for (int column = 0; column < columns; ++column)
        {
            double coefficient = 0.0;
            for (int row = 0; row < rows; ++row)
                coefficient +=
                    recycle.c[static_cast<std::size_t>(row) * rank + direction] *
                    residual[static_cast<std::size_t>(row) * columns + column];
            recycle.gamma[static_cast<std::size_t>(direction) * columns + column] =
                coefficient;
        }
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
        {
            double residual_update = 0.0;
            double solution_update = 0.0;
            for (int direction = 0; direction < rank; ++direction)
            {
                const double coefficient = recycle.gamma[
                    static_cast<std::size_t>(direction) * columns + column];
                residual_update += recycle.c[
                    static_cast<std::size_t>(row) * rank + direction] * coefficient;
                solution_update += recycle.z[
                    static_cast<std::size_t>(row) * rank + direction] * coefficient;
            }
            residual[static_cast<std::size_t>(row) * columns + column] -=
                residual_update;
            solution[static_cast<std::size_t>(row) * columns + column] +=
                solution_update;
        }
    timing.projection += profile_seconds(started);
    return std::all_of(residual.begin(), residual.end(),
                       [](double value) { return std::isfinite(value); });
}

/* Compress the normalized multi-RHS residual before applying another cached
   factor solve.  A column-pivoted modified Gram-Schmidt pass constructs

       R D^{-1} = U C + E,

   where every column of E is below one percent of the requested solver
   tolerance.  The correction therefore needs only rank(U) cached-factor
   right-hand sides instead of all columns.  The residual is updated with the
   exact target operator; the caller remains responsible for the final
   explicit certification when requested. */
bool rank_compressed_residual_correction(
    const CachedForwardFactorization &factor,
    FactorWorkspace &factor_workspace,
    BatchedKktOperator &target_operator,
    const std::vector<double> &rhs_denominators,
    int rows, int columns, double tolerance, TimingProfile &timing,
    std::vector<double> &solution, std::vector<double> &residual_vector,
    GmresWorkspace &workspace, int &compressed_rank,
    double &maximum_residual)
{
    compressed_rank = 0;
    maximum_residual = std::numeric_limits<double>::infinity();
    if (columns <= 1 || rows <= 0 ||
        solution.size() != residual_vector.size() ||
        residual_vector.size() != static_cast<std::size_t>(rows) * columns ||
        rhs_denominators.size() != static_cast<std::size_t>(columns))
        return columns <= 1;

    const std::size_t size = residual_vector.size();
    const auto workspace_started = ProfileClock::now();
    workspace.compressed_remainder.resize(size);
    workspace.compressed_basis.resize(
        static_cast<std::size_t>(rows) * columns);
    workspace.compressed_coefficients.resize(
        static_cast<std::size_t>(columns) * columns);
    workspace.compressed_norms.assign(columns, 0.0);
    workspace.compressed_candidate.resize(rows);
    workspace.compressed_errors.assign(columns, 0.0);
    timing.workspace += profile_seconds(workspace_started);

    auto &remainder = workspace.compressed_remainder;
    auto &basis = workspace.compressed_basis;
    auto &coefficients = workspace.compressed_coefficients;
    auto &norms = workspace.compressed_norms;
    auto &candidate = workspace.compressed_candidate;
    const auto compression_started = ProfileClock::now();
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
        {
            const std::size_t index =
                static_cast<std::size_t>(row) * columns + column;
            const double value =
                residual_vector[index] / rhs_denominators[column];
            remainder[index] = value;
            norms[column] += value * value;
        }

    // At the solver's 1e-8 target, a 0.2-tolerance residual range is enough
    // to leave a full order of magnitude for the preconditioned correction.
    // Stopping early also avoids the loss of orthogonality seen when MGS was
    // incorrectly driven down to the numerical-rank floor.
    const double compression_tolerance = std::max(
        100.0 * std::numeric_limits<double>::epsilon(),
        0.2 * tolerance);
    while (compressed_rank < columns)
    {
        int pivot = 0;
        for (int column = 1; column < columns; ++column)
            if (norms[column] > norms[pivot])
                pivot = column;
        const double pivot_norm =
            std::sqrt(std::max(0.0, norms[pivot]));
        if (!(pivot_norm > compression_tolerance))
            break;

        for (int row = 0; row < rows; ++row)
            candidate[row] = remainder[
                static_cast<std::size_t>(row) * columns + pivot];
        const double candidate_norm = std::sqrt(std::max(
            0.0, contiguous_squared_norm(candidate)));
        if (!(candidate_norm > compression_tolerance))
            break;
        const double inverse = 1.0 / candidate_norm;
        for (int row = 0; row < rows; ++row)
            basis[static_cast<std::size_t>(row) * columns +
                  compressed_rank] = candidate[row] * inverse;

        double *coefficient = coefficients.data() +
            static_cast<std::size_t>(compressed_rank) * columns;
        std::fill_n(coefficient, columns, 0.0);
        for (int row = 0; row < rows; ++row)
        {
            const double left = basis[
                static_cast<std::size_t>(row) * columns + compressed_rank];
            const double *source = remainder.data() +
                static_cast<std::size_t>(row) * columns;
            for (int column = 0; column < columns; ++column)
                coefficient[column] += left * source[column];
        }
        std::fill(norms.begin(), norms.end(), 0.0);
        for (int row = 0; row < rows; ++row)
        {
            const double left = basis[
                static_cast<std::size_t>(row) * columns + compressed_rank];
            double *target = remainder.data() +
                static_cast<std::size_t>(row) * columns;
            for (int column = 0; column < columns; ++column)
            {
                target[column] -= left * coefficient[column];
                norms[column] += target[column] * target[column];
            }
        }
        ++compressed_rank;
    }
    timing.projection += profile_seconds(compression_started);

    // A full-rank representation would not reduce the cached-factor work.
    if (compressed_rank <= 0 || compressed_rank >= columns)
    {
        compressed_rank = 0;
        return true;
    }

    workspace.compressed_rhs.resize(
        static_cast<std::size_t>(rows) * compressed_rank);
    for (int row = 0; row < rows; ++row)
        for (int direction = 0; direction < compressed_rank; ++direction)
            workspace.compressed_rhs[
                static_cast<std::size_t>(row) * compressed_rank + direction] =
                basis[static_cast<std::size_t>(row) * columns + direction];

    if (!factor.solve(workspace.compressed_rhs, compressed_rank,
                      workspace.compressed_solution, timing,
                      factor_workspace))
        return false;
    const auto operator_started = ProfileClock::now();
    if (!target_operator.apply(
            workspace.compressed_solution, compressed_rank,
            workspace.compressed_image, packed_solution(factor_workspace)))
        return false;
    timing.difference_matvec += profile_seconds(operator_started);

    const auto update_started = ProfileClock::now();
    workspace.compressed_weights.resize(
        static_cast<std::size_t>(compressed_rank) * columns);
    for (int direction = 0; direction < compressed_rank; ++direction)
        for (int column = 0; column < columns; ++column)
            workspace.compressed_weights[
                static_cast<std::size_t>(direction) * columns + column] =
                coefficients[
                    static_cast<std::size_t>(direction) * columns + column] *
                rhs_denominators[column];
    apply_compressed_update(
        rows, columns, compressed_rank, workspace.compressed_weights.data(),
        workspace.compressed_solution.data(),
        workspace.compressed_image.data(), solution.data(),
        residual_vector.data(), workspace.compressed_errors.data());
    maximum_residual = 0.0;
    for (int column = 0; column < columns; ++column)
    {
        workspace.compressed_errors[column] =
            std::sqrt(workspace.compressed_errors[column]) /
            rhs_denominators[column];
        maximum_residual = std::max(
            maximum_residual, workspace.compressed_errors[column]);
    }
    timing.final_update += profile_seconds(update_started);
    return std::isfinite(maximum_residual);
}

bool global_gmres_certified(
    const CachedForwardFactorization &factor, FactorWorkspace &factor_workspace,
    BatchedKktOperator &target_operator,
    ExactDifferenceOperator *difference_operator,
    const std::vector<double> &rhs,
    int columns, double tolerance, int max_iterations, TimingProfile &timing,
    std::vector<double> &solution, double &iterations, double &residual,
    GmresWorkspace &workspace, RecycleSpace &recycle, int recycle_rank,
    bool certify_residual, const std::vector<double> *initial_residual,
    const std::vector<double> *initial_errors = nullptr,
    const std::vector<double> *initial_residual_norms = nullptr,
    const std::vector<double> *initial_rhs_denominators = nullptr,
    const std::vector<int> *initial_active_columns = nullptr,
    double initial_max_residual = std::numeric_limits<double>::infinity())
{
    if (columns <= 0 || rhs.size() % static_cast<std::size_t>(columns) != 0)
        return false;
    const int rows = static_cast<int>(rhs.size()) / columns;

    iterations = 0.0;
    bool use_initial_metrics = false;
    if (initial_residual != nullptr)
    {
        if (initial_residual->size() != rhs.size() ||
            solution.size() != rhs.size())
            return false;
        if (initial_residual != &workspace.active_residual)
        {
            const auto copy_started = ProfileClock::now();
            workspace.active_residual = *initial_residual;
            timing.input_copy += profile_seconds(copy_started);
        }
        use_initial_metrics =
            initial_errors != nullptr &&
            initial_residual_norms != nullptr &&
            initial_rhs_denominators != nullptr &&
            initial_errors->size() == static_cast<std::size_t>(columns) &&
            initial_residual_norms->size() ==
                static_cast<std::size_t>(columns) &&
            initial_rhs_denominators->size() ==
                static_cast<std::size_t>(columns) &&
            std::isfinite(initial_max_residual);
        if (use_initial_metrics)
        {
            if (initial_errors != &workspace.column_errors)
                workspace.column_errors = *initial_errors;
            residual = initial_max_residual;
        }
        else
        {
            const auto control_started = ProfileClock::now();
            residual = column_relative_residuals(
                workspace.active_residual, rhs, columns,
                workspace.column_errors);
            timing.control += profile_seconds(control_started);
        }
    }
    else
    {
        // Exact one-shot path. The residual is formed as B - K_B X_0, so this
        // remains valid even when the cached triangular solves have roundoff.
        if (!factor.solve(rhs, columns, solution, timing, factor_workspace))
            return false;
        if (!explicit_column_residuals(
                target_operator, rhs, solution, columns,
                workspace.active_residual, workspace.column_errors,
                residual, timing))
            return false;
    }
    if (recycle_rank > 0 && recycle.rank > 0)
    {
        if (!apply_recycle_projection(
                recycle, workspace.active_residual, solution, columns, timing))
            return false;
        residual = column_relative_residuals(
            workspace.active_residual, rhs, columns, workspace.column_errors);
        use_initial_metrics = false;
    }
    if (residual <= tolerance)
        return true;

    if (use_initial_metrics && initial_active_columns != nullptr)
    {
        if (initial_active_columns != &workspace.active_columns)
            workspace.active_columns = *initial_active_columns;
    }
    else
    {
        const auto column_scan_started = ProfileClock::now();
        workspace.active_columns.clear();
        for (int column = 0; column < columns; ++column)
            if (workspace.column_errors[column] > tolerance)
                workspace.active_columns.push_back(column);
        timing.control += profile_seconds(column_scan_started);
    }
    const int active = static_cast<int>(workspace.active_columns.size());
    if (active == 0)
        return true;

    const bool full_width = active == columns;
    const std::size_t active_size = static_cast<std::size_t>(rows) * active;
    const auto active_workspace_started = ProfileClock::now();
    {
        ExcludeAblationPreparation excluded;
        if (!full_width)
        {
            workspace.active_rhs.resize(active_size);
            workspace.active_solution.resize(active_size);
            workspace.initial_active_residual.resize(active_size);
        }
        workspace.residual_scales.resize(active);
        workspace.rhs_denominators.resize(active);
        workspace.failed_columns.reserve(columns);
    }
    timing.workspace += profile_seconds(active_workspace_started);
    const auto active_control_started = ProfileClock::now();
    if (!full_width)
        for (int row = 0; row < rows; ++row)
            for (int local = 0; local < active; ++local)
            {
                const int column = workspace.active_columns[local];
                const std::size_t source =
                    static_cast<std::size_t>(row) * columns + column;
                const std::size_t destination =
                    static_cast<std::size_t>(row) * active + local;
                workspace.active_rhs[destination] = rhs[source];
                workspace.active_solution[destination] = solution[source];
                workspace.initial_active_residual[destination] =
                    workspace.active_residual[source];
            }
    const std::vector<double> &active_rhs =
        full_width ? rhs : workspace.active_rhs;
    std::vector<double> &active_solution =
        full_width ? solution : workspace.active_solution;
    std::vector<double> &initial_active_residual =
        full_width ? workspace.active_residual
                   : workspace.initial_active_residual;
    for (int local = 0; local < active; ++local)
    {
        if (use_initial_metrics)
        {
            const int column = workspace.active_columns[local];
            workspace.residual_scales[local] =
                (*initial_residual_norms)[column];
            workspace.rhs_denominators[local] =
                (*initial_rhs_denominators)[column];
        }
        else
        {
            workspace.residual_scales[local] =
                column_norm(initial_active_residual, rows, active, local);
            const double rhs_norm =
                column_norm(active_rhs, rows, active, local);
            workspace.rhs_denominators[local] =
                rhs_norm > 1e-14 ? rhs_norm : 1.0;
        }
    }
    timing.control += profile_seconds(active_control_started);

    int compressed_rank = 0;
    if (active > 1)
    {
        double compressed_maximum = residual;
        if (!rank_compressed_residual_correction(
                factor, factor_workspace, target_operator,
                workspace.rhs_denominators, rows, active, tolerance, timing,
                active_solution, initial_active_residual, workspace,
                compressed_rank, compressed_maximum))
            return false;
        if (compressed_rank > 0)
        {
            for (int local = 0; local < active; ++local)
                workspace.column_errors[workspace.active_columns[local]] =
                    workspace.compressed_errors[local];
            residual = 0.0;
            for (double error : workspace.column_errors)
                residual = std::max(residual, error);

            if (!full_width)
                for (int row = 0; row < rows; ++row)
                    for (int local = 0; local < active; ++local)
                        solution[static_cast<std::size_t>(row) * columns +
                                 workspace.active_columns[local]] =
                            active_solution[
                                static_cast<std::size_t>(row) * active +
                                local];

            if (compressed_maximum <= tolerance)
            {
                if (!certify_residual)
                {
                    iterations = static_cast<double>(compressed_rank) /
                        columns;
                    return true;
                }
                if (!explicit_column_residuals(
                        target_operator, rhs, solution, columns,
                        workspace.active_residual, workspace.column_errors,
                        residual, timing))
                    return false;
                if (residual <= tolerance)
                {
                    iterations = static_cast<double>(compressed_rank) /
                        columns;
                    return true;
                }
                if (!full_width)
                    for (int row = 0; row < rows; ++row)
                        for (int local = 0; local < active; ++local)
                            initial_active_residual[
                                static_cast<std::size_t>(row) * active +
                                local] = workspace.active_residual[
                                    static_cast<std::size_t>(row) * columns +
                                    workspace.active_columns[local]];
            }

            for (int local = 0; local < active; ++local)
                workspace.residual_scales[local] = column_norm(
                    initial_active_residual, rows, active, local);
        }
    }

    int global_steps = 0;
    const int global_limit =
        active > 1 ? std::min(max_iterations, std::max(4, 2 * active)) : 0;
    if (global_limit > 0)
    {
        const auto global_workspace_started = ProfileClock::now();
        workspace.prepare_global_lazy(active_size, global_limit);
        auto &basis = workspace.basis;
        auto &preconditioned = workspace.preconditioned;
        auto &h_qr = workspace.h;
        auto &c = workspace.c;
        auto &s = workspace.s;
        auto &g = workspace.g;
        auto &y = workspace.y;
        workspace.global_h_raw.assign(
            static_cast<std::size_t>(global_limit + 1) * global_limit, 0.0);
        workspace.global_y.assign(global_limit, 0.0);
        auto &h_raw = workspace.global_h_raw;
        timing.workspace += profile_seconds(global_workspace_started);
        auto Hraw = [&](int row, int column) -> double & {
            return h_raw[static_cast<std::size_t>(row) * global_limit +
                         column];
        };
        auto Hqr = [&](int row, int column) -> double & {
            return h_qr[static_cast<std::size_t>(row) * global_limit +
                        column];
        };

        const auto basis_setup_started = ProfileClock::now();
        const double beta = std::sqrt(static_cast<double>(active));
        for (int row = 0; row < rows; ++row)
            for (int local = 0; local < active; ++local)
            {
                const std::size_t index =
                    static_cast<std::size_t>(row) * active + local;
                basis[0][index] = initial_active_residual[index] /
                    (workspace.residual_scales[local] * beta);
            }
        if (!(beta > 0.0) || !std::isfinite(beta))
            return false;
        g[0] = beta;
        timing.control += profile_seconds(basis_setup_started);

        double best_column_residual = residual;
        std::vector<double> reconstruction_coefficients(
            static_cast<std::size_t>(global_limit) + 1, 0.0);
        std::vector<double> normalized_squared(active, 0.0);
        for (int k = 0; k < global_limit; ++k)
        {
            preconditioned[k].resize(active_size);
            if (!factor.solve(basis[k], active, preconditioned[k], timing,
                              factor_workspace))
                return false;
            auto &candidate = basis[k + 1];
            const auto operator_started = ProfileClock::now();
            if (!apply_right_preconditioned(
                    target_operator, difference_operator, basis[k],
                    preconditioned[k], active, candidate,
                    packed_solution(factor_workspace)))
                return false;
            timing.difference_matvec += profile_seconds(operator_started);

            const auto arnoldi_started = ProfileClock::now();
            const double norm_before = std::sqrt(std::max(
                0.0, contiguous_squared_norm(candidate)));
            for (int pass = 0; pass < 2; ++pass)
            {
                for (int j = 0; j <= k; ++j)
                {
                    const double coefficient =
                        contiguous_dot(basis[j], candidate);
                    Hraw(j, k) += coefficient;
                    axpy_raw(-coefficient, basis[j].data(),
                             candidate.data(),
                             static_cast<int>(active_size));
                }
                if (pass == 0)
                {
                    const double norm_after = std::sqrt(std::max(
                        0.0, contiguous_squared_norm(candidate)));
                    if (norm_after > 0.7 * norm_before)
                        break;
                }
            }

            Hraw(k + 1, k) = std::sqrt(std::max(
                0.0, contiguous_squared_norm(candidate)));
            const bool breakdown = Hraw(k + 1, k) <= 1e-15;
            if (!breakdown)
            {
                const double inverse = 1.0 / Hraw(k + 1, k);
                for (double &value : candidate)
                    value *= inverse;
            }

            for (int row = 0; row <= k + 1; ++row)
                Hqr(row, k) = Hraw(row, k);
            for (int j = 0; j < k; ++j)
            {
                const double top =
                    c[j] * Hqr(j, k) + s[j] * Hqr(j + 1, k);
                Hqr(j + 1, k) =
                    -s[j] * Hqr(j, k) + c[j] * Hqr(j + 1, k);
                Hqr(j, k) = top;
            }
            const double rho = std::hypot(Hqr(k, k), Hqr(k + 1, k));
            c[k] = rho == 0.0 ? 1.0 : Hqr(k, k) / rho;
            s[k] = rho == 0.0 ? 0.0 : Hqr(k + 1, k) / rho;
            Hqr(k, k) = rho;
            Hqr(k + 1, k) = 0.0;
            g[k + 1] = -s[k] * g[k];
            g[k] *= c[k];
            const int trial_steps = k + 1;

            bool singular = false;
            std::fill(y.begin(), y.end(), 0.0);
            for (int row = trial_steps - 1; row >= 0; --row)
            {
                double value = g[row];
                for (int inner = row + 1; inner < trial_steps; ++inner)
                    value -= Hqr(row, inner) * y[inner];
                if (std::abs(Hqr(row, row)) < 1e-30)
                {
                    singular = true;
                    break;
                }
                y[row] = value / Hqr(row, row);
            }
            if (singular)
                break;
            global_steps = trial_steps;
            std::copy_n(y.begin(), global_steps, workspace.global_y.begin());

            std::fill(reconstruction_coefficients.begin(),
                      reconstruction_coefficients.end(), 0.0);
            for (int basis_index = 0;
                 basis_index <= global_steps; ++basis_index)
            {
                double coefficient = basis_index == 0 ? beta : 0.0;
                for (int iteration = 0;
                     iteration < global_steps; ++iteration)
                    coefficient -=
                        Hraw(basis_index, iteration) *
                        workspace.global_y[iteration];
                reconstruction_coefficients[basis_index] = coefficient;
            }

            workspace.reconstructed_residual.resize(active_size);
            std::fill(normalized_squared.begin(), normalized_squared.end(),
                      0.0);
            for (std::size_t index = 0; index < active_size; ++index)
            {
                double value = 0.0;
                for (int basis_index = 0;
                     basis_index <= global_steps; ++basis_index)
                    value += reconstruction_coefficients[basis_index] *
                             basis[basis_index][index];
                workspace.reconstructed_residual[index] = value;
                const int local = static_cast<int>(index % active);
                normalized_squared[local] += value * value;
            }
            best_column_residual = 0.0;
            for (int local = 0; local < active; ++local)
            {
                const double normalized_norm =
                    std::sqrt(normalized_squared[local]);
                const double estimate =
                    workspace.residual_scales[local] * normalized_norm /
                    workspace.rhs_denominators[local];
                workspace.column_errors[workspace.active_columns[local]] =
                    estimate;
                best_column_residual =
                    std::max(best_column_residual, estimate);
            }
            if (timing.residual_history.size() <=
                static_cast<std::size_t>(global_steps))
                timing.residual_history.resize(global_steps + 1, 0.0);
            timing.residual_history[global_steps] = std::max(
                timing.residual_history[global_steps], best_column_residual);
            timing.arnoldi += profile_seconds(arnoldi_started);
            if (best_column_residual <= tolerance || breakdown)
                break;
        }

        const auto update_started = ProfileClock::now();
        for (int row = 0; row < rows; ++row)
            for (int local = 0; local < active; ++local)
            {
                const std::size_t index =
                    static_cast<std::size_t>(row) * active + local;
                double value = active_solution[index];
                for (int iteration = 0;
                     iteration < global_steps; ++iteration)
                {
                    value +=
                        workspace.global_y[iteration] *
                        preconditioned[iteration][index] *
                        workspace.residual_scales[local];
                }
                active_solution[index] = value;
            }
        timing.final_update += profile_seconds(update_started);
        if (!full_width)
            for (int row = 0; row < rows; ++row)
                for (int local = 0; local < active; ++local)
                    solution[static_cast<std::size_t>(row) * columns +
                             workspace.active_columns[local]] =
                        active_solution[
                            static_cast<std::size_t>(row) * active + local];

        if (certify_residual)
        {
            if (!explicit_column_residuals(
                    target_operator, rhs, solution, columns,
                    workspace.active_residual, workspace.column_errors,
                    residual, timing))
                return false;
        }
        else if (global_steps > 0)
        {
            const auto residual_control_started = ProfileClock::now();
            residual = best_column_residual;
            for (int local = 0; local < active; ++local)
            {
                const int column = workspace.active_columns[local];
                if (workspace.column_errors[column] > tolerance)
                    for (int row = 0; row < rows; ++row)
                        workspace.active_residual[
                            static_cast<std::size_t>(row) * columns + column] =
                            workspace.residual_scales[local] *
                            workspace.reconstructed_residual[
                                static_cast<std::size_t>(row) * active + local];
            }
            timing.control += profile_seconds(residual_control_started);
        }
    }

    if (recycle_rank > 0 && recycle.rank == 0 && global_steps > 0)
        if (!build_recycle_space(
                factor, factor_workspace, target_operator,
                difference_operator, workspace, rows, active, global_steps,
                recycle_rank, timing, recycle))
            return false;

    double weighted_iterations = static_cast<double>(compressed_rank) +
        static_cast<double>(global_steps) * active;
    int remaining_iterations = std::max(0, max_iterations - global_steps - compressed_rank);
    while (remaining_iterations > 0)
    {
        workspace.failed_columns.clear();
        for (int column = 0; column < columns; ++column)
            if (workspace.column_errors[column] > tolerance)
                workspace.failed_columns.push_back(column);
        if (workspace.failed_columns.empty())
            break;
        const int failed = static_cast<int>(workspace.failed_columns.size());
        const std::size_t failed_size =
            static_cast<std::size_t>(rows) * failed;
        resize_solve_workspace(workspace.correction_rhs, failed_size);
        for (int row = 0; row < rows; ++row)
            for (int local = 0; local < failed; ++local)
                workspace.correction_rhs[
                    static_cast<std::size_t>(row) * failed + local] =
                    workspace.active_residual[
                        static_cast<std::size_t>(row) * columns +
                        workspace.failed_columns[local]];

        double correction_iterations = 0.0;
        double correction_residual = 0.0;
        double maximum_error = 0.0;
        for (int column : workspace.failed_columns)
            maximum_error = std::max(
                maximum_error, workspace.column_errors[column]);
        // gmres() measures its residual relative to the correction RHS,
        // whereas tolerance is relative to the original RHS.  If the current
        // relative residual is e0, requiring a correction-relative residual
        // of tolerance / e0 gives the requested original-system tolerance.
        const double correction_tolerance = std::max(
            10.0 * std::numeric_limits<double>::epsilon(),
            tolerance / maximum_error);
        workspace.correction_solution.clear();
        auto previous_history = std::move(timing.residual_history);
        timing.residual_history.clear();
        // Use restarted GMRES for a correction.  The total iteration budget is
        // unchanged, but a short cycle avoids allocating a KKT-sized basis for
        // the full dimension when the retained factors normally need only a
        // handful of correction steps.
        constexpr int correction_restart = 32;
        const int cycle_budget = std::min(
            remaining_iterations, correction_restart);
        const bool estimated_convergence = gmres(
            factor, factor_workspace, target_operator, difference_operator,
            workspace.correction_rhs, failed, correction_tolerance,
            cycle_budget, timing, workspace.correction_solution,
            correction_iterations, correction_residual, workspace);
        auto cycle_history = std::move(timing.residual_history);
        timing.residual_history = std::move(previous_history);
        if (timing.residual_history.empty())
            timing.residual_history.push_back(maximum_error);
        for (std::size_t index = 1; index < cycle_history.size(); ++index)
            timing.residual_history.push_back(cycle_history[index] * maximum_error);
        const int cycle_steps = workspace.used.empty() ? 0 :
            *std::max_element(workspace.used.begin(), workspace.used.end());
        if (cycle_steps <= 0 || workspace.correction_solution.size() != failed_size ||
            !std::all_of(workspace.correction_solution.begin(),
                         workspace.correction_solution.end(),
                         [](double value) { return std::isfinite(value); }))
        {
            iterations = weighted_iterations / columns;
            return false;
        }
        remaining_iterations -= cycle_steps;
        for (int row = 0; row < rows; ++row)
            for (int local = 0; local < failed; ++local)
                solution[static_cast<std::size_t>(row) * columns +
                         workspace.failed_columns[local]] +=
                    workspace.correction_solution[
                        static_cast<std::size_t>(row) * failed + local];
        weighted_iterations += correction_iterations * failed;
        if (certify_residual)
        {
            // A small recursive estimate is only a candidate for acceptance.
            // Restart from b-A*x if certification fails, sharing one budget.
            if (!explicit_column_residuals(
                    target_operator, rhs, solution, columns,
                    workspace.active_residual, workspace.column_errors,
                    residual, timing))
            {
                iterations = weighted_iterations / columns;
                return false;
            }
            timing.residual_history.back() = residual;
        }
        else
        {
            for (int column : workspace.failed_columns)
                // gmres() reports residual relative to the correction RHS.
                // Convert it back to the original-RHS scale before applying
                // the outer stopping test.
                workspace.column_errors[column] *= correction_residual;
            residual = 0.0;
            for (double error : workspace.column_errors)
                residual = std::max(residual, error);
            if (!estimated_convergence)
                break;
        }
    }

    iterations = weighted_iterations / columns;
    if (certify_residual)
    {
        if (!explicit_column_residuals(
                target_operator, rhs, solution, columns,
                workspace.active_residual, workspace.column_errors,
                residual, timing))
            return false;
    }
    return residual <= tolerance;
}
std::shared_ptr<CachedForwardFactorization> build_exact_updated_factorization(
    const CachedForwardFactorization &forward,
    const double *target_hessian, const double *target_terminal_hessian,
    const double *target_dynamics)
{
    const int horizon = forward.horizon;
    const int nx = forward.nx;
    const int nu = forward.nu;
    const int stage_dimension = nx + nu;
    if (horizon <= 0 || nx <= 0 || nu <= 0)
        throw std::invalid_argument("exact factor update requires positive OCP dimensions");

    std::vector<double> value_hessians(
        static_cast<std::size_t>(horizon + 1) * nx * nx, 0.0);
    std::vector<double> control_cholesky(
        static_cast<std::size_t>(horizon) * nu * nu, 0.0);
    std::vector<double> cross_factors(
        static_cast<std::size_t>(horizon) * nx * nu, 0.0);
    std::copy_n(target_terminal_hessian, static_cast<std::size_t>(nx) * nx,
                value_hessians.data() +
                    static_cast<std::size_t>(horizon) * nx * nx);

    std::vector<double> propagated(static_cast<std::size_t>(nx) * stage_dimension);
    std::vector<double> stage_matrix(
        static_cast<std::size_t>(stage_dimension) * stage_dimension);
    std::vector<double> target_control(static_cast<std::size_t>(nu) * nu);
    std::vector<double> triangular_rhs(nu);
    const std::vector<double> &forward_controls = forward.cached_control_factors();

    for (int stage = horizon - 1; stage >= 0; --stage)
    {
        const double *dynamics = target_dynamics +
            static_cast<std::size_t>(stage) * nx * stage_dimension;
        const double *hessian = target_hessian +
            static_cast<std::size_t>(stage) * stage_dimension * stage_dimension;
        const double *next_value = value_hessians.data() +
            static_cast<std::size_t>(stage + 1) * nx * nx;

        for (int row = 0; row < nx; ++row)
            for (int column = 0; column < stage_dimension; ++column)
            {
                double value = 0.0;
                for (int inner = 0; inner < nx; ++inner)
                    value += next_value[static_cast<std::size_t>(row) * nx + inner] *
                             dynamics[static_cast<std::size_t>(inner) *
                                          stage_dimension + column];
                propagated[static_cast<std::size_t>(row) * stage_dimension + column] =
                    value;
            }

        for (int row = 0; row < stage_dimension; ++row)
            for (int column = 0; column < stage_dimension; ++column)
            {
                double value = hessian[
                    static_cast<std::size_t>(row) * stage_dimension + column];
                for (int inner = 0; inner < nx; ++inner)
                    value += dynamics[static_cast<std::size_t>(inner) *
                                          stage_dimension + row] *
                             propagated[static_cast<std::size_t>(inner) *
                                            stage_dimension + column];
                stage_matrix[static_cast<std::size_t>(row) * stage_dimension + column] =
                    value;
            }
        for (int row = 0; row < stage_dimension; ++row)
            for (int column = row + 1; column < stage_dimension; ++column)
            {
                const double symmetric = 0.5 * (
                    stage_matrix[static_cast<std::size_t>(row) * stage_dimension + column] +
                    stage_matrix[static_cast<std::size_t>(column) * stage_dimension + row]);
                stage_matrix[static_cast<std::size_t>(row) * stage_dimension + column] =
                    symmetric;
                stage_matrix[static_cast<std::size_t>(column) * stage_dimension + row] =
                    symmetric;
            }

        for (int row = 0; row < nu; ++row)
            for (int column = 0; column < nu; ++column)
                target_control[static_cast<std::size_t>(row) * nu + column] =
                    stage_matrix[static_cast<std::size_t>(nx + row) *
                                     stage_dimension + nx + column];

        const double *cached_control = forward_controls.data() +
            static_cast<std::size_t>(stage) * nu * nu;
        std::vector<double> updated_control = exact_signed_control_factor_update(
            cached_control, target_control, nu);
        double *stage_control = control_cholesky.data() +
            static_cast<std::size_t>(stage) * nu * nu;
        std::copy(updated_control.begin(), updated_control.end(), stage_control);

        double *stage_cross = cross_factors.data() +
            static_cast<std::size_t>(stage) * nx * nu;
        for (int state = 0; state < nx; ++state)
        {
            for (int control = 0; control < nu; ++control)
            {
                double value = stage_matrix[
                    static_cast<std::size_t>(nx + control) * stage_dimension + state];
                for (int inner = 0; inner < control; ++inner)
                    value -= stage_control[
                                 static_cast<std::size_t>(control) * nu + inner] *
                             triangular_rhs[inner];
                triangular_rhs[control] = value / stage_control[
                    static_cast<std::size_t>(control) * nu + control];
            }
            for (int control = 0; control < nu; ++control)
                stage_cross[static_cast<std::size_t>(state) * nu + control] =
                    triangular_rhs[control];
        }

        double *stage_value = value_hessians.data() +
            static_cast<std::size_t>(stage) * nx * nx;
        for (int row = 0; row < nx; ++row)
            for (int column = 0; column < nx; ++column)
            {
                double value = stage_matrix[
                    static_cast<std::size_t>(row) * stage_dimension + column];
                for (int control = 0; control < nu; ++control)
                    value -= stage_cross[static_cast<std::size_t>(row) * nu + control] *
                             stage_cross[static_cast<std::size_t>(column) * nu + control];
                stage_value[static_cast<std::size_t>(row) * nx + column] = value;
            }
    }

    return std::make_shared<CachedForwardFactorization>(
        horizon, nx, nu, target_hessian, target_terminal_hessian,
        target_dynamics, value_hessians.data(), control_cholesky.data(),
        cross_factors.data());
}

} // namespace

extern "C" const char *frs_last_error(void)
{
    return last_error.c_str();
}

extern "C" int frs_create(
    int horizon, int nx, int nu, const double *h_forward,
    const double *h_forward_terminal, const double *a_blocks, FrsHandle *handle)
{
    clear_error();
    if (horizon <= 0 || nx <= 0 || nu < 0 || h_forward == nullptr ||
        h_forward_terminal == nullptr || a_blocks == nullptr || handle == nullptr)
        return fail(-1, "invalid forward-factorization arguments");
    *handle = nullptr;
    try
    {
        *handle = new PreparedSolver(horizon, nx, nu, h_forward,
                                     h_forward_terminal, a_blocks);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during forward factorization");
    }
}

extern "C" int frs_get_fresh_factorization_seconds(
    FrsHandle handle, double *seconds)
{
    clear_error();
    if (handle == nullptr || seconds == nullptr)
        return fail(-1, "invalid fresh-factorization timing arguments");
    try
    {
        const PreparedSolver &prepared =
            *static_cast<const PreparedSolver *>(handle);
        *seconds =
            prepared.factorization->fresh_numeric_factorization_seconds;
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception while reading factorization timing");
    }
}

extern "C" int frs_create_from_factors(
    int horizon, int nx, int nu, const double *h_forward,
    const double *h_forward_terminal, const double *a_blocks,
    const double *value_hessians, const double *control_cholesky,
    const double *cross_factors, FrsHandle *handle)
{
    clear_error();
    if (horizon <= 0 || nx <= 0 || nu < 0 || h_forward == nullptr ||
        h_forward_terminal == nullptr || a_blocks == nullptr ||
        value_hessians == nullptr || control_cholesky == nullptr ||
        cross_factors == nullptr || handle == nullptr)
        return fail(-1, "invalid cached-factor arguments");
    *handle = nullptr;
    try
    {
        *handle = new PreparedSolver(
            horizon, nx, nu, h_forward, h_forward_terminal, a_blocks,
            value_hessians, control_cholesky, cross_factors);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception while importing FATROP factors");
    }
}

extern "C" int frs_create_from_cached_solve(
    int horizon, int nx, int nu, FrsCachedSolve cached_solve,
    void *context, FrsHandle *handle)
{
    clear_error();
    if (horizon <= 0 || nx <= 0 || nu < 0 || cached_solve == nullptr ||
        context == nullptr || handle == nullptr)
        return fail(-1, "invalid native cached-solve arguments");
    *handle = nullptr;
    try
    {
        *handle = new PreparedSolver(
            horizon, nx, nu, cached_solve, context);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception while binding native cached solve");
    }
}

extern "C" int frs_clone(FrsHandle source, FrsHandle *clone)
{
    clear_error();
    if (source == nullptr || clone == nullptr)
        return fail(-1, "invalid clone arguments");
    *clone = nullptr;
    try
    {
        PreparedSolver &prepared = *static_cast<PreparedSolver *>(source);
        *clone = new PreparedSolver(prepared.data, prepared.factorization);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception while cloning solver workspace");
    }
}

extern "C" void frs_destroy(FrsHandle handle)
{
    delete static_cast<PreparedSolver *>(handle);
}

extern "C" int frs_apply_preconditioned_operator(
    FrsHandle handle, int nrhs, const double *h_target,
    const double *h_target_terminal, const double *a_target,
    const double *input, double *output)
{
    clear_error();
    if (handle == nullptr || nrhs <= 0 || h_target == nullptr ||
        h_target_terminal == nullptr || a_target == nullptr ||
        input == nullptr || output == nullptr)
        return fail(-1, "invalid preconditioned-operator audit arguments");

    try
    {
        PreparedSolver &prepared = *static_cast<PreparedSolver *>(handle);
        CachedForwardFactorization &forward = *prepared.factorization;
        ForwardData &data = *prepared.data;
        const std::size_t size =
            static_cast<std::size_t>(forward.dimension) * nrhs;
        std::vector<double> input_values(input, input + size);
        std::vector<double> preconditioned;
        std::vector<double> applied;
        TimingProfile timing;
        if (!forward.solve(input_values, nrhs, preconditioned, timing,
                           prepared.factor_workspace))
            return fail(-2, "cached solve failed during operator audit");

        BatchedKktOperator target_operator(
            forward.horizon, forward.nx, forward.nu, h_target,
            h_target_terminal, a_target, false);
        std::unique_ptr<ExactDifferenceOperator> difference_operator;
        if (data.has_forward_matrix)
        {
            prepared.h_difference.resize(data.h_forward.size());
            prepared.terminal_difference.resize(data.h_forward_terminal.size());
            prepared.a_difference.resize(data.a_blocks.size());
            for (std::size_t index = 0;
                 index < prepared.h_difference.size(); ++index)
                prepared.h_difference[index] =
                    h_target[index] - data.h_forward[index];
            for (std::size_t index = 0;
                 index < prepared.terminal_difference.size(); ++index)
                prepared.terminal_difference[index] =
                    h_target_terminal[index] - data.h_forward_terminal[index];
            for (std::size_t index = 0;
                 index < prepared.a_difference.size(); ++index)
                prepared.a_difference[index] =
                    a_target[index] - data.a_blocks[index];
            difference_operator = std::make_unique<ExactDifferenceOperator>(
                forward.horizon, forward.nx, forward.nu,
                prepared.h_difference.data(),
                prepared.terminal_difference.data(),
                prepared.a_difference.data());
        }
        if (!apply_right_preconditioned(
                target_operator, difference_operator.get(), input_values,
                preconditioned, nrhs, applied,
                packed_solution(prepared.factor_workspace)))
            return fail(-2, "right-preconditioned operator audit failed");
        std::copy(applied.begin(), applied.end(), output);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during operator audit");
    }
}

extern "C" int frs_solve(
    FrsHandle handle, int nrhs, const double *h_target,
    const double *h_target_terminal, const double *a_target,
    const double *rhs,
    double tolerance, int max_iterations, int chunk_size, int recycle_rank,
    int certify_residual, double *solution, FrsStats *stats)
{
    clear_error();
    if (handle == nullptr || nrhs <= 0 || tolerance <= 0.0 ||
        h_target == nullptr ||
        h_target_terminal == nullptr || a_target == nullptr || rhs == nullptr ||
        solution == nullptr || stats == nullptr)
        return fail(-1, "invalid target-solve arguments");

    try
    {
        PreparedSolver &prepared = *static_cast<PreparedSolver *>(handle);
        CachedForwardFactorization &forward = *prepared.factorization;
        ForwardData &data = *prepared.data;
        const auto started = std::chrono::steady_clock::now();
        TimingProfile timing;
        const auto input_copy_started = ProfileClock::now();
        const std::size_t rhs_size =
            static_cast<std::size_t>(forward.dimension) * nrhs;
        {
            ExcludeAblationPreparation excluded;
            if (rhs != prepared.all_rhs.data())
                prepared.all_rhs.assign(rhs, rhs + rhs_size);
            else if (prepared.all_rhs.size() != rhs_size)
                return fail(-1, "aliased native RHS has the wrong size");
            // Every solve path overwrites the complete solution.
            prepared.all_solution.resize(prepared.all_rhs.size());
        }
        timing.input_copy += profile_seconds(input_copy_started);
        auto &all_rhs = prepared.all_rhs;
        auto &all_solution = prepared.all_solution;
        const auto copy_output = [&]() {
            ExcludeAblationPreparation excluded;
            const auto copy_started = ProfileClock::now();
            std::copy(all_solution.begin(), all_solution.end(), solution);
            timing.final_update += profile_seconds(copy_started);
        };

        const bool target_matches_forward_system = [&]() {
            ExcludeAblationPreparation excluded;
            return data.has_forward_matrix &&
                std::equal(data.h_forward.begin(), data.h_forward.end(), h_target) &&
                std::equal(data.h_forward_terminal.begin(),
                           data.h_forward_terminal.end(), h_target_terminal) &&
                std::equal(data.a_blocks.begin(), data.a_blocks.end(), a_target);
        }();
        if (target_matches_forward_system)
        {
            if (!forward.solve(all_rhs, nrhs, all_solution, timing,
                               prepared.factor_workspace))
                return fail(-2, "cached-factor solve failed");
            double relative_residual = 0.0;
            if (certify_residual)
            {
                const auto residual_started = ProfileClock::now();
                BatchedKktOperator forward_operator(
                    forward.horizon, forward.nx, forward.nu,
                    data.h_forward.data(), data.h_forward_terminal.data(),
                    data.a_blocks.data());
                relative_residual = forward_operator.relative_residual(
                    all_rhs, all_solution, nrhs);
                timing.residual += profile_seconds(residual_started);
            }
            copy_output();
            stats->elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->mean_iterations = 0.0;
            stats->mean_cached_backsolves = 1.0;
            stats->relative_residual = relative_residual;
            stats->used_fallback = 0;
            store_profile(*stats, timing);
            return 0;
        }

        double iteration_sum = 0.0;
        double max_residual = 0.0;
        bool converged = true;
        const auto setup_started = ProfileClock::now();
        BatchedKktOperator target_operator(
            forward.horizon, forward.nx, forward.nu, h_target,
            h_target_terminal, a_target);
        std::unique_ptr<ExactDifferenceOperator> difference_operator;
        // Certified solves use A*(M^-1*v), retaining the cached-solve defect.
        if (data.has_forward_matrix && !certify_residual)
        {
            prepared.h_difference.resize(data.h_forward.size());
            prepared.terminal_difference.resize(data.h_forward_terminal.size());
            prepared.a_difference.resize(data.a_blocks.size());
            for (std::size_t index = 0;
                 index < prepared.h_difference.size(); ++index)
                prepared.h_difference[index] =
                    h_target[index] - data.h_forward[index];
            for (std::size_t index = 0;
                 index < prepared.terminal_difference.size(); ++index)
                prepared.terminal_difference[index] =
                    h_target_terminal[index] - data.h_forward_terminal[index];
            for (std::size_t index = 0;
                 index < prepared.a_difference.size(); ++index)
                prepared.a_difference[index] =
                    a_target[index] - data.a_blocks[index];
            difference_operator = std::make_unique<ExactDifferenceOperator>(
                forward.horizon, forward.nx, forward.nu,
                prepared.h_difference.data(),
                prepared.terminal_difference.data(),
                prepared.a_difference.data());
        }
        timing.setup += profile_seconds(setup_started);
        RecycleSpace recycle;

        if (!forward.solve(all_rhs, nrhs, all_solution, timing,
                           prepared.factor_workspace))
            return fail(-2, "batched cached-factor one-shot failed");
        double initial_max_residual = 0.0;
        auto &initial_errors = prepared.gmres_workspace.column_errors;
        auto &initial_norms = prepared.gmres_workspace.column_residual_norms;
        auto &initial_denominators =
            prepared.gmres_workspace.column_rhs_denominators;
        auto &initial_active = prepared.gmres_workspace.active_columns;
        {
            ExcludeAblationPreparation excluded;
            initial_errors.resize(nrhs);
            initial_active.clear();
            initial_active.reserve(nrhs);
        }
        if (difference_operator)
        {
            const auto initial_operator_started = ProfileClock::now();
            if (!difference_operator->apply(
                    all_solution, nrhs, prepared.initial_residual,
                    packed_solution(prepared.factor_workspace), &all_rhs,
                    &initial_norms, &initial_denominators, -1.0))
                return fail(-2, "batched initial defect residual failed");
            timing.difference_matvec +=
                profile_seconds(initial_operator_started);
            const auto initial_control_started = ProfileClock::now();
            for (int column = 0; column < nrhs; ++column)
            {
                const double residual_norm = std::sqrt(initial_norms[column]);
                const double rhs_norm = std::sqrt(initial_denominators[column]);
                const double denominator = rhs_norm > 1e-14 ? rhs_norm : 1.0;
                const double error = residual_norm / denominator;
                initial_norms[column] = residual_norm;
                initial_denominators[column] = denominator;
                initial_errors[column] = error;
                if (error > tolerance)
                    initial_active.push_back(column);
                initial_max_residual = std::max(initial_max_residual, error);
            }
            timing.control += profile_seconds(initial_control_started);
        }
        else if (!explicit_column_residuals(
                     target_operator, all_rhs, all_solution, nrhs,
                     prepared.initial_residual, initial_errors,
                     initial_max_residual, timing, &initial_norms,
                     &initial_denominators, &initial_active, tolerance,
                     packed_solution(prepared.factor_workspace)))
        {
            return fail(-2, "native cached-solve initial residual failed");
        }
        if (difference_operator && certify_residual &&
            initial_max_residual <= tolerance)
        {
            if (!explicit_column_residuals(
                    target_operator, all_rhs, all_solution, nrhs,
                    prepared.initial_residual, initial_errors,
                    initial_max_residual, timing, &initial_norms,
                    &initial_denominators, &initial_active, tolerance,
                    packed_solution(prepared.factor_workspace)))
                return fail(-2, "one-shot true-residual certification failed");
        }
        if (initial_max_residual <= tolerance)
        {
            copy_output();
            stats->elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->mean_iterations = 0.0;
            stats->mean_cached_backsolves = 1.0;
            stats->relative_residual = initial_max_residual;
            stats->used_fallback = 0;
            store_profile(*stats, timing);
            return 0;
        }

        if (chunk_size >= nrhs)
        {
            prepared.gmres_workspace.active_residual.swap(
                prepared.initial_residual);
            double iterations = 0.0;
            double residual = 0.0;
            const bool success = global_gmres_certified(
                forward, prepared.factor_workspace, target_operator,
                difference_operator.get(),
                all_rhs, nrhs, tolerance, max_iterations, timing,
                all_solution, iterations, residual,
                prepared.gmres_workspace, recycle, recycle_rank,
                certify_residual != 0,
                &prepared.gmres_workspace.active_residual,
                &prepared.gmres_workspace.column_errors,
                &prepared.gmres_workspace.column_residual_norms,
                &prepared.gmres_workspace.column_rhs_denominators,
                &prepared.gmres_workspace.active_columns,
                initial_max_residual);
            if (!success || !std::isfinite(residual) || residual > tolerance)
            {
                stats->elapsed_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                stats->mean_iterations = iterations;
                stats->mean_cached_backsolves = iterations + 1.0;
                stats->relative_residual = residual;
                stats->used_fallback = 0;
                store_profile(*stats, timing);
                std::ostringstream message;
                message << std::scientific << std::setprecision(10)
                        << "GMRES did not reach the requested tolerance; "
                        << "initial_relative_residual=" << initial_max_residual
                        << ", final_relative_residual=" << residual
                        << ", final_over_tolerance=" << residual / tolerance
                        << ", iterations=" << iterations
                        << ", max_iterations=" << max_iterations;
                return fail(1, message.str());
            }
            copy_output();
            stats->elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->mean_iterations = iterations;
            stats->mean_cached_backsolves = iterations + 1.0;
            stats->relative_residual = residual;
            stats->used_fallback = 0;
            store_profile(*stats, timing);
            return 0;
        }

        for (int start = 0; start < nrhs; start += chunk_size)
        {
            const int columns = std::min(chunk_size, nrhs - start);
            prepared.rhs_chunk.resize(
                static_cast<std::size_t>(forward.dimension) * columns);
            prepared.solution_chunk.resize(
                static_cast<std::size_t>(forward.dimension) * columns);
            prepared.residual_chunk.resize(
                static_cast<std::size_t>(forward.dimension) * columns);
            auto &chunk = prepared.rhs_chunk;
            auto &chunk_solution = prepared.solution_chunk;
            auto &chunk_residual = prepared.residual_chunk;
            for (int row = 0; row < forward.dimension; ++row)
                for (int column = 0; column < columns; ++column)
                {
                    chunk[row * columns + column] =
                        all_rhs[row * nrhs + start + column];
                    chunk_solution[row * columns + column] =
                        all_solution[row * nrhs + start + column];
                    chunk_residual[row * columns + column] =
                        prepared.initial_residual[
                            row * nrhs + start + column];
                }

            double iterations = 0.0;
            double residual = 0.0;
            const bool success = global_gmres_certified(
                forward, prepared.factor_workspace, target_operator,
                difference_operator.get(),
                chunk, columns, tolerance, max_iterations, timing,
                chunk_solution, iterations, residual,
                prepared.gmres_workspace, recycle, recycle_rank,
                certify_residual != 0, &chunk_residual);
            iteration_sum += iterations * columns;
            max_residual = std::max(max_residual, residual);
            if (!success || residual > tolerance)
            {
                converged = false;
                break;
            }
            for (int row = 0; row < forward.dimension; ++row)
                for (int column = 0; column < columns; ++column)
                    all_solution[row * nrhs + start + column] =
                        chunk_solution[row * columns + column];
        }

        stats->used_fallback = 0;
        if (!converged)
        {
            stats->elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->mean_iterations = iteration_sum / nrhs;
            stats->mean_cached_backsolves = stats->mean_iterations + 1.0;
            stats->relative_residual = max_residual;
            stats->used_fallback = 0;
            store_profile(*stats, timing);
            return fail(
                1,
                "GMRES did not reach the requested tolerance; "
                "initial_relative_residual=" +
                    std::to_string(initial_max_residual) +
                ", final_relative_residual=" +
                    std::to_string(max_residual) +
                ", final_over_tolerance=" +
                    std::to_string(max_residual / tolerance) +
                ", mean_iterations=" +
                    std::to_string(stats->mean_iterations) +
                ", max_iterations=" + std::to_string(max_iterations));
        }

        copy_output();
        stats->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        stats->mean_iterations = iteration_sum / nrhs;
        stats->mean_cached_backsolves = stats->mean_iterations + 1.0;
        stats->relative_residual = max_residual;
        store_profile(*stats, timing);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during target solve");
    }
}

extern "C" int frs_solve_unpreconditioned(
    FrsHandle handle, int nrhs, const double *h_target,
    const double *h_target_terminal, const double *a_target,
    const double *rhs, double tolerance, int max_iterations,
    int certify_residual, double *solution, FrsStats *stats)
{
    clear_error();
    if (handle == nullptr || nrhs <= 0 || tolerance <= 0.0 ||
        max_iterations <= 0 || h_target == nullptr ||
        h_target_terminal == nullptr || a_target == nullptr || rhs == nullptr ||
        solution == nullptr || stats == nullptr)
        return fail(-1, "invalid unpreconditioned-solve arguments");

    try
    {
        PreparedSolver &prepared = *static_cast<PreparedSolver *>(handle);
        CachedForwardFactorization &forward = *prepared.factorization;
        const auto started = std::chrono::steady_clock::now();
        TimingProfile timing;

        const auto input_copy_started = ProfileClock::now();
        const std::size_t rhs_size =
            static_cast<std::size_t>(forward.dimension) * nrhs;
        {
            ExcludeAblationPreparation excluded;
            if (rhs != prepared.all_rhs.data())
                prepared.all_rhs.assign(rhs, rhs + rhs_size);
            else if (prepared.all_rhs.size() != rhs_size)
                return fail(-1, "aliased native RHS has the wrong size");
            prepared.all_solution.resize(rhs_size);
        }
        timing.input_copy += profile_seconds(input_copy_started);

        BatchedKktOperator target_operator(
            forward.horizon, forward.nx, forward.nu, h_target,
            h_target_terminal, a_target);
        double iterations = 0.0;
        double estimated_residual = std::numeric_limits<double>::infinity();
        const bool converged = gmres(
            forward, prepared.factor_workspace, target_operator, nullptr,
            prepared.all_rhs, nrhs, tolerance, max_iterations, timing,
            prepared.all_solution, iterations, estimated_residual,
            prepared.gmres_workspace, false);

        double reported_residual = estimated_residual;
        if (certify_residual)
        {
            const auto residual_started = ProfileClock::now();
            reported_residual = target_operator.relative_residual(
                prepared.all_rhs, prepared.all_solution, nrhs);
            timing.residual += profile_seconds(residual_started);
        }

        const auto output_started = ProfileClock::now();
        {
            ExcludeAblationPreparation excluded;
            std::copy(prepared.all_solution.begin(), prepared.all_solution.end(),
                      solution);
        }
        timing.final_update += profile_seconds(output_started);

        stats->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        stats->mean_iterations = iterations;
        stats->mean_cached_backsolves = 0.0;
        stats->relative_residual = reported_residual;
        stats->used_fallback = 0;
        store_profile(*stats, timing);

        if (!converged || !std::isfinite(reported_residual) ||
            reported_residual > tolerance)
            return fail(1, "unpreconditioned GMRES did not reach the requested tolerance");
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during unpreconditioned solve");
    }
}

extern "C" int frs_solve_ablation(
    FrsHandle handle, const double *h_target,
    const double *h_target_terminal, const double *a_target,
    const double *rhs, double tolerance, int max_iterations,
    int reuse_forward_factors, double *solution, FrsStats *stats,
    double *call_seconds)
{
    clear_error();
    if (handle == nullptr || h_target == nullptr ||
        h_target_terminal == nullptr || a_target == nullptr || rhs == nullptr ||
        solution == nullptr || stats == nullptr || call_seconds == nullptr ||
        tolerance <= 0.0 || max_iterations <= 0 ||
        (reuse_forward_factors != 0 && reuse_forward_factors != 1) ||
        ablation_timer != nullptr)
        return fail(-1, "invalid solve-only ablation arguments");
    *stats = {};
    *call_seconds = 0.0;
    AblationSolveTimer timer;
    BindAblationTimer binding(timer);
    const int status = reuse_forward_factors
        ? frs_solve(handle, 1, h_target, h_target_terminal, a_target, rhs,
                    tolerance, max_iterations, 1, 0, 1, solution, stats)
        : frs_solve_unpreconditioned(
              handle, 1, h_target, h_target_terminal, a_target, rhs,
              tolerance, max_iterations, 1, solution, stats);
    timer.stop();
    *call_seconds = timer.call_seconds;
    stats->elapsed_seconds = timer.solve_seconds;
    return status;
}

extern "C" int frs_solve_adjoint_batch(
    FrsHandle *handles, int batch_size, const double *h_target_batch,
    const double *h_target_terminal_batch, const double *a_target_batch,
    const double *rhs_batch, double tolerance, int max_iterations,
    int certify_residual, double *solution_batch, FrsStats *stats_batch)
{
    clear_error();
    if (handles == nullptr || batch_size <= 0 || h_target_batch == nullptr ||
        h_target_terminal_batch == nullptr || a_target_batch == nullptr ||
        rhs_batch == nullptr || tolerance <= 0.0 || max_iterations <= 0 ||
        solution_batch == nullptr || stats_batch == nullptr)
        return fail(-1, "invalid batched adjoint-solve arguments");

    try
    {
        if (handles[0] == nullptr)
            return fail(-1, "batched adjoint solve received a null handle");
        PreparedSolver &first = *static_cast<PreparedSolver *>(handles[0]);
        const CachedForwardFactorization &first_factor = *first.factorization;
        const ForwardData &first_data = *first.data;
        const std::size_t h_stride = first_data.h_forward.size();
        const std::size_t terminal_stride = first_data.h_forward_terminal.size();
        const std::size_t a_stride = first_data.a_blocks.size();
        const std::size_t rhs_stride =
            static_cast<std::size_t>(first_factor.dimension);

        for (int index = 0; index < batch_size; ++index)
        {
            if (handles[index] == nullptr)
                return fail(-1, "batched adjoint solve received a null handle");
            PreparedSolver &prepared =
                *static_cast<PreparedSolver *>(handles[index]);
            const CachedForwardFactorization &factor = *prepared.factorization;
            const ForwardData &data = *prepared.data;
            if (factor.horizon != first_factor.horizon ||
                factor.nx != first_factor.nx || factor.nu != first_factor.nu ||
                factor.dimension != first_factor.dimension ||
                data.h_forward.size() != h_stride ||
                data.h_forward_terminal.size() != terminal_stride ||
                data.a_blocks.size() != a_stride)
                return fail(-1, "batched adjoint entries have incompatible dimensions");
        }

        std::vector<int> statuses(static_cast<std::size_t>(batch_size), 0);
#pragma omp parallel for schedule(static)
        for (int index = 0; index < batch_size; ++index)
        {
            stats_batch[index] = FrsStats{};
            statuses[static_cast<std::size_t>(index)] = frs_solve(
                handles[index], 1,
                h_target_batch + static_cast<std::size_t>(index) * h_stride,
                h_target_terminal_batch +
                    static_cast<std::size_t>(index) * terminal_stride,
                a_target_batch + static_cast<std::size_t>(index) * a_stride,
                rhs_batch + static_cast<std::size_t>(index) * rhs_stride,
                tolerance, max_iterations, 1, 0, certify_residual,
                solution_batch + static_cast<std::size_t>(index) * rhs_stride,
                stats_batch + index);
        }
        for (int index = 0; index < batch_size; ++index)
        {
            const int status = statuses[static_cast<std::size_t>(index)];
            if (status != 0)
                return fail(status, "an entry in the batched adjoint solve failed");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during batched adjoint solve");
    }
}

extern "C" int frs_build_chain_target_and_solve(
    FrsHandle handle, int parameters, const double *optimal_primal,
    const double *dynamics_multipliers, const double *coupling,
    const double *nonlinearity, const double *state_weights,
    const double *control_weights, const double *state_parameters,
    const double *control_parameters, const double *terminal_parameters,
    const double *dynamics_parameters, double terminal_multiplier,
    double tolerance, int max_iterations, int chunk_size, int recycle_rank,
    int certify_residual, double *solution, FrsStats *stats)
{
    clear_error();
    if (handle == nullptr || parameters <= 0 || optimal_primal == nullptr ||
        dynamics_multipliers == nullptr || coupling == nullptr ||
        nonlinearity == nullptr || state_weights == nullptr ||
        control_weights == nullptr || state_parameters == nullptr ||
        control_parameters == nullptr || terminal_parameters == nullptr ||
        dynamics_parameters == nullptr || terminal_multiplier <= 0.0 ||
        tolerance <= 0.0 || solution == nullptr || stats == nullptr)
        return fail(-1, "invalid native target-build arguments");

    try
    {
        PreparedSolver &prepared = *static_cast<PreparedSolver *>(handle);
        CachedForwardFactorization &forward = *prepared.factorization;
        const int horizon = forward.horizon;
        const int nx = forward.nx;
        const int nu = forward.nu;
        if (nu <= 0 || nx % 2 != 0)
            return fail(-1, "nonlinear-chain builder requires positive nu and even nx");

        constexpr double dt = 0.05;
        constexpr double barrier_mu = 2.0e-2;
        constexpr double barrier_delta = 5.0e-2;
        constexpr double control_bound = 0.35;
        constexpr double position_bound = 0.32;
        constexpr double velocity_bound = 0.80;

        const int stage_dimension = nx + nu;
        const int primal_dimension = horizon * stage_dimension + nx;
        const int dual_dimension = (horizon + 1) * nx;
        const int dimension = primal_dimension + dual_dimension;
        const int masses = nx / 2;
        const auto initialize_once = [](std::vector<double> &values,
                                        std::size_t size) {
            if (values.size() != size)
                values.assign(size, 0.0);
        };
        initialize_once(
            prepared.target_hessian_workspace,
            static_cast<std::size_t>(horizon) * stage_dimension *
                stage_dimension);
        initialize_once(
            prepared.target_terminal_hessian_workspace,
            static_cast<std::size_t>(nx) * nx);
        initialize_once(
            prepared.target_dynamics_workspace,
            static_cast<std::size_t>(horizon) * nx * stage_dimension);
        initialize_once(
            prepared.all_rhs,
            static_cast<std::size_t>(dimension) * parameters);
        auto &target_hessian = prepared.target_hessian_workspace;
        auto &target_terminal_hessian =
            prepared.target_terminal_hessian_workspace;
        auto &target_dynamics = prepared.target_dynamics_workspace;
        auto &rhs = prepared.all_rhs;
        // The initial-condition sensitivity block is structurally zero. All
        // remaining RHS entries are overwritten below in contiguous rows.
        std::fill_n(
            rhs.data() + static_cast<std::size_t>(primal_dimension) * parameters,
            static_cast<std::size_t>(nx) * parameters, 0.0);

        const auto stage_angle = [horizon](int stage) {
            return 2.0 * std::acos(-1.0) *
                   (static_cast<double>(stage) + 0.5) / std::max(1, horizon);
        };
        const auto linear_stiffness = [&](int mass, int stage) {
            return coupling[mass] *
                   (1.0 + 0.25 * std::sin(stage_angle(stage) + 0.17 * mass));
        };
        const auto cubic_stiffness = [&](int mass, int stage) {
            return nonlinearity[mass % nu] *
                   (1.0 + 0.20 * std::cos(stage_angle(stage) + 0.11 * mass));
        };
        const auto neighbor_stiffness = [&](int mass, int other, int stage) {
            return 0.18 * std::sqrt(coupling[mass] * coupling[other]) *
                   (1.0 + 0.15 * std::sin(
                       stage_angle(stage) + 0.07 * (mass + other)));
        };
        const auto relaxed_curvature = [barrier_delta](double slack) {
            return slack > barrier_delta
                       ? 1.0 / (slack * slack)
                       : 1.0 / (barrier_delta * barrier_delta);
        };
        const auto barrier_curvature = [&](double value, double bound) {
            return barrier_mu *
                   (relaxed_curvature(bound - value) +
                    relaxed_curvature(bound + value));
        };

        const auto target_started = ProfileClock::now();
        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dimension;
            const double *x = optimal_primal + primal;
            const double *lambda = dynamics_multipliers + stage * nx;
            double *hessian = target_hessian.data() +
                static_cast<std::size_t>(stage) * stage_dimension * stage_dimension;
            for (int state = 0; state < nx; ++state)
            {
                const double bound = state < masses ? position_bound : velocity_bound;
                hessian[static_cast<std::size_t>(state) * stage_dimension + state] =
                    2.0 * state_weights[state] + barrier_curvature(x[state], bound);
            }
            for (int input = 0; input < nu; ++input)
            {
                const int diagonal = nx + input;
                hessian[static_cast<std::size_t>(diagonal) * stage_dimension + diagonal] =
                    2.0 * control_weights[input] +
                    barrier_curvature(optimal_primal[primal + nx + input], control_bound);
            }
            for (int mass = 0; mass < masses; ++mass)
            {
                const int position = mass;
                const int velocity = masses + mass;
                hessian[static_cast<std::size_t>(position) * stage_dimension + position] +=
                    -6.0 * dt * cubic_stiffness(mass, stage) *
                    x[position] * lambda[velocity];
            }

            double *dynamics = target_dynamics.data() +
                static_cast<std::size_t>(stage) * nx * stage_dimension;
            for (int mass = 0; mass < masses; ++mass)
            {
                const int position = mass;
                const int velocity = masses + mass;
                const int control = mass % nu;
                const double linear = linear_stiffness(mass, stage);
                const double cubic = cubic_stiffness(mass, stage);
                const double left = mass > 0
                    ? neighbor_stiffness(mass, mass - 1, stage) : 0.0;
                const double right = mass + 1 < masses
                    ? neighbor_stiffness(mass, mass + 1, stage) : 0.0;
                dynamics[static_cast<std::size_t>(position) * stage_dimension + position] = -1.0;
                dynamics[static_cast<std::size_t>(position) * stage_dimension + velocity] = -dt;
                dynamics[static_cast<std::size_t>(velocity) * stage_dimension + velocity] =
                    -(1.0 - dt * (0.08 + 0.02 * static_cast<double>(mass % 4)));
                dynamics[static_cast<std::size_t>(velocity) * stage_dimension + position] =
                    -dt * (-linear - 3.0 * cubic * x[position] * x[position] -
                           left - right);
                if (mass > 0)
                    dynamics[static_cast<std::size_t>(velocity) * stage_dimension + position - 1] =
                        -dt * left;
                if (mass + 1 < masses)
                    dynamics[static_cast<std::size_t>(velocity) * stage_dimension + position + 1] =
                        -dt * right;
                dynamics[static_cast<std::size_t>(velocity) * stage_dimension + nx + control] =
                    -dt / (1.0 + 0.10 * mass);
            }
        }
        for (int state = 0; state < nx; ++state)
        {
            const double bound = state < masses ? position_bound : velocity_bound;
            target_terminal_hessian[static_cast<std::size_t>(state) * nx + state] =
                2.0 * terminal_multiplier * state_weights[state] +
                barrier_curvature(
                    optimal_primal[horizon * stage_dimension + state], bound);
        }
        const double target_build_seconds = profile_seconds(target_started);

        const auto sensitivity_started = ProfileClock::now();
        for (int stage = 0; stage < horizon; ++stage)
        {
            const int primal = stage * stage_dimension;
            for (int state = 0; state < nx; ++state)
            {
                double *destination = rhs.data() +
                    static_cast<std::size_t>(primal + state) * parameters;
                const double *source = state_parameters +
                    static_cast<std::size_t>(state) * parameters;
                const double coefficient = 2.0 * state_weights[state];
                for (int parameter = 0; parameter < parameters; ++parameter)
                    destination[parameter] = coefficient * source[parameter];
            }
            for (int input = 0; input < nu; ++input)
            {
                double *destination = rhs.data() +
                    static_cast<std::size_t>(primal + nx + input) * parameters;
                const double *source = control_parameters +
                    static_cast<std::size_t>(input) * parameters;
                const double coefficient = 2.0 * control_weights[input];
                for (int parameter = 0; parameter < parameters; ++parameter)
                    destination[parameter] = coefficient * source[parameter];
            }
            for (int state = 0; state < nx; ++state)
            {
                double *destination = rhs.data() +
                    static_cast<std::size_t>(
                        primal_dimension + (stage + 1) * nx + state) *
                        parameters;
                const double *source = dynamics_parameters +
                    static_cast<std::size_t>(state) * parameters;
                for (int parameter = 0; parameter < parameters; ++parameter)
                    destination[parameter] = dt * source[parameter];
            }
        }
        const int terminal = horizon * stage_dimension;
        for (int state = 0; state < nx; ++state)
        {
            double *destination = rhs.data() +
                static_cast<std::size_t>(terminal + state) * parameters;
            const double *source = terminal_parameters +
                static_cast<std::size_t>(state) * parameters;
            const double coefficient =
                2.0 * terminal_multiplier * state_weights[state];
            for (int parameter = 0; parameter < parameters; ++parameter)
                destination[parameter] = coefficient * source[parameter];
        }
        const double sensitivity_linearization_seconds =
            profile_seconds(sensitivity_started);

        FrsStats solve_stats{};
        const int status = frs_solve(
            handle, parameters, target_hessian.data(),
            target_terminal_hessian.data(), target_dynamics.data(), rhs.data(),
            tolerance, max_iterations, chunk_size, recycle_rank,
            certify_residual, solution, &solve_stats);
        if (status != 0)
            return status;
        *stats = solve_stats;
        stats->target_build_seconds = target_build_seconds;
        stats->sensitivity_linearization_seconds =
            sensitivity_linearization_seconds;
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during native target build");
    }
}

extern "C" int frs_solve_parallel(
    FrsHandle *handles, int handle_count, int nrhs, const double *h_target,
    const double *h_target_terminal, const double *a_target,
    const double *rhs, double tolerance,
    int max_iterations, int tile_size, int recycle_rank,
    int certify_residual, double *solution,
    FrsStats *stats)
{
    clear_error();
    if (handles == nullptr || handle_count <= 0 || nrhs <= 0 ||
        h_target == nullptr || h_target_terminal == nullptr ||
        a_target == nullptr || rhs == nullptr ||
        tolerance <= 0.0 || max_iterations <= 0 || tile_size <= 0 ||
        recycle_rank < 0 ||
        solution == nullptr ||
        stats == nullptr)
        return fail(-1, "invalid parallel target-solve arguments");

    if (handles[0] == nullptr)
        return fail(-1, "target solve received a null handle");
    return frs_solve(
        handles[0], nrhs, h_target, h_target_terminal, a_target, rhs,
        tolerance, max_iterations, tile_size, recycle_rank,
        certify_residual, solution, stats);

    try
    {
        const int active = std::min(handle_count, nrhs);
        std::vector<PreparedSolver *> prepared(active, nullptr);
        for (int worker = 0; worker < active; ++worker)
        {
            if (handles[worker] == nullptr)
                return fail(-1, "parallel target solve received a null handle");
            prepared[worker] = static_cast<PreparedSolver *>(handles[worker]);
        }

        PreparedSolver &base = *prepared[0];
        CachedForwardFactorization &base_factorization = *base.factorization;
        ForwardData &base_data = *base.data;
        const int dimension = base_factorization.dimension;
        for (int worker = 1; worker < active; ++worker)
        {
            const CachedForwardFactorization &factor = *prepared[worker]->factorization;
            if (factor.horizon != base_factorization.horizon ||
                factor.nx != base_factorization.nx || factor.nu != base_factorization.nu ||
                factor.dimension != dimension)
                return fail(-1, "parallel handles have incompatible dimensions");
        }

        if (!parallel_executor || parallel_executor->capacity() < active)
            parallel_executor = std::make_unique<PersistentExecutor>(active);
        const auto started = std::chrono::steady_clock::now();
        const bool target_matches_forward_system =
            std::equal(base_data.h_forward.begin(), base_data.h_forward.end(),
                       h_target) &&
            std::equal(base_data.h_forward_terminal.begin(),
                       base_data.h_forward_terminal.end(), h_target_terminal) &&
            std::equal(base_data.a_blocks.begin(), base_data.a_blocks.end(),
                       a_target);

        const auto setup_started = ProfileClock::now();
        const double setup_seconds = profile_seconds(setup_started);

        struct WorkerResult
        {
            TimingProfile timing;
            int columns = 0;
            double iteration_sum = 0.0;
            double residual = 0.0;
            bool success = false;
            std::string error;
        };
        std::vector<WorkerResult> results(active);
        const auto solve_worker = [&](int worker)
        {
            WorkerResult &result = results[worker];
            try
            {
                BatchedKktOperator target_operator(
                    base_factorization.horizon, base_factorization.nx, base_factorization.nu,
                    h_target, h_target_terminal, a_target);
                const int first_column = nrhs * worker / active;
                const int last_column = nrhs * (worker + 1) / active;
                result.success = true;
                RecycleSpace recycle;
                for (int start = first_column; start < last_column;
                     start += tile_size)
                {
                    const int columns = std::min(tile_size, last_column - start);
                    prepared[worker]->rhs_chunk.resize(
                        static_cast<std::size_t>(dimension) * columns);
                    auto &chunk = prepared[worker]->rhs_chunk;
                    for (int row = 0; row < dimension; ++row)
                        std::copy_n(
                            rhs + static_cast<std::size_t>(row) * nrhs + start,
                            columns,
                            chunk.data() +
                                static_cast<std::size_t>(row) * columns);

                    auto &chunk_solution = prepared[worker]->solution_chunk;
                    double mean_iterations = 0.0;
                    double tile_residual = 0.0;
                    if (target_matches_forward_system)
                        result.success = prepared[worker]->factorization->solve(
                            chunk, columns, chunk_solution, result.timing,
                            prepared[worker]->factor_workspace);
                    else
                        result.success = global_gmres_certified(
                            *prepared[worker]->factorization,
                            prepared[worker]->factor_workspace,
                            target_operator, nullptr, chunk, columns, tolerance,
                            max_iterations, result.timing, chunk_solution,
                            mean_iterations, tile_residual,
                            prepared[worker]->gmres_workspace,
                            recycle, recycle_rank, certify_residual != 0,
                            nullptr);
                    if (!result.success)
                        return;
                    result.iteration_sum += mean_iterations * columns;
                    result.residual =
                        std::max(result.residual, tile_residual);
                    for (int row = 0; row < dimension; ++row)
                        std::copy_n(
                            chunk_solution.data() +
                                static_cast<std::size_t>(row) * columns,
                            columns,
                            solution + static_cast<std::size_t>(row) * nrhs +
                                start);
                }
            }
            catch (const std::exception &error)
            {
                result.error = error.what();
            }
            catch (...)
            {
                result.error = "unknown worker exception";
            }
        };

        for (int worker = 0; worker < active; ++worker)
        {
            const int first_column = nrhs * worker / active;
            const int last_column = nrhs * (worker + 1) / active;
            results[worker].columns = last_column - first_column;
        }
        parallel_executor->run(active, solve_worker);

        bool converged = true;
        double weighted_iterations = 0.0;
        double max_residual = 0.0;
        TimingProfile timing;
        timing.setup = setup_seconds;
        for (const WorkerResult &result : results)
        {
            if (!result.error.empty())
                return fail(-3, result.error);
            converged = converged && result.success &&
                        (target_matches_forward_system || result.residual <= tolerance);
            weighted_iterations += result.iteration_sum;
            max_residual = std::max(max_residual, result.residual);
            timing.rhs_pack = std::max(timing.rhs_pack,
                                       result.timing.rhs_pack);
            timing.backward = std::max(timing.backward,
                                       result.timing.backward);
            timing.trsm = std::max(timing.trsm, result.timing.trsm);
            timing.forward = std::max(timing.forward, result.timing.forward);
            timing.output_unpack = std::max(timing.output_unpack,
                                            result.timing.output_unpack);
            timing.arnoldi = std::max(timing.arnoldi,
                                      result.timing.arnoldi);
            timing.residual = std::max(timing.residual,
                                       result.timing.residual);
            timing.difference_matvec = std::max(
                timing.difference_matvec,
                result.timing.difference_matvec);
            timing.projection = std::max(
                timing.projection, result.timing.projection);
            if (timing.residual_history.size() <
                result.timing.residual_history.size())
                timing.residual_history.resize(
                    result.timing.residual_history.size(), 0.0);
            for (std::size_t index = 0;
                 index < result.timing.residual_history.size(); ++index)
                timing.residual_history[index] = std::max(
                    timing.residual_history[index],
                    result.timing.residual_history[index]);
        }

        stats->used_fallback = 0;
        if (converged && certify_residual &&
            target_matches_forward_system)
        {
            std::vector<double> all_rhs(
                rhs, rhs + static_cast<std::size_t>(dimension) * nrhs);
            std::vector<double> all_solution(
                solution,
                solution + static_cast<std::size_t>(dimension) * nrhs);
            BatchedKktOperator target_operator(
                base_factorization.horizon, base_factorization.nx, base_factorization.nu,
                h_target, h_target_terminal, a_target);
            const auto residual_started = ProfileClock::now();
            max_residual = target_operator.relative_residual(
                all_rhs, all_solution, nrhs);
            timing.residual += profile_seconds(residual_started);
            converged = std::isfinite(max_residual) &&
                        max_residual <= tolerance;
        }
        if (!converged)
            return fail(1, "parallel GMRES did not reach the requested tolerance");

        stats->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        stats->mean_iterations = weighted_iterations / nrhs;
        stats->mean_cached_backsolves = target_matches_forward_system
            ? 1.0
            : stats->mean_iterations + 1.0;
        stats->relative_residual = max_residual;
        store_profile(*stats, timing);
        return 0;
    }
    catch (const std::exception &error)
    {
        return fail(-3, error.what());
    }
    catch (...)
    {
        return fail(-3, "unknown exception during parallel target solve");
    }
}
