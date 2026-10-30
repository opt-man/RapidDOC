#ifndef FRS_MINGW_IPOPT_CONFIG_H
#define FRS_MINGW_IPOPT_CONFIG_H
#include "config_ipopt.h"
#define HAVE_WINDOWS_H 1
#define HAVE_CSTDDEF 1
#define HAVE_CSTDIO 1
#define HAVE_CSTDLIB 1
#define HAVE_CSTRING 1
#define HAVE_CFLOAT 1
#define HAVE_CMATH 1
#define HAVE_CASSERT 1
#define HAVE_CSTDARG 1
#define HAVE_CTIME 1
#define HAVE_INTTYPES_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDDEF_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_MEMORY_H 1
#define HAVE_UNISTD_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_VA_COPY 1
#define IPOPT_HAS_VA_COPY 1
#define IPOPT_HAS_RAND 1
#define IPOPT_HAS_LAPACK 1
#define IPOPT_HAS_MUMPS 1
#define IPOPT_C_FINITE std::isfinite
#define F77_FUNC(name,NAME) name ## _
#define F77_FUNC_(name,NAME) name ## _
#define IPOPT_BLAS_FUNC(name,NAME) name ## _
#define IPOPT_LAPACK_FUNC(name,NAME) name ## _
#define IPOPT_PARDISO_FUNC(name,NAME) name ## _
#define IPOPT_WSMP_FUNC(name,NAME) name ## _
#define IPOPT_WSMP_FUNC_(name,NAME) name ## _
#endif
