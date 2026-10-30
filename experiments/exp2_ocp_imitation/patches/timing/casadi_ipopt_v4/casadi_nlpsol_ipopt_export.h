#pragma once
#ifdef CASADI_NLPSOL_IPOPT_BUILD
#define CASADI_NLPSOL_IPOPT_EXPORT __declspec(dllexport)
#else
#define CASADI_NLPSOL_IPOPT_EXPORT __declspec(dllimport)
#endif
