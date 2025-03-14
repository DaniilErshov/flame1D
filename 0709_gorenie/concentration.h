#pragma once
#include "functions_sundials.h"
#include "chemkinReader.h"
#include "boost/regex.hpp"
extern bool save_chem_koeffs;

double my_mol_weight(int k);

void make_averageY(double* Yavg, double* Yi, double* Yipred);

void Get_mole_fr(double* X, double* Y);

void Get_molar_cons(double* X, double* Y, double T);

void get_Y(double Y_H2O, double& Y_H2, double& Y_O2, double Y_N2);

double get_W(double* Y);

double get_rho(double* Y, double T);

double get_GradRho(double* Yi, double* Yinext, double x, double xnext, double Ti, double Tinext);

void get_grad(double* gradX, double* Xi, double* Xinext, double x, double xnext);

double YkVk_func(int k, double T, double* Y, double* gradX, double* Xi, double* Y_average);

double Dij_func5(int i, int j, double T, int number_cell);

void moleFraction_to_massFraction(double* X, double* Y);

void chem_vel(double* Sn, double* Hn, double* forward, double* reverse, double* equilib, double Tcurr, double* y, double* yprime, int num_cell);