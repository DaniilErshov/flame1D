#pragma once
#include <iostream>
#include <fstream>
#include "constants.h"
#include "concentration.h"
#include "thermodynamic.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cmath>
#include "chemkinReader.h"
#include "boost/regex.hpp"
#include "json.h"
using namespace std;

#include <ida/ida.h>   
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */
#include <sundials/sundials_math.h>     /* access to SUNRexp               */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
//#include <cvode/cvode.h>   
#include <nvector/nvector_openmp.h>
#include <omp.h>

#define FTOL   RCONST(1.e-9)/* function tolerance */
#define STOL   RCONST(1.e-25) /* step tolerance     */

#define ZERO   RCONST(0.0)
#define PT25   RCONST(0.25)
#define PT5    RCONST(0.5)
#define ONE    RCONST(1.0)
#define ONEPT5 RCONST(1.5)
#define TWO    RCONST(2.0)

#define Ith(v,i)    NV_Ith_S(v,i-1)
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1)


#define PI     RCONST(3.1415926)

/* Problem Constants */
extern thread_local double M_global;
extern int num_gas_species;
extern int num_react;
extern double k_mol;
extern double Y_N2;
extern double Y_max;
extern double P;
extern double R;
extern double x_center;
extern double koeff_l;
extern double l;
extern double* norm;
extern long int myiter;
extern long int nniters;
extern double eps_x;
extern double eps_func;
extern double eps_fr;
extern double Tstart;
extern double Tfinish;
extern double nevyaz_Y;
extern double nevyaz_T;
extern const double kB;
extern const double Angstroem__;
extern const double santimetr;
extern thread_local double** Dij_res;
extern double ftol;
extern double stoler;
extern thread_local double* Yi;
extern thread_local double* Yiprev;
extern thread_local double* Yinext;
extern thread_local double rho_l, rho_r, Vc_l, Vc_r;

extern thread_local double* YkVk_r;
extern thread_local double* YkVk_l;

extern thread_local double* gradX_r;
extern thread_local double* gradX_l;

extern thread_local double* X_tmp_r;
extern thread_local double* X_tmp_l;

extern thread_local double* Y_tmp_r;
extern thread_local double* Y_tmp_l;

extern thread_local double* Xiprev;
extern thread_local double* Xi;
extern thread_local double* Xinext;

extern thread_local double* gradX;

extern thread_local double* Y_tmp;
extern thread_local double* X_tmp;
extern thread_local double* YkVk;

extern thread_local double* Sn;
extern thread_local double* Hn;
extern thread_local double* Cpn;

extern thread_local  double* forward_arr;
extern thread_local double* reverse_arr;
extern thread_local double* equilib_arr;
extern  double* Y_left_bound;
extern thread_local double* wk_add;
extern thread_local double* ydot;
extern double*** diff_polynom;
extern double** lambda_polynom;
extern double* mol_weight;
extern int jactimes;
extern vector<vector<double>> Cp_arr;
extern vector<vector<double>> H_arr;
extern vector<vector<double>> Lambda_arr;
extern vector<vector<double>> Lambda_arr_r;
extern vector<vector<double>> Lambda_arr_l;
extern vector<vector<vector<double>>> Dij_arr;
extern vector<vector<vector<double>>> Dij_arr_r;
extern vector<vector<vector<double>>> Dij_arr_l;
extern vector<vector<double>> forward_arr_save;
extern vector<vector<double>> reverse_arr_save;


extern int num_threads_global;

extern int ida_steps;
extern double eps;
extern vector<string> name_species;
extern std::unordered_map<std::string, int> komponents;
extern std::unordered_map<int, string> komponents_str;
extern std::unordered_map<int, std::unordered_map<string, double>> Dij_saved;
extern std::map<std::string, int> mykomponents;
extern std::map<int, string> mykomponents_str;
extern vector<double> x_vect;
extern vector<double> Y_vect;
extern vector<double> T_vect;
extern bool flag_use_save_koeffs;
extern bool saved_chem_koeffs;
extern bool need_update_koeffs;
extern int N_x;
typedef struct {
    int N_m;
    int NEQ;
    int NEQ_Y;
    int N_centr;
    int my_numjac;
    double my_tcur;
    void* sun_mem;
    realtype Tl;
    realtype T_center;
    IO::ChemkinReader* chemkinReader;
} *UserData;

void resize_koeff_vectors(int N_x);

void updateKoeffs(double* yval, UserData data);

static int check_retval(void* retvalvalue, const char* funcname, int opt);

double F_right(UserData data,
    double Tprev, double T, double Tnext, double xprev, double x, double xnext, int number_cell);

double F_rightY(UserData data, int k_spec,
    double Tprev, double T, double Tnext, double xprev, double x, double xnext, int number_cell);


int InitialData(int& Nx, vector<double>& x_vect, vector<double>& T_vect, vector<double>& Y_vect, double& M, double Tstart, double Tfinish, double* Ystart, double* Yend);

void Write_to_file(string str, ofstream& fout, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, vector<double>& Yp_vect, double M, int N_x, int number);

int integrate_Y_IDA(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, double t_fix);

static int func_Y_IDA(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);

int integrate_All_IDA(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter, double t_fix, double M_rtol_need);

static int func_All_IDA(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);

double get_M(double* Yiprev, double* Yi, double* Yinext,
    double Tprev, double T, double Tnext, double xprev, double x, double xnext, double* Xiprev, double* Xi, double* Xinext, double* gradX, double* Y_tmp, double* X_tmp,
    double M, double* ydot, double* wk_add);

void Add_elem_simple(vector<double>& T, vector<double>& Y, vector<double>& x, int& N_x, int& N_center, double b, int number, int number_start, double& T_center);

void Init_Data(UserData data, int N_x, vector<double>& x_vect,
    vector<double>& T_vect, int NEQ,
    int N_center, double* Y_leftb);

static int func_Y_IDA(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);

int integrate_Y_IDA(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb);

int Integrate_Kinsol(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter);

static int func_kinsol(N_Vector u, N_Vector f, void* user_data);

void MakeYvectors(UserData data,
    double* Y, int myNx, int i, double Tl);

int Find_final_state_IDA(double& Tinitial, double& Tend, double* Y_vect, double* Y_end);
static int func_final_state(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);

void makeYstart(double koeff_topl, double* Ystart);

void find_diff_slag(UserData data, double Tcurr, double Tnext, double* Yi, double* Yinext,
    double* Xi, double* Xinext, double* Ykvk_side, double* Y_tmp_side, double* X_tmp_side, double* gradX_side, double& rho_side, double& Vc_side, int i, char side);

void MakeYvectorsY(UserData data,
    double* Y, int myNx, int i, double Tl);

double get_M(double Tprev, double T, double Tnext,
    double xprev, double x, double xnext, int number_cell);


int Integrate_Kinsol_dense(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter);

static int func_kinsol_dense(N_Vector u, N_Vector f, void* user_data);


void MakeYvectors_dense(UserData data,
    double* Y, int myNx, int i, double Tl);

void MakeYvectors_kins(UserData data,
    double* Y, int myNx, int i, double Tl);


int integrate_All_IDA_M(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter, double t_fix, double M_rtol_needed);


static int func_All_IDA_M(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);

void set_Dij_res(double T, int number_cell, char side);


int integrate_All_IDA_dense(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter, double t_fix);


static int func_All_IDA_dense(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);


int Integrate_Kinsol_withoutM(int N_x, vector<double>& x_vect,
    vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, double* Y_leftb, int iter);

static int func_kinsol_withoutM(N_Vector u, N_Vector f, void* user_data);

double F_right_IDA(UserData data,
    double Tprev, double T, double Tnext, double xprev, double x, double xnext, int i);


int Find_final_state_KINSOL(double& Tinitial, double& Tend, double* Y_vect, double* Y_end);

static int func_final_state_kinsol(N_Vector u, N_Vector f, void* user_data);

void makeYstart(double koeff_topl, string fuel, double O2_in, double N2_in, double* Ystart);