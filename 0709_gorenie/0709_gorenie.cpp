#include "functions_sundials.h"

double k_mol = pow(10, 3);
double Y_N2 = 0.745187;
double Y_max = 1 - Y_N2;
double P = 0.101325;
double P_atm;
double R = 8.314;
double koeff_l = 0.4;
double l = 0.1;
double x_center;
long int myiter = 0;
long int nniters;
double eps_func = pow(10, -8);
double* norm;
double Tstart = 700;
double Tfinish = 2385.4;
double nevyaz_Y;
double nevyaz_T;
double eps_x = pow(10, -6);
double eps_fr = pow(10, -6);
const double kB = 1.3806504e-23;
const double Angstroem__ = 1.0e-10;
const double santimetr = 1.0e-8;


//////////////////////////////
int num_threads_global = 2;///
/////////////////////////////





thread_local double* Yi;
thread_local double* Yiprev;
thread_local double* Yinext;

thread_local double* YkVk_r;
thread_local double* YkVk_l;

thread_local double* gradX_r;
thread_local double* gradX_l;

thread_local double* X_tmp_r;
thread_local double* X_tmp_l;

thread_local double* Y_tmp_r;
thread_local double* Y_tmp_l;

thread_local double* Xiprev;
thread_local double* Xi;
thread_local double* Xinext;

thread_local double* gradX;

thread_local double* Y_tmp;
thread_local double* X_tmp;
thread_local double* YkVk;
thread_local double** Dij_res;

thread_local double* Sn;
thread_local double* Hn;
thread_local double* Cpn;
thread_local double* ydot;

thread_local double rho_l, rho_r, Vc_l, Vc_r;
thread_local double M_global;

thread_local double* forward_arr;
thread_local double* reverse_arr;
thread_local double* equilib_arr;
double* Y_left_bound;
thread_local double* wk_add;
double*** diff_polynom;
double** lambda_polynom;
double* mol_weight;
 double* Ystart;
 double* Yend;
double eps = pow(10, -8);
double* X;
int jactimes = -1;
bool flag_use_save_koeffs = 0;
bool saved_chem_koeffs = 0;
bool need_update_koeffs = 1;

double ftol;
double stoler;
int num_gas_species;
int num_react;
int ida_steps;
vector<double> x_vect;
vector<double> Y_vect;
vector<double> T_vect;

vector<vector<double>> Cp_arr;
vector<vector<double>> H_arr;
vector<vector<double>> Lambda_arr;
vector<vector<double>> Lambda_arr_r;
vector<vector<double>> Lambda_arr_l;
vector<vector<vector<double>>> Dij_arr;
vector<vector<vector<double>>> Dij_arr_r;
vector<vector<vector<double>>> Dij_arr_l;
vector<vector<double>> forward_arr_save;
vector<vector<double>> reverse_arr_save;
vector<string> name_species;

std::unordered_map<std::string, int> komponents{
};

std::unordered_map<int, string> komponents_str{
};

map<string, double> elem_mol_weight{
    {"H", 1.00797},
    {"O", 15.9994},
    {"N", 14.0067},
    {"C", 12.0096},
    {"AR", 39.94},
    {"HE", 4.002602}
};

int N_x;

int main()
{
    string confname = "json_file//config.dat";

    //string confname = "powders.dat";
    Json::Value config;
    json_parse_file_or_die(confname, &config);
    Tstart = config["T_left"].asDouble();
    Tfinish = config["T_right"].asDouble();
    l = config["l"].asDouble();
    P_atm = config["P"].asDouble();
    P *= P_atm;
    double koeff_topl = config["koeff_fuel"].asDouble();
    N_x = config["Nx_start"].asInt();
    double u_initial = config["u_initial"].asDouble();

    num_threads_global = config["num_threads"].asInt();
    string Fuel = config["Fuel"].asString();

    const std::string thermfile = config["therm"].asString();
    const std::string transfile = config["tran"].asString();
    const std::string chemfile = config["chem"].asString();

    init_consts(chemfile, thermfile, transfile, num_gas_species, num_react);
    cout << "after init Ystart = " << Ystart << "\n";
    norm = new double;
    double b = 0.1;
    double M;
    double W, rho, Y_H2, Y_O2;
    int N_center;
    int retval;
    double w_dot;
    double* my_x;
    ofstream fout_v; 
    ofstream fout;
    double X_H2O, X_H2, X_O2, X_N2;
    cout << "pre omp Ystart = " << Ystart << "\n";
#pragma omp parallel num_threads(num_threads_global)
    {
#pragma omp critical
        {
            int thread_id = omp_get_thread_num();
            allocate_paralel_memory();
        }
    }
    cout << "aftert omp Ystart = " << Ystart << "\n";

    double tout1 = pow(10, -7);
    double Tend = Tfinish;
    double T_start = Tstart;
    double T_finish = Tfinish;
    double T_center;

    x_vect.resize(N_x); Y_vect.resize(N_x * num_gas_species); T_vect.resize(N_x);
    resize_koeff_vectors(N_x);

    makeYstart(koeff_topl, Fuel, 0.21, 0.79, Ystart);

    Find_final_state_IDA(T_start, Tend, Ystart, Yend);
    Find_final_state_KINSOL(T_start, Tend, Ystart, Yend);

    M = u_initial * get_rho(Ystart, Tstart);
    int j_t = 1;
    N_center = InitialData(N_x, x_vect, T_vect, Y_vect, M, T_start, Tend, Ystart, Yend);
    Tfinish = Tend;

    Write_to_file("detail/initial" + to_string(N_x), fout, x_vect,
        T_vect, Y_vect, Y_vect, M, N_x, 1);

    double t_Y = pow(10, -7), t_full = pow(10, -6);
    T_center = T_vect[N_center];

    double M_rtol_need = 0;
    ida_steps = pow(10, 2);
    eps = pow(10, -5);
    t_full = pow(10, -6);

    integrate_All_IDA(N_x, x_vect,
        T_vect, Y_vect, M, N_center, Ystart, 1, t_full, M_rtol_need);
    Write_to_file("detail/IDA_" + to_string(N_x), fout, x_vect,
        T_vect, Y_vect, Y_vect, M, N_x, 1);
    while (N_x < 80)
    {
        ida_steps = pow(10, 5);
        eps = pow(10, -5);
        t_full = pow(10, -5);
        M_rtol_need = 1e-12;
        integrate_All_IDA_M(N_x, x_vect,
            T_vect, Y_vect, M, N_center, Ystart, 1, t_full, M_rtol_need);
        Write_to_file("detail/IDA_M_" + to_string(N_x), fout, x_vect,
            T_vect, Y_vect, Y_vect, M, N_x, 1);


        Integrate_Kinsol(N_x, x_vect,
            T_vect, Y_vect, M, N_center, Ystart, 6);

        Write_to_file("detail/KINSOL_" + to_string(N_x), fout, x_vect,
            T_vect, Y_vect, Y_vect, M, N_x, 1);

        Add_elem_simple(T_vect, Y_vect, x_vect, N_x, N_center, 0.01, 2, 2, T_center);
    }

    int number_epoch = 0;
    int add_cell_end = 2;
    int add_cell_start = 2;
    ftol = pow(10, -6);
    stoler = pow(10, -20);

    while (N_x < 500)
    {
        cout << to_string(koeff_topl) << " N_x = " << N_x << "\n";
        flag_use_save_koeffs = 0;
        saved_chem_koeffs = 0;
        need_update_koeffs = 0;

        eps = pow(10,-5);
        Integrate_Kinsol(N_x, x_vect,
            T_vect, Y_vect, M, N_center, Ystart, 6);

        Write_to_file("detail/KINSOL_" + to_string(N_x), fout, x_vect,
            T_vect, Y_vect, Y_vect, M, N_x, 1);
        cout << to_string(koeff_topl) << " v = " << M / get_rho(Ystart, Tstart) << "\n";
        cout << to_string(koeff_topl) << " Tend = " << T_vect[T_vect.size() - 1] << "\n";
        cout << " norm = " << norm[0] << "\n";
        cout << " x_right = " << x_vect[x_vect.size() - 1] << "\n";

        Add_elem_simple(T_vect, Y_vect, x_vect, N_x, N_center, 0.01, add_cell_end, add_cell_start, T_center);
    }



    free_memory();
    return 0;
}