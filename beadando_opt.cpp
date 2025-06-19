#include <iostream>
#include <fstream>
#include <vector>
#include <future>
#include <chrono>
#include <thread>
#define _USE_MATH_DEFINES
#include <cmath>


// Duffing egyenlet paraméterei
const double delta = 0.02;
const double alpha = 1.0;
const double beta = 5.0;

const double gamma_coeff = 8;

const double omega = 0.5;

// Állapot vektor: x és y    2 double a structban, 16 bájtot foglal a memóriában, egyszerűbb vele de futni ugyanolyan gyorsan fut 
struct Állapot
{
    double x;
    double y;
};

// Duffing rendszer jobb oldala (deriváltak)
Állapot duffing_rhs(const Állapot &s, double t)
{               // bemenetnek vár egy structot amit nem belemásolok, hanem a memóriacímet kapja meg de nem változtathatja, ez így gyorsabb 
    Állapot ds; //
    ds.x = s.y;

    ds.y = -delta * s.y - alpha * s.x - beta * s.x * s.x * s.x + gamma_coeff * std::cos(omega * t); // [\,opt cosinuszt?]

    return ds;
}

// RK4 léptetés, így egyszerű rendszer esetén fix lépésközzel költséghatékonyabb kézzel megírt függvényt használni
Állapot rk4_step(const Állapot &s, double t, double dt)
{
    Állapot k1 = duffing_rhs(s, t);
    Állapot k2 = duffing_rhs({s.x + dt * k1.x / 2, s.y + dt * k1.y / 2}, t + dt / 2);
    Állapot k3 = duffing_rhs({s.x + dt * k2.x / 2, s.y + dt * k2.y / 2}, t + dt / 2);
    Állapot k4 = duffing_rhs({s.x + dt * k3.x, s.y + dt * k3.y}, t + dt);

    Állapot s_next;
    s_next.x = s.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6;
    s_next.y = s.y + dt * (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6;
    return s_next;
}


void run_simulation(double x0, int index)
{
    double dt = 0.01;
    double T = 2 * M_PI / omega;

    // Fázistér szimuláció 
    double t_max_fazis = 1000.0;  
    Állapot s = {x0, 0.0};

    std::ofstream fout_phase("duffing_out_" + std::to_string(index) + ".csv");
    fout_phase << "x,y\n";

    for (double t = 0; t < t_max_fazis; t += dt)
    {
        s = rk4_step(s, t, dt);
        fout_phase << s.x << "," << s.y << "\n";
    }

    fout_phase.close();

    // Poincaré szimuláció, ez hosszabb, hogy mutassa a kaotikus viselkedést
    double t_max_poincare = 10000000.0;  


    s = {x0, 0.0};
    double next_poincare_time = 0.0;

    std::ofstream fout_poincare("duffing_out_poincare_" + std::to_string(index) + ".csv");
    fout_poincare << "x,y\n";

    for (double t = 0; t < t_max_poincare; t += dt)
    {
        s = rk4_step(s, t, dt);
        if (std::abs(t - next_poincare_time) < dt / 2.0)
        {
            fout_poincare << s.x << "," << s.y << "\n";
            next_poincare_time += T;
        }
    }

    fout_poincare.close();
}


void run_simulation_batch(int start_idx, int count, double x_start, double x_end, int total_points)
{
    for (int j = 0; j < count; ++j)
    {
        int idx = start_idx + j;
        double ratio = total_points > 1 ? static_cast<double>(idx) / (total_points - 1) : 0.0;
        double x0 = x_start + ratio * (x_end - x_start);
        run_simulation(x0, idx);
    }
}


// A párhuzamos szimulációkat elindító main függvény
int main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int total_points = 10;
    double interval_start = 0.0;
    double interval_end = 1.0;

    if (argc > 1)
        total_points = std::stoi(argv[1]);
    if (argc > 2)
        interval_start = std::stod(argv[2]);
    if (argc > 3)
        interval_end = std::stod(argv[3]);

    if (total_points <= 0)
    {
        std::cerr << "A kezdőpontok számának nagyobbnak kell lennie nullánál.\n";
        return 1;
    }

    unsigned int szalak = std::thread::hardware_concurrency();
    if (szalak == 0)
        szalak = 1;
    if (static_cast<unsigned int>(total_points) < szalak)
        szalak = total_points;

    std::vector<std::future<void>> futures;

    int points_per_thread = total_points / szalak;
    int remainder = total_points % szalak;
    int start_index = 0;
    for (unsigned int i = 0; i < szalak; ++i)
    {
        int count = points_per_thread + (i < static_cast<unsigned int>(remainder) ? 1 : 0);
        int thread_start = start_index;
        futures.push_back(std::async(std::launch::async, [=]() {
            run_simulation_batch(thread_start, count, interval_start, interval_end, total_points);
        }));
        start_index += count;
    }

    for (auto &f : futures)
    {
        f.get();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Összesen " << total_points << " szimuláció futott le " << duration << " ms alatt.\n";

    return 0;
}
