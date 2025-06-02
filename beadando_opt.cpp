#include <iostream>
#include <fstream>
#include <vector>
#include <future>
#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>


// Duffing egyenlet paraméterei
const double delta = 0.02;
const double alpha = 1.0;
const double beta = 5.0;
const double gamma = 8;
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
    ds.y = -delta * s.y - alpha * s.x - beta * s.x * s.x * s.x + gamma * std::cos(omega * t); // [\,opt cosinuszt?]
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


// A párhuzamos szimulációkat elindító main függvény, annyi kezdeti feltételből indít el futást, ahány szál a gépen elérhető
int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int szalak = std::thread::hardware_concurrency(); // ennyi szál érhető el

    std::vector<std::future<void>> futures;

    for (int i = 0; i < szalak; ++i)
    {
        double x0 = 0.1 + 0.05 * i; // különböző kezdeti x értékek
        futures.push_back(std::async(std::launch::async, run_simulation, x0, i));
    }

    // Megvárjuk, míg az összes szimuláció befejeződik
    for (auto &f : futures)
    {
        f.get();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Összesen " << szalak << " szimuláció futott le pacekul " << duration << " ms alatt.\n";

    return 0;
}
