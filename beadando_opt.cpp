#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <future>
#include <chrono>
#include <algorithm>



// Duffing egyenlet paraméterei, amelyek a wikipedian szerepelnek https://en.wikipedia.org/wiki/Duffing_equation
const double DELTA = 0.02;
const double ALPHA = 1.0;
const double BETA = 5.0;
const double GAMMA = 8;
const double OMEGA = 0.5;

// Állapot vektor, structként csináltam, 
struct State
{
    double x;
    double y;
};
                        
// Szimuláció eredményének típusa (fázistér + Poincaré)
struct SimulationResult {
    double initial_x; // azért, hogy tudjuk, melyik kezdőfeltételhez tartozik az eredmény
    std::vector<std::pair<double, double>> phase;
    std::vector<std::pair<double, double>> poincare;
};

// Duffing rendszer jobb oldala (deriváltak)
State duffing_rhs(const State &state, double t)          //Itt hivatkozok az állapotra így nem változtatja
{
    State ds;
    ds.x = state.y;
    ds.y = -DELTA * state.y - ALPHA * state.x - BETA * state.x * state.x * state.x + GAMMA * std::cos(OMEGA * t);
    return ds;
}

// RK4 léptetés
State rk4_step(const State &state, double t, double dt)
{
    State k1 = duffing_rhs(state, t);
    State k2 = duffing_rhs({state.x + dt * k1.x / 2, state.y + dt * k1.y / 2}, t + dt / 2);
    State k3 = duffing_rhs({state.x + dt * k2.x / 2, state.y + dt * k2.y / 2}, t + dt / 2);
    State k4 = duffing_rhs({state.x + dt * k3.x, state.y + dt * k3.y}, t + dt);

    State next_state;
    next_state.x = state.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6;
    next_state.y = state.y + dt * (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6;
    return next_state;
}



// Szimuláció futtatása, eredmény visszaadása memóriában
SimulationResult run_simulation(double initial_x)
{
    double dt = 0.01;
    double period = 2 * M_PI / OMEGA;

    // Fázistér szimuláció
    double t_max_phase = 1000.0;
    State state = {initial_x, 1};   //a kezdeti sebesseg erteke 1

    std::vector<std::pair<double, double>> phase_data; // fázistér adatok tárolása, hogy vissza tudja küldeni a száll


    for (double t = 0; t < t_max_phase; t += dt)
    {
        state = rk4_step(state, t, dt);
        phase_data.emplace_back(state.x, state.y); // fázistér adat hozzáadása 
        //az emplace_back segítségével, ami hatékonyabb, mint a push_back

    }

    // Poincaré szimuláció
    double t_max_poincare = 10000.0;
    state = {initial_x, 1.0};
    double next_poincare_time = 0.0;


    std::vector<std::pair<double, double>> poincare_data; // Poincaré adatok tárolása, hogy vissza tudja küldeni a száll


    for (double t = 0; t < t_max_poincare; t += dt)
    {
        state = rk4_step(state, t, dt);
        if (std::abs(t - next_poincare_time) < dt / 2.0)
        {
            poincare_data.emplace_back(state.x, state.y);
            next_poincare_time += period;
        }
    }

    return {initial_x, phase_data, poincare_data}; // visszatér a szimuláció eredményével, nem ír a fájlba
}                                                   //az előző verzióhoz képest, ahol minden szál külön kiírta

int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = std::thread::hardware_concurrency();
    int num_initials = 50; // Ennyi pontot szeretnénk követni a fázistérben 

    // Generáljuk a kezdőfeltételeket
    std::vector<double> initial_xs(num_initials);
    for (int i = 0; i < num_initials; ++i)
        initial_xs[i] = 1 + (1.0/30.0) * i;

    // Minden szálra kiosztjuk a kezdőfeltételek egy részét
    std::vector<std::future<std::vector<SimulationResult>>> futures; // a future nem void értékű,
    // hanem a korábban létrehozott SimulationResult típusú structokat tárolja, hogy az eredmények visszatérjenek a fő 
    // szálra és kiirhassuk azokat


    for (int t = 0; t < num_threads; ++t)
    {
        futures.push_back(std::async(std::launch::async, [&, t]() { // az eddigiekhez hasonlóan inditjuk a szálakat
            // MInden száll egy lambda függvényt futtat, ami a kezdőfeltételek egy részét dolgozza fel
            std::vector<SimulationResult> results; // minden száll saját eredmény vektort hoz létre
            for (int i = t; i < num_initials; i += num_threads) // Ha a szállak száma pl 4 akkor a 0. szál az 0, 4, 8, ...
            //kezdőfeltételeket dolgozza fel, az 1. szál az 1, 5, 9, ... kezdőfeltételeket     
                results.push_back(run_simulation(initial_xs[i]));
            return results;
        }));
    }

    // Eredmények összegyűjtése
    std::vector<SimulationResult> all_results;
    for (auto& fut : futures)   //bonyolult a futures vektor, az elemek simres struktokat tartalmazó vektorok, így 
    {                              //auto betippeli ezt 
        auto res = fut.get(); // megvárja amig a száll végez es lekériaz eredményát
        all_results.insert(all_results.end(), res.begin(), res.end()); // c+++11-től kezdve a vektorok összefűzése
        // hatékonyabb, mint a korábbi push_back módszer, így nem kell külön vektorokat létrehozni
    }

    // Rendezés kezdőfeltétel szerint, annak segitsegevel hogy a structunk SimulationResult tartalmaz egy initial_x mezőt
    std::sort(all_results.begin(), all_results.end(), [](const SimulationResult& a, const SimulationResult& b) {
        return a.initial_x < b.initial_x;   //ez a lambda compare function 
    }); // a plottolás miatt nem kéne sorbarendezni, csak ha az egymás melletti kezdőfeltételeket
        //vizsgáljuk adott időre, akkor ez segít, így könnyebb megfigyelni a kaotikus viselkedést

    // Fázistér eredmények kiírása, kezdőfeltételenként, üres sorokkal elválasztva
    std::ofstream fout_phase("duffing_all_phase.csv");
    for (const auto& result : all_results)
    {
        fout_phase << "# initial_x = " << result.initial_x << "\n";
        fout_phase << "x,y\n";
        for (const auto& [x, y] : result.phase)
            fout_phase << x << "," << y << "\n";
        fout_phase << "\n\n";
    }
    fout_phase.close();

    // Poincaré eredmények kiírása, kezdőfeltételenként, üres sorokkal elválasztva
    std::ofstream fout_poincare("duffing_all_poincare.csv");
    for (const auto& result : all_results)
    {
        fout_poincare << "# initial_x = " << result.initial_x << "\n";
        fout_poincare << "x,y\n";
        for (const auto& [x, y] : result.poincare)
            fout_poincare << x << "," << y << "\n";
        fout_poincare << "\n\n";
    }
    fout_poincare.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Összesen " << num_initials << " szimuláció futott le " << num_threads << " szálon " << duration << " ms alatt pacekul.\n";

    return 0;
}