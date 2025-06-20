#include <iostream>
#include <fstream>
#include <cmath>

#define DELTA 0.02
#define ALPHA 1.0
#define BETA 5.0
#define GAMMA 8.0
#define OMEGA 0.5

struct State
{
    double x;
    double y;
};

// Duffing egyenlet CUDA device függvény
__device__ State duffing_rhs(State state, double t)
{
    State ds;
    ds.x = state.y;
    ds.y = -DELTA * state.y - ALPHA * state.x - BETA * state.x * state.x * state.x + GAMMA * cos(OMEGA * t);
    return ds;
}

// RK4 CUDA device függvény
__device__ State rk4_step(State state, double t, double dt)
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

// CUDA kernel: minden szál egy initial_x-hez tartozó szimulációt futtat
__global__ void run_simulation_kernel(
    double *initial_xs, double *phase_x, double *phase_y,
    int num_initials, int num_steps, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_initials)
        return;

    double x0 = initial_xs[idx];
    State state = {x0, 0.0};
    double t = 0.0;

    for (int step = 0; step < num_steps; ++step)
    {
        state = rk4_step(state, t, dt);
        phase_x[idx * num_steps + step] = state.x;
        phase_y[idx * num_steps + step] = state.y;
        t += dt;
    }
}

int main()
{
    int num_initials = 1000;
    int num_steps = 1000; // például 100.0 idő, 0.01 lépésköz
    double dt = 0.01;

    // Host oldali adatok
    double *h_initial_xs = new double[num_initials];
    double *h_phase_x = new double[num_initials * num_steps];
    double *h_phase_y = new double[num_initials * num_steps];

    for (int i = 0; i < num_initials; ++i)
        h_initial_xs[i] = 0.1 + 0.05 * i;

    // Device oldali adatok
    double *d_initial_xs, *d_phase_x, *d_phase_y;
    cudaMalloc(&d_initial_xs, num_initials * sizeof(double));
    cudaMalloc(&d_phase_x, num_initials * num_steps * sizeof(double));
    cudaMalloc(&d_phase_y, num_initials * num_steps * sizeof(double));

    cudaMemcpy(d_initial_xs, h_initial_xs, num_initials * sizeof(double), cudaMemcpyHostToDevice);

    // Időmérés kezdete
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel indítása
    int blockSize = 128;
    int gridSize = (num_initials + blockSize - 1) / blockSize;
    run_simulation_kernel<<<gridSize, blockSize>>>(d_initial_xs, d_phase_x, d_phase_y, num_initials, num_steps, dt);
    cudaDeviceSynchronize();

    // Időmérés vége
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel futási ideje: " << milliseconds << " ms" << std::endl;

    // Szálak és szimulációk összefoglalása
    std::cout << "Összefoglaló:" << std::endl;
    std::cout << "  Szálak száma (blockSize): " << blockSize << std::endl;
    std::cout << "  Grid méret (gridSize): " << gridSize << std::endl;
    std::cout << "  Összes indított CUDA szál: " << (blockSize * gridSize) << std::endl;
    std::cout << "  Szimulációk száma: " << num_initials << std::endl;
    std::cout << "  Egy szimuláció lépései: " << num_steps << std::endl;
    std::cout << "  Teljes futási idő: " << milliseconds << " ms" << std::endl;

    // Eredmények visszamásolása
    cudaMemcpy(h_phase_x, d_phase_x, num_initials * num_steps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phase_y, d_phase_y, num_initials * num_steps * sizeof(double), cudaMemcpyDeviceToHost);

    // Kiírás fájlba (kezdeti feltételek szerint, üres sorokkal elválasztva)
    std::ofstream fout("duffing_cuda_phase.csv");
    for (int i = 0; i < num_initials; ++i)
    {
        fout << "# initial_x = " << h_initial_xs[i] << "\n";
        fout << "x,y\n";
        for (int step = 0; step < num_steps; ++step)
        {
            fout << h_phase_x[i * num_steps + step] << "," << h_phase_y[i * num_steps + step] << "\n";
        }
        fout << "\n\n";
    }
    fout.close();

    // Felszabadítás
    cudaFree(d_initial_xs);
    cudaFree(d_phase_x);
    cudaFree(d_phase_y);
    delete[] h_initial_xs;
    delete[] h_phase_x;
    delete[] h_phase_y;

    return 0;
}