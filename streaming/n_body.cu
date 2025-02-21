#include "files.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct {
  float x, y, z, vx, vy, vz;
} Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

__global__ void bodyForce(Body *p, float dt, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float Fx = 0.0f;
  float Fy = 0.0f;
  float Fz = 0.0f;

  for (int j = 0; j < n; j++) {
    float dx = p[j].x - p[idx].x;
    float dy = p[j].y - p[idx].y;
    float dz = p[j].z - p[idx].z;

    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    Fx += dx * invDist3;
    Fy += dy * invDist3;
    Fz += dz * invDist3;
  }

  p[idx].vx += dt * Fx;
  p[idx].vy += dt * Fy;
  p[idx].vz += dt * Fz;
}

__global__ void integratePositions(Body *p, float dt, int n) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  p[idx].x += p[idx].vx * dt;
  p[idx].y += p[idx].vy * dt;
  p[idx].z += p[idx].vz * dt;
}

int main(const int argc, const char **argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody
  // report files
  int nBodies = 2 << 11;
  if (argc > 1)
    nBodies = 2 << atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for
  // correctness. You should not make changes to these files, or else the
  // assessment will not work.
  const char *initialized_values;
  const char *solution_values;

  if (nBodies == 2 << 11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2)
    initialized_values = argv[2];
  if (argc > 3)
    solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);

  // Get device props
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount,
                         deviceId);

  Body *p;
  cudaMallocManaged(&p, bytes);

  read_values_from_file(initialized_values, (float *)p, bytes);

  // Prefetch data to GPU
  cudaMemPrefetchAsync(p, bytes, deviceId);

  const int threadsPerBlock = BLOCK_SIZE;
  const int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

  // Create cuda streams
  cudaStream_t computeStream, integrationStream;
  cudaStreamCreate(&computeStream);
  cudaStreamCreate(&integrationStream);

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    /*
     * You will likely wish to refactor the work being done in `bodyForce`,
     * and potentially the work to integrate the positions.
     */

    bodyForce<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(
        p, dt, nBodies); // compute interbody forces
    cudaStreamSynchronize(computeStream);

    /*
     * This position integration cannot occur until this round of `bodyForce`
     * has completed. Also, the next round of `bodyForce` cannot begin until the
     * integration is complete.
     */

    integratePositions<<<blocksPerGrid, threadsPerBlock, 0,
                         integrationStream>>>(p, dt, nBodies);
    cudaStreamSynchronize(integrationStream);

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  // prefetch data back to CPU for verification
  cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);
  cudaDeviceSynchronize();

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, (float *)p, bytes);

  // You will likely enjoy watching this value grow as you accelerate the
  // application, but beware that a failure to correctly synchronize the device
  // might result in unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  // Cleanup
  cudaStreamDestroy(computeStream);
  cudaStreamDestroy(integrationStream);
  cudaFree(p);
}
