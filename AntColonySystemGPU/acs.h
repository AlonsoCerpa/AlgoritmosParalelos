#ifndef ACS_H
#define ACS_H

#include <vector>
#include <list>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <limits>

#include "create_graph.h"

#define NODES_SIZE_CT 198
#define NUM_THREADS_WARP 32
#define NUM_THREADS_GLOBAL_PU 128

float get_euc_dist(Coordfloat &a, Coordfloat &b)
{
    return sqrt(pow(b.x - a.x, 2.0) + pow(b.y - a.y, 2.0));
}

float get_nn_distance(std::vector<Coordfloat> &nodes, int &node_idx, std::list<int> &unvisited_nodes)
{
    float min_dist = std::numeric_limits<float>::max();
    float curr_dist;
    int next_node;
    std::list<int>::iterator findIter;
    for (auto it = unvisited_nodes.begin(); it != unvisited_nodes.end(); ++it)
    {
        curr_dist = get_euc_dist(nodes[node_idx], nodes[*it]);
        if (curr_dist < min_dist)
        {
            min_dist = curr_dist;
            next_node = *it;
            findIter = it;
        }
    }

    unvisited_nodes.erase(findIter);
    node_idx = next_node;
    return min_dist;
}

float nearest_neighbour_tsp(std::vector<Coordfloat> &nodes)
{
    int node_idx = 0;
    float cost = 0;
    std::list<int> unvisited_nodes;
    for (int i = 0; i < nodes.size(); ++i)
    {
        unvisited_nodes.push_back(i);
    }
    unvisited_nodes.erase(unvisited_nodes.begin());

    for (int i = 0; i < nodes.size() - 1; ++i)
    {
        cost += get_nn_distance(nodes, node_idx, unvisited_nodes);
    }
    unvisited_nodes.push_back(0);
    cost += get_nn_distance(nodes, node_idx, unvisited_nodes);

    return cost;
}

struct NodeCoordfloat
{
    int idx;
    Coordfloat coord;

    NodeCoordfloat () {}
    NodeCoordfloat(int i, Coordfloat c)
    {
        coord = c;
        idx = i;
    }
};


struct GreaterNC
{
    Coordfloat coord;
    GreaterNC(Coordfloat c)
    {
        coord = c;
    }
    bool operator()(NodeCoordfloat a, NodeCoordfloat b)
    {
        if (get_euc_dist(a.coord, coord) > get_euc_dist(b.coord, coord))
            return true;
        else
            return false;
    }
};

__global__
void setup_kernel(curandState *state)
{
    int id = threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

__global__
void build_ant_solutions_alt_k(float *d_pheromones, float *d_distances, int *d_cand_set,
                               int *d_remain, int *d_routes,
                               float *d_costs, int nodes_size, int warp_size,
                               curandState *state, float beta, float rho, float pheromone0)
{
    int id = threadIdx.x + (blockIdx.x * 32);
    int ant_idx = blockIdx.x;
    int warp_idx = threadIdx.x;

    __shared__ int routes[NODES_SIZE_CT];
    __shared__ bool tabu_list[NODES_SIZE_CT];

    int work_p_thread = nodes_size / warp_size;
    for (int i = 0; i < work_p_thread; ++i)
    {
        tabu_list[i+(work_p_thread*warp_idx)] = false;
    }
    int rem_work = nodes_size % warp_size;
    if (warp_idx == 0)
    {
        for (int i = 0; i < rem_work; ++i)
        {
            tabu_list[i+(work_p_thread*warp_size)] = false;
        }
    }

    curandState localState = state[ant_idx];
    float random_f;
    random_f = curand_uniform(&localState);
    int city0 = (int) truncf(random_f * ((nodes_size - 1) + 0.999999));

    if (warp_idx == 0)
    {
        routes[0] = city0;
        tabu_list[city0] = true;
    }

    int nn_city;
    int pred;
    unsigned res_ball;
    float v, aux, s, s_last;
    int idx_edge;
    int d;
    int warp_idx_next_n;
    float curr_dist;
    __shared__ float total_cost;
    __shared__ bool valid;
    //unsigned mask;


    if (warp_idx == 0)
    {
        total_cost = 0.0;
    }

    printf("NODES SIZE = %d\n", nodes_size);
    for (int i = 1; i < nodes_size; ++i)
    {
        nn_city = d_cand_set[(routes[i-1] * warp_size) + warp_idx];
        if (tabu_list[nn_city] == false)
            pred = 1;
        else
            pred = 0;

        res_ball = __ballot_sync(0xffffffff, pred);
        
        if (res_ball > 0) //hay ciudades validas en el cand_set => regla proporcional
        {
            idx_edge = (routes[i-1] * nodes_size) + nn_city;
            curr_dist = d_distances[idx_edge];
            v = d_pheromones[idx_edge] / pow(curr_dist, beta);
            s = v;
            printf("ANTES: id=%d, v=%.9f\n", id, v);
            for (int j = 0; j < warp_size - 1; ++j)
            {
                aux = __shfl_up_sync(0xffffffff, v, j+1);
                if (warp_idx > j)
                {
                    s += aux;
                }
            }
            printf("DESPUES: id=%d, v=%.9f, s=%.9f\n", id, v, s);
            s_last = __shfl_sync(0xffffffff, s, warp_size - 1);
            printf("s_last=%.9f\n", s_last);
            random_f = curand_uniform(&localState);
            printf("random_f=%f\n", random_f);
            random_f *= s_last;

            if (random_f >= s)
            {
                d = 1;
            }
            else
            {
                d = 0;
            }
            warp_idx_next_n = __popc(__ballot_sync(0xffffffff, d));
            if (warp_idx_next_n > warp_size - 1)
            {
                warp_idx_next_n = warp_size - 1;           
            }
            printf("warp_idx_next_n=%d\n", warp_idx_next_n);

            if (warp_idx == 0)
                valid = false;

            if (warp_idx == warp_idx_next_n)
            {
                if (tabu_list[nn_city] == false)
                {
                    total_cost += curr_dist;
                    routes[i] = nn_city;
                    tabu_list[nn_city] = true;
                    valid = true;
                }
            }
            while (valid == false)
            {
                warp_idx_next_n = (warp_idx_next_n + 1) % 32;
                if (warp_idx == warp_idx_next_n)
                {
                    if (tabu_list[nn_city] == false)
                    {
                        total_cost += curr_dist;
                        routes[i] = nn_city;
                        tabu_list[nn_city] = true;
                        valid = true;
                    }
                }
            }
        }
        else    //regla del valor maximo
        {
            printf("Valor maximo %d\n", id);
            int remain_size = (nodes_size - 1) - warp_size;
            work_p_thread = remain_size / warp_size;
            rem_work = remain_size % warp_size;
            int idx_remain = (routes[i-1] * remain_size) + (warp_idx * work_p_thread);
            int curr_city;

            float max = -1.0;
            int next_city_max = 0;

            for (int j = 0; j < work_p_thread; ++j)
            {
                curr_city = d_remain[idx_remain + j];
                if (tabu_list[curr_city] == false)
                {
                    idx_edge = (routes[i-1] * nodes_size) + curr_city;
                    v = d_pheromones[idx_edge] / pow(d_distances[idx_edge], beta);
                    if (v > max)
                    {
                        max = v;
                        next_city_max = curr_city;
                    }
                }
            }
            if (warp_idx == 0)
            {
                idx_remain = (routes[i-1] * remain_size) + (NUM_THREADS_WARP * work_p_thread);
                for (int j = 0; j < rem_work; ++j)
                {
                    curr_city = d_remain[idx_remain + j];
                    if (tabu_list[curr_city] == false)
                    {
                        idx_edge = (routes[i-1] * nodes_size) + curr_city;
                        v = d_pheromones[idx_edge] / pow(d_distances[idx_edge], beta);
                        if (v > max)
                        {
                            max = v;
                            next_city_max = curr_city;
                        }
                    }
                }
            }

            float otherMax;
            int otherNexCityMax;
            for (unsigned offset = warp_size / 2; offset > 0; offset /= 2) 
            {
                otherMax = __shfl_down_sync(0xffffffff, max, offset);
                otherNexCityMax = __shfl_down_sync(0xffffffff, next_city_max, offset);
                if (otherMax > max)
                {
                    max = otherMax;
                    next_city_max = otherNexCityMax;
                }
            }
            
            if (warp_idx == 0)
            {
                idx_edge = (routes[i-1] * nodes_size) + next_city_max;
                curr_dist = d_distances[idx_edge];
                total_cost += curr_dist;
                routes[i] = next_city_max;
                tabu_list[next_city_max] = true;
            }
        }

        //local update
        if (warp_idx == 0)
        {
            int idx_curr_pher = (routes[i-1] * nodes_size) + routes[i];
            float curr_pher = d_pheromones[idx_curr_pher];
            d_pheromones[idx_curr_pher] = ((1.0 - rho) * curr_pher) + (rho * pheromone0);
        }
    }

    if (warp_idx == 0)
    {
        int idx_curr_pher = (routes[nodes_size-1] * nodes_size) + routes[0];
        float curr_pher = d_pheromones[idx_curr_pher];
        d_pheromones[idx_curr_pher] = ((1.0 - rho) * curr_pher) + (rho * pheromone0);
    }

    work_p_thread = nodes_size / warp_size;
    rem_work = nodes_size % warp_size;
    int idx_routes = warp_idx * work_p_thread;
    int idx_d_routes = (nodes_size * ant_idx) + idx_routes;

    printf("work_p_thread=%d\n", work_p_thread);
    for (int i = 0; i < work_p_thread; ++i)
    {
        printf("id=%d, node=%d\n", id, routes[idx_routes + i]);
        d_routes[idx_d_routes + i] = routes[idx_routes + i];
    }
    if (warp_idx == 0)
    {
        idx_routes = work_p_thread * warp_size;
        idx_d_routes = (nodes_size * ant_idx) + idx_routes;
        for (int i = 0; i < rem_work; ++i)
        {
            printf("id=%d, node=%d\n", id, routes[idx_routes + i]);
            d_routes[idx_d_routes + i] = routes[idx_routes + i];
        }
        d_costs[ant_idx] = total_cost;
        printf("id=%d, cost=%.9f\n", id, total_cost);
    }
}

__global__
void global_pheromone_update_k(float *d_pheromones, int *d_routes, float *d_costs, float alpha,
                               int *d_global_best_route, int num_ants, int nodes_size,
                               float *d_cost_global_best_route)
{
    int num_threads = 128;
    int work_p_thread = num_ants / num_threads;
    int rem_work = num_ants % num_threads;
    int offset = work_p_thread * threadIdx.x;
    __shared__ float sh_costs[NUM_THREADS_GLOBAL_PU];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NUM_THREADS_GLOBAL_PU; ++i)
        {
            sh_costs[i] = 999999999.9999f;
        }
    }
    __shared__ int sh_idx[NUM_THREADS_GLOBAL_PU];
    float min = d_costs[0];
    int idx;
    for (int i = 0; i < work_p_thread; ++i)
    {
        if (min >= d_costs[offset + i])
        {
            min = d_costs[offset + i];
            idx = offset + i;
        }
    }
    sh_costs[threadIdx.x] = min;
    sh_idx[threadIdx.x] = idx;
    if (threadIdx.x == 0)
    {
        offset = work_p_thread * num_threads;
        for (int i = 0; i < rem_work; ++i)
        {
            if (min >= d_costs[offset + i])
            {
                min = d_costs[offset + i];
                idx = offset + i;
            }
        }
        sh_costs[threadIdx.x] = min;
        sh_idx[threadIdx.x] = idx;
    }

    //int i_aux;
    min = sh_costs[0];
    for (int i = 0; i < num_threads; ++i)
    {
        if (min >= sh_costs[i])
        {
            min = sh_costs[i];
            idx = sh_idx[i];
            //i_aux = i;
        }
    }

    work_p_thread = nodes_size / num_threads;
    rem_work = nodes_size % num_threads;
    offset = work_p_thread * threadIdx.x;
    int offset_routes = (idx * nodes_size) + offset;

    printf("min=%f\n", min);
    printf("d_cost_gbr=%f\n", *d_cost_global_best_route);
    if (*d_cost_global_best_route > min)
    {
        printf("updating\n");
        *d_cost_global_best_route = min;
        for (int i = 0; i < work_p_thread; ++i)
        {
            d_global_best_route[offset + i] = d_routes[offset_routes + i];
            printf("gbr=%d\n", d_global_best_route[offset + i]);
        }
        if (threadIdx.x == 0)
        {
            offset = work_p_thread * num_threads;
            offset_routes = (idx * nodes_size) + offset;
            for (int i = 0; i < rem_work; ++i)
            {
                d_global_best_route[offset + i] = d_routes[offset_routes + i];
                printf("gbr=%d\n", d_global_best_route[offset + i]);
            }
        }
    }

    offset = work_p_thread * threadIdx.x * nodes_size;
    for (int i = 0; i < work_p_thread; ++i)
    {
        for (int j = 0; j < nodes_size; ++j)
        {
            d_pheromones[offset + j] = (1 - alpha) * d_pheromones[offset + j]; 
        }
        offset += nodes_size;
    }
    if (threadIdx.x == 0)
    {
        offset = work_p_thread * num_threads * nodes_size;
        for (int i = 0; i < rem_work; ++i)
        {
            for (int j = 0; j < nodes_size; ++i)
            {
                d_pheromones[offset + j] = (1 - alpha) * d_pheromones[offset + j];
            }
            offset += nodes_size;
        }
    }

    offset = work_p_thread * threadIdx.x;
    int src, tgt;
    for (int i = 0; i < work_p_thread; ++i)
    {
        src = d_global_best_route[offset + i];
        if (offset + i + 1  != nodes_size)
            tgt = d_global_best_route[offset + i + 1];
        else
            tgt = d_global_best_route[0];
        d_pheromones[(src * nodes_size) + tgt] += alpha / (*d_cost_global_best_route);
    }
    if (threadIdx.x == 0)
    {
        offset = work_p_thread * num_threads;
        for (int i = 0; i < rem_work; ++i)
        {
            src = d_global_best_route[offset + i];
            if (offset + i + 1  != nodes_size)
                tgt = d_global_best_route[offset + i + 1];
            else
                tgt = d_global_best_route[0];
            d_pheromones[(src * nodes_size) + tgt] += alpha / (*d_cost_global_best_route);
        }
    }
}

void ant_colony_system(std::vector<Coordfloat> &nodes, int num_iter, int num_ants, float beta, float q0,
                       float alpha, float rho, int warp_size, std::vector<int> &global_best_route,
                       float &cost_global_best_route)
{
////////////////////////////////////////INICIALIZACION//////////////////////////////////////////////////

    std::vector<std::vector<float> > pheromones(nodes.size(), std::vector<float>(nodes.size()));
    float cost_nn = nearest_neighbour_tsp(nodes);
    float pheromone0 = 1.0 / (nodes.size() * cost_nn);

    std::cout << "Costo Heuristica NN = " << cost_nn << "\n";

    for (int i = 0; i < nodes.size(); ++i)
    {
        for (int j = 0; j < nodes.size(); ++j)
        {
            pheromones[i][j] = pheromone0;
        }
    }

    std::vector<std::vector<float> > distances(nodes.size(), std::vector<float>(nodes.size()));
    for (int i = 0; i < nodes.size(); ++i)
    {
        for (int j = 0; j < nodes.size(); ++j)
        {
            distances[i][j] = get_euc_dist(nodes[i], nodes[j]);
        }
    }

    std::vector<std::vector<int> > cand_set(nodes.size(), std::vector<int>(warp_size));
    std::vector<std::vector<int> > remain(nodes.size(), std::vector<int>(nodes.size() - warp_size));
    for (int i = 0; i < nodes.size(); ++i)
    {
        Coordfloat c = nodes[i];
        GreaterNC gnc(c);
        std::priority_queue<NodeCoordfloat, std::vector<NodeCoordfloat>, GreaterNC> nns(gnc);

        for (int j = 0; j < nodes.size(); ++j)
        {
            nns.push(NodeCoordfloat(j, nodes[j]));
        }

        NodeCoordfloat nc;

        nns.pop();
        for (int x = 0; x < warp_size; ++x)
        {
            nc = nns.top();
            cand_set[i][x] = nc.idx;
            nns.pop();
        }
        for (int j = 0; j < (nodes.size() - 1) - warp_size; ++j)
        {
            nc = nns.top();
            remain[i][j] = nc.idx;
            nns.pop();
        }
    }



    for (int i = 0; i < nodes.size(); ++i)
    {
        std::cout << i << ": ";
        for (int j = 0; j < warp_size; ++j)
        {
            std::cout << cand_set[i][j] << " ";
        }
        std::cout << "\n";
    }

/////////////////////////////////////COPY TO GLOBAL MEMORY//////////////////////////////////////////


    float *d_pheromones, *d_distances, *d_costs, *d_cost_global_best_route;
    int *d_cand_set, *d_remain, *d_global_best_route, *d_routes;

    //INPUT
    int size_pheromones = nodes.size() * nodes.size() * sizeof(float);
    cudaMalloc((void **) &d_pheromones, size_pheromones);
    for (int i = 0; i < nodes.size(); ++i)
    {
        cudaMemcpy(d_pheromones + (i * nodes.size()), &pheromones[i][0], nodes.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    int size_distances = nodes.size() * nodes.size() * sizeof(float);
    cudaMalloc((void **) &d_distances, size_distances);
    for (int i = 0; i < nodes.size(); ++i)
    {
        cudaMemcpy(d_distances + (i * nodes.size()), &distances[i][0], nodes.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    int size_cand_set = nodes.size() * warp_size * sizeof(int);
    cudaMalloc((void **) &d_cand_set, size_cand_set);
    for (int i = 0; i < nodes.size(); ++i)
    {
        cudaMemcpy(d_cand_set + (i * warp_size), &cand_set[i][0], warp_size * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    int size_remain = nodes.size() * (nodes.size() - warp_size) * sizeof(int);
    cudaMalloc((void **) &d_remain, size_remain);
    for (int i = 0; i < nodes.size(); ++i)
    {
        cudaMemcpy(d_remain + (i * (nodes.size() - warp_size)), &remain[i][0], (nodes.size() - warp_size) * sizeof(int), cudaMemcpyHostToDevice);
    }


    curandState *devStates;
    cudaMalloc((void **) &devStates, num_ants * sizeof(curandState));

    //OUTPUT
    int size_global_best_route = nodes.size() * sizeof(int);
    cudaMalloc((void **) &d_global_best_route, size_global_best_route);

    int size_routes = num_ants * nodes.size() * sizeof(int);
    cudaMalloc((void **) &d_routes, size_routes);

    int size_costs = num_ants * sizeof(float);
    cudaMalloc((void **) &d_costs, size_costs);

    int size_cost_global_best_route = sizeof(float);
    cudaMalloc((void **) &d_cost_global_best_route, size_cost_global_best_route);
    float cgbr = 99999999999;
    cudaMemcpy(d_cost_global_best_route, &cgbr, sizeof(float), cudaMemcpyHostToDevice);

    //KERNEL 
    setup_kernel<<<1, num_ants>>>(devStates);

    for (int i = 0; i < num_iter; ++i)
    {
        build_ant_solutions_alt_k<<<num_ants, warp_size>>>(d_pheromones, d_distances, d_cand_set,
                                                           d_remain, d_routes,
                                                           d_costs, nodes.size(), warp_size,
                                                           devStates, beta, rho, pheromone0);
        global_pheromone_update_k<<<1, 128>>>(d_pheromones, d_routes, d_costs, alpha, d_global_best_route,
                                              num_ants, nodes.size(), d_cost_global_best_route);
    }

    global_best_route = std::vector<int>(nodes.size());
    cudaMemcpy(&global_best_route[0], d_global_best_route, size_global_best_route, cudaMemcpyDeviceToHost);

    cudaMemcpy(&cost_global_best_route, d_cost_global_best_route, size_cost_global_best_route, cudaMemcpyDeviceToHost);


    cudaFree(d_pheromones);
    cudaFree(d_distances);
    cudaFree(d_cand_set);
    cudaFree(d_global_best_route);
    cudaFree(devStates);
    cudaFree(d_remain);
    cudaFree(d_cost_global_best_route);
}

#endif //ACS_H