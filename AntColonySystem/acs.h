#ifndef ACS_H
#define ACS_H

#include <vector>
#include <list>
#include <cmath>
#include <random>
#include <algorithm>

#include "create_graph.h"

double get_euc_dist(CoordDouble &a, CoordDouble &b)
{
    return sqrt(pow(b.x - a.x, 2.0) + pow(b.y - a.y, 2.0));
}

double get_nn_distance(std::vector<CoordDouble> &nodes, int &node_idx, std::list<int> &unvisited_nodes)
{
    double min_dist = std::numeric_limits<double>::max();
    double curr_dist;
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

double nearest_neighbour_tsp(std::vector<CoordDouble> &nodes)
{
    int node_idx = 0;
    double cost = 0;
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

//parameters
//


int select_next_node(int node_source, std::list<int> &no_route_nodes_k, double &dist, double q0, double beta,
                     std::vector<std::vector<double> > &pheromones, std::vector<CoordDouble> &nodes)
{
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double q = uni(rng);
    double dist_aux;
    int next_node;
    auto it_erase = no_route_nodes_k.begin();

    if (q < q0)                                //seleccion voraz
    {
        double max = std::numeric_limits<double>::lowest();
        double argmax;

        for (auto it = no_route_nodes_k.begin(); it != no_route_nodes_k.end(); ++it)
        {
            dist_aux = get_euc_dist(nodes[node_source], nodes[*it]);
            argmax = pheromones[node_source][*it] / pow(dist_aux, beta);
            if (argmax > max)
            {
                it_erase = it;
                next_node = *it;
                dist = dist_aux;
                max = argmax;
            }
        }

        no_route_nodes_k.erase(it_erase);
        return next_node;
    }
    else                                       //seleccion proporcional pseudo-aleatoria
    {
        std::vector<double> probs(no_route_nodes_k.size());
        double dist_aux, division;
        double sum = 0.0;
        int i = 0;

        for (auto it = no_route_nodes_k.begin(); it != no_route_nodes_k.end(); ++it)
        {
            dist_aux = get_euc_dist(nodes[node_source], nodes[*it]);
            division = pheromones[node_source][*it] / pow(dist_aux , beta);
            probs[i] = division;
            sum += division;
            ++i;
        }

        for (int i = 0; i < probs.size(); ++i)
        {
            probs[i] = probs[i] / sum;
        }

        double rand_num = uni(rng);
        double count = 0.0;

        for (int i = 0; i < probs.size(); ++i)
        {
            if (rand_num >= count && rand_num < count + probs[i])
            {
                auto next_it = std::next(no_route_nodes_k.begin(), i);
                next_node = *next_it;
                dist = get_euc_dist(nodes[node_source], nodes[next_node]);
                no_route_nodes_k.erase(next_it);
                return next_node;
            }
            count += probs[i];
        } //hay casos donde no retorna nada porque no entra al if
    }
}

void local_pheromone_update(std::vector<std::vector<double> > &pheromones, int source, int target, double pheromone0, double rho)
{
    pheromones[source][target] = ((1 - rho) * pheromones[source][target]) + (rho * pheromone0);
}

void global_pheromone_update(std::vector<std::vector<double> > &pheromones, std::vector<int> &global_best_route, double cost_global_best_route,
                             double alpha)
{
    for (int i = 0; i < pheromones.size(); ++i)
    {
        for (int j = 0; j < pheromones.size(); ++j)
        {
            pheromones[i][j] = (1 - alpha) * pheromones[i][j];
        }
    }

    int src, tgt;
    for (int i = 0; i < global_best_route.size() - 1; ++i)
    {
        src = global_best_route[i];
        tgt = global_best_route[i+1];
        pheromones[src][tgt] = pheromones[src][tgt] + (alpha / cost_global_best_route);
    }
}

void ant_colony_system(std::vector<CoordDouble> &nodes, int num_iter, int num_ants, double beta, double q0,
                       double alpha, double rho, std::vector<int> &global_best_route, double &cost_global_best_route)
{
    std::vector<std::vector<double> > pheromones(nodes.size(), std::vector<double>(nodes.size()));
    double cost_nn = nearest_neighbour_tsp(nodes);
    double pheromone0 = 1.0 / (nodes.size() * cost_nn);

    std::cout << "Costo Heuristica NN = " << cost_nn << "\n";

    for (int i = 0; i < nodes.size(); ++i)
    {
        for (int j = 0; j < nodes.size(); ++j)
        {
            pheromones[i][j] = pheromone0;
        }
    }

    std::vector<int> nodes_first_visited;
    
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)

    std::vector<std::vector<int> > routes(num_ants, std::vector<int>(nodes.size()));
    std::vector<double> cost_routes;
    std::vector<std::list<int> > no_route_nodes(num_ants);
    cost_global_best_route = std::numeric_limits<double>::max();

    for (int i = 0; i < num_iter; ++i)
    {
        //Inicializar las rutas: la primera ciudad de cada hormiga es aleatoria

        cost_routes = std::vector<double>(num_ants, 0.0);
        nodes_first_visited = std::vector<int>(nodes.size());
        for (int k = 0; k < num_ants; ++k)
        {
            for (int j = 0; j < nodes.size(); ++j)
            {
                no_route_nodes[k].push_back(j);
            }
        }

        for (int j = 0; j < nodes.size(); ++j)
        {
            nodes_first_visited[j] = j;
        }

        for (int k = 0; k < num_ants; ++k)
        {
            std::uniform_int_distribution<int> uni(0, nodes_first_visited.size() - 1);
            int random_int = uni(rng);
            routes[k][0] = nodes_first_visited[random_int];
            nodes_first_visited.erase(nodes_first_visited.begin() + random_int);

            //borrar el nodo inicial de los nodos que no estan el ruta
            auto it = std::find(no_route_nodes[k].begin(), no_route_nodes[k].end(), routes[k][0]);
            no_route_nodes[k].erase(it);
        }

        //Construccion de las rutas de cada hormiga
        int next_node;
        double dist;
        for (int j = 1; j < nodes.size(); ++j)
        {
            for (int k = 0; k < num_ants; ++k)
            {
                next_node = select_next_node(routes[k][j-1], no_route_nodes[k], dist, q0, beta, pheromones, nodes);  //cambiar parametros
                routes[k][j] = next_node;
                cost_routes[k] += dist;
                local_pheromone_update(pheromones, routes[k][j-1], routes[k][j], pheromone0, rho);
            }
        }

        //Aristas que cierran la ruta
        int node1, node2;
        for (int k = 0; k < num_ants; ++k)
        {
            node1 = routes[k][nodes.size() - 1];
            node2 = routes[k][0];
            cost_routes[k] += get_euc_dist(nodes[node1], nodes[node2]);
            local_pheromone_update(pheromones, node1, node2, pheromone0, rho);
        }

        //Seleccionando la mejor ruta local
        int ant_local_best_route;
        double cost_local_best_route = std::numeric_limits<double>::max();
        for (int k = 0; k < num_ants; ++k)
        {
            if (cost_routes[k] < cost_local_best_route)
            {
                cost_local_best_route = cost_routes[k];
                ant_local_best_route = k;
            }
        }

        //Se actualiza la mejor ruta global si es peor que la mejor ruta local
        if (cost_local_best_route < cost_global_best_route)
        {
            global_best_route = routes[ant_local_best_route];
            cost_global_best_route = cost_local_best_route;
        }

        global_pheromone_update(pheromones, global_best_route, cost_global_best_route, alpha);
    }
}

#endif //ACS_H