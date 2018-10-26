#include <iostream>

#include "create_graph.h"
#include "acs.h"


int main()
{
    std::vector<CoordDouble> nodes;
    std::string name_file = "d198.tsp";
    read_tsp_file(nodes, name_file);   

    /*
    for (int i = 0; i < nodes.size(); ++i)
    {
        std::cout << "Node " << i << ": x = " << nodes[i].x << ", y = " << nodes[i].y << "\n";
    }*/

    int num_iter = 1000;
    int num_ants = 30;
    double beta = 2.0;
    double q0 = 0.9;
    double alpha = 0.1;
    double rho = 0.1;
    std::vector<int> global_best_route;
    double cost_global_best_route;
    
    //el numero de ants tiene que ser menor o igual al numero de nodos/ciudades
    ant_colony_system(nodes, num_iter, num_ants, beta, q0, alpha, rho, global_best_route, cost_global_best_route);
    
    for (int i = 0; i < global_best_route.size(); ++i)
    {
        std::cout << global_best_route[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Costo = " << cost_global_best_route << "\n";

    return 0;
}