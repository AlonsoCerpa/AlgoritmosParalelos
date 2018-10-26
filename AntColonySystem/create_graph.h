#ifndef CREATE_GRAPH_H
#define CREATE_GRAPH_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/tokenizer.hpp>


typedef boost::tokenizer<boost::char_separator<char> > tokenizer;


struct CoordDouble
{
    double x;
    double y;
};

void read_tsp_file(std::vector<CoordDouble> &nodes, std::string &name_file)
{
    std::ifstream data_tsp(name_file);
    if (data_tsp.is_open())
    {
        std::string line;
        boost::char_separator<char> sep(" ");
        bool found = false;

        tokenizer::iterator tok_iter;

        while (found == false && getline(data_tsp, line))
        {
            std::cout << line << "\n";
            tokenizer tokens = tokenizer(line, sep);
            tok_iter = tokens.begin();
            if (*tok_iter == "DIMENSION")
                found = true;
        }
        ++tok_iter;
        ++tok_iter;
        int dim = std::stoi(*tok_iter);
        nodes = std::vector<CoordDouble>(dim);
        while (getline(data_tsp, line) && line != "NODE_COORD_SECTION")
        {
            std::cout << line << "\n";
        }
        
        CoordDouble c;
        int i = 0;
        while(getline(data_tsp, line) && line != "EOF")
        {
            tokenizer tokens = tokenizer(line, sep);
            tok_iter = tokens.begin();
            ++tok_iter;
            c.x = std::stod(*tok_iter);
            ++tok_iter;
            c.y = std::stod(*tok_iter);
            nodes[i] = c; 
            ++i;
        }
    }
    else
    {
        std::cout << "No se pudo abrir el archivo: " << name_file << "\n";
    }

    data_tsp.close();
}

#endif //CREATE_GRAPH_H