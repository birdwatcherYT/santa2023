#include <iostream>
#include <random>

using namespace std;

mt19937 rand_engine;

inline double get_rand(){
	uniform_real_distribution<double> rand01(0.0, 1.0);
	return rand01(rand_engine);
}

int main(int argc, char *argv[]) {
    if (argc==4){
        int seed=atoi(argv[1]);
        double x=atof(argv[2]);
        double y=atof(argv[3]);
        rand_engine.seed(seed);
        double score = -(x-get_rand())*(x-get_rand()) -(y-get_rand())*(y-get_rand());
        cout<<score<<endl;
    }
    return 0;
}
