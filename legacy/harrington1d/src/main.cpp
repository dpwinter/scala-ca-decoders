// HCAD-1D 

#include "RepCode.h"
#include "CA.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <format>
#include <algorithm>

void applyCorrections(RepCode& code, std::vector<Location>& cors, int L) {
	for (int i=0; i<L; i++) 
		code.flip(i, cors[i]);
}

// double benchmark(int L, int U, int N, double fN, double fC, int pi) {
// 	std::ofstream file;
// 	/* std::string path = "./data/L=" + std::to_string(L) + "_pi=" + std::to_string(pi) + ".csv"; */
// 	std::string path = "./data_U=5/L=" + std::to_string(L) + "_pi=" + std::to_string(pi) + ".csv";
// 	file.open(path, std::ios_base::app);

// 	RepCode code{L};
// 	CA ca{L,U,fC,fN};

// 	int tot_count = 0;
// 	for (int n=0; n<N; n++) {
// 		code.reset();
// 		ca.reset();
// 		int count = 0;
// 		while (!code.hasLogErr()) {
// 			double p = 1./pi;
// 			code.noise(p);
// 			std::vector<bool> syndrome = code.getSyndrome();
// 			std::vector<Location> cors = ca.step(syndrome);
// 			applyCorrections(code, cors, L);
// 			count += 1;
// 		}
// 		file << count << std::endl;
// 		tot_count += count;
// 	}
// 	return tot_count;
// }

unsigned long benchmark_cc(int L, int U, int N, double fN, double fC, double p)
{
	std::ofstream file;
	std::string path = std::format("./data/cc/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/cc2/L={0:d}_p={1:.4f}.csv", L,p);
	file.open(path, std::ios_base::app);
	std::cout << path << '\n';

	unsigned long counts = 0;
	for (int n=0; n<N; n++) 
	{
		CA ca{L,U,fC,fN};
		RepCode code{L};
		code.noise(p);
		
		int log_err = 0;
		std::vector<bool> syndrome = code.getSyndrome();
		while(std::accumulate(syndrome.begin(), syndrome.end(), 0.0) != 0.0)
		{
			std::vector<Location> cors = ca.step(syndrome);
			applyCorrections(code, cors, L);
			syndrome = code.getSyndrome();
		}

		if(code.hasLogErr())
			log_err = 1;
		
		file << log_err << std::endl;
		counts += log_err;
	}

	return counts;
}

unsigned long benchmark_pheno(int L, int U, int N, double fN, double fC, double p)
{
	std::ofstream file;
	std::string path = std::format("./data/pheno_U=16/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno2/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno3/L={0:d}_p={1:.4f}.csv", L,p);
	file.open(path, std::ios_base::app);
	std::cout << path << '\n';

	unsigned long tot_count = 0;
	for (int n=0; n<N; n++) 
	{
		CA ca{L,U,fC,fN};
		RepCode code{L};
		
		std::vector<bool> syndrome;
		unsigned long count = 0;
		while(!code.hasLogErr())
		{
			code.noise(p);
			syndrome = code.getSyndrome();
			std::vector<Location> cors = ca.step(syndrome);
			applyCorrections(code, cors, L);
			count += 1;
		}

		file << count << std::endl;
		tot_count += count;
	}

	return tot_count;
}

unsigned long benchmark_pheno_q(int L, int U, int N, double fN, double fC, double p)
{
	std::ofstream file;
	std::string path = std::format("./data/pheno_U=16_q/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno2/L={0:d}_p={1:.4f}.csv", L,p);
	// std::string path = std::format("./data/pheno3/L={0:d}_p={1:.4f}.csv", L,p);
	file.open(path, std::ios_base::app);
	std::cout << path << '\n';

	unsigned long tot_count = 0;
	for (int n=0; n<N; n++) 
	{
		CA ca{L,U,fC,fN};
		RepCode code{L};
		
		std::vector<bool> syndrome;
		unsigned long count = 0;
		while(!code.hasLogErr())
		{
			// code.noise(p);
			syndrome = code.getSyndrome(p);
			std::vector<Location> cors = ca.step(syndrome);
			applyCorrections(code, cors, L);
			count += 1;
		}

		file << count << std::endl;
		tot_count += count;
	}

	return tot_count;
}

void pheno_min_w(int L, int U, int N, double fN, double fC, double p)
{
	int Q = 3;
	float df = std::log2(L) / std::log2(Q);
	float max_errs = std::pow(2,df);
	std::cout << max_errs << '\n';

	for (int n=0; n<N; n++) 
	{
		CA ca{L,U,fC,fN};
		RepCode code{L};
		
		std::vector<bool> syndrome;
		std::vector<std::vector<bool>> store;
		int n_errs = 0;
		int n_steps = 0;
		while(!code.hasLogErr())
		{
			code.noise(p);
			n_errs += std::accumulate(code.qubits.begin(), code.qubits.end(), 0.0);
			store.push_back(code.qubits);
			syndrome = code.getSyndrome();
			std::vector<Location> cors = ca.step(syndrome);
			applyCorrections(code, cors, L);
			n_steps += 1;
		}

		if (n_errs < max_errs) {
			std::cout << "jo: " << n_errs << ", " << n_steps << '\n';
			for (auto &conf : store) {
				for (bool qb : conf) {
					std::cout << qb << " ";
				}
				std::cout << '\n';
			}
		}


	}
}

int main() {

	int N = 100;
	// int U = 10;
	int U = 16;
	double fC = 9/10.;
	// double fC = 7/10.;
	double fN = 4/10.;

	std::vector<int> Ls = {3,9,27,81};
	// std::vector<int> Ls = {81};
	// std::vector<int> Ls = {27};
	// std::vector<int> Ls = {9};
	// std::vector<int> Ls = {3};
	// std::vector<double> ps = {9e-1,8e-1,7e-1,6e-1,5e-1,4e-1,3e-1,2e-1,1e-1};
	// std::vector<double> ps = {2e-1, 1e-1};
	// std::vector<double> ps = {9e-2,8e-2,7e-2,6e-2,5e-2,4e-2,3e-2,2e-2,1e-2};
	std::vector<double> ps = {9e-3,8e-3,7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3};
	// std::vector<double> ps = {9e-3,8e-3,7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3};
	// std::vector<double> ps = {1e-1, 9e-2, 8e-2, 7e-2, 6e-2,5e-2,4e-2};
	// std::vector<double> ps = {9e-3,8e-3,7e-3,6e-3,5e-3};
	// std::vector<double> ps = {5e-2, 6e-2, 4e-2};

	for (int L : Ls) {
		std::cout << "--- Lattice size " << L << " ---\n";
		for (double p : ps) {
			std::cout << "    --- " << std::format("p={0:.4f}", p) << "---" << '\n';
			// unsigned long tot_count = benchmark_cc(L,U,N,fN,fC,p);
			// unsigned long tot_count = benchmark_pheno(L,U,N,fN,fC,p);
			unsigned long tot_count = benchmark_pheno_q(L,U,N,fN,fC,p);
			std::cout << "mu=" << tot_count / double(N) << '\n';
		}
	}

	// double p = 0.05;
	// int L = 9;

	// pheno_min_w(L, U, N, fN, fC, p);

}
