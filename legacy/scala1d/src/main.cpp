#include <iostream>
#include "RepCode.h"
#include "CA.h"
#include "Random.h"
#include <array>
#include <fstream>
#include <string>
#include <format>
#include <numeric>
#include <algorithm>

#ifndef L
#define L 7 // Default value
#endif

unsigned long benchmark_CodeCapacity(int N, double p)
{
	std::ofstream file;
	std::string path = std::format("./data/code_capacity/L={0:d}_p={1:.3f}.csv", L,p);
	std::cout << path << '\n';
	file.open(path, std::ios_base::app);

	unsigned long count = 0;
	for (int n=0; n<N; n++)
	{
		RepCode<L> code;
		CA<L> ca;
		code.setNoise(p);

		for (int i=0; i<L; i++)
		// for (int i=0; i<3*L; i++) // for smallest reset time
		{
			ca.step(code);
		}

		// check if syndrome resolved
		std::array<bool,L> syndrome = code.getSyndrome();
		double nDefects = std::accumulate(syndrome.begin(), syndrome.end(), 0.0);
		if (nDefects != 0) {
			std::cout << "FAILED!" << '\n';
		}

		bool logErr = code.hasLogErr();
		file << logErr << std::endl;
		count += code.hasLogErr();

	}
	return count;
}

void benchmark_Phenomenological_var_reset(int N, double p) {

	// int t_r_min = 1;
	// int t_r_max = int( (L+1)/2 ); 
	int t_r_min = 3;
	int t_r_max = 3;

	for (int t_r=t_r_min; t_r<t_r_max + 1; t_r++) 
	{
		std::ofstream file;
		std::string path = std::format("./data/pheno_var_reset/L={0:d}_p={1:.3f}_t={2:d}.csv", L,p,t_r);
		std::cout << path << '\n';
		file.open(path, std::ios_base::app);


		unsigned long tot_count = 0;
		for (int n=0; n<N; n++) {
			RepCode<L> code;
			CA<L> ca;
			unsigned long count = 0;
			int n_errs = 0;
			while (!code.hasLogErr()) {
				code.setNoise(p);
				n_errs += code.getNErrs();
				ca.step(code);
				count += 1;

				if (count % t_r == 0)
					ca.reset();
			}
			// if (n_errs < code.getNQubits()/2+1)
			// 	std::cout << "Sample " << n << ", #errs: " << n_errs << '\n';
			file << count << std::endl;
			tot_count += count;
		}
		double frac = double(tot_count) / N;
		std::cout << std::format("t_r = {0:d} -- {1:.4f}", t_r, frac) << '\n';
	}
	// return tot_count;
}

void benchmark_Phenomenological_q_var_reset(int N, double p) {

	int t_r_min = 1;
	int t_r_max = int( (L+3)/2 );
	// int t_r_max = L;

	for (int t_r=t_r_min; t_r<t_r_max + 1; t_r++) 
	{
		std::ofstream file;
		std::string path = std::format("./data/pheno_q_var_reset/L={0:d}_p={1:.3f}_t={2:d}.csv", L,p,t_r);
		std::cout << path << '\n';
		file.open(path, std::ios_base::app);


		unsigned long tot_count = 0;
		for (int n=0; n<N; n++) {
			RepCode<L> code;
			CA<L> ca;
			unsigned long count = 0;
			int n_errs = 0;
			while (!code.hasLogErr()) {
				// code.setNoise(p);
				n_errs += code.getNErrs();
				ca.step(code, p); // meas noise (only)
				count += 1;

				if (count % t_r == 0)
					ca.reset();
			}
			// if (n_errs < code.getNQubits()/2+1)
			// 	std::cout << "Sample " << n << ", #errs: " << n_errs << '\n';
			file << count << std::endl;
			tot_count += count;
		}
		double frac = double(tot_count) / N;
		std::cout << std::format("t_r = {0:d} -- {1:.4f}", t_r, frac) << '\n';
	}
	// return tot_count;
}

void benchmark_Phenomenological_psig_var_reset(int N, double p, double psig) {

	// int t_r_min = 1;
	// int t_r_max = int( (L+1) / 2);
	int t_r_min = 16;
	int t_r_max = L;
	// int t_r_max = L;

	for (int t_r=t_r_min; t_r<t_r_max + 1; t_r++) 
	{
		std::ofstream file;
		// std::string path = std::format("./data/pheno_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
		std::string path = std::format("./data/pheno_psig_var_reset/L={0:d}_p={1:.3f}_psig={2:.3f}_t={3:d}.csv", L,p,psig,t_r);
		file.open(path, std::ios_base::app);
		std::cout << path << '\n';


		unsigned long tot_count = 0;
		for (int n=0; n<N; n++) {
			RepCode<L> code;
			CA<L> ca;
			unsigned long count = 0;
			while (!code.hasLogErr()) {
				code.setNoise(p);
				ca.step(code, 0.0, psig);
				count += 1;

				if (count % t_r == 0)
					ca.reset();
			}
			file << count << std::endl;
			tot_count += count;
		}

		double frac = double(tot_count) / N;
		std::cout << std::format("t_r = {0:d} -- {1:.4f}", t_r, frac) << '\n';
    }
}

void benchmark_Phenomenological_q_psig_var_reset(int N, double p, double psig) {
	// p, q, psig with p=q=psig

	int t_r_min = 1;
	int t_r_max = int( (L+1) / 2);
	// int t_r_min = 25;
	// int t_r_max = 25;
	// int t_r_max = L;

	for (int t_r=t_r_min; t_r<t_r_max + 1; t_r++) 
	{
		std::ofstream file;
		// std::string path = std::format("./data/pheno_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
		std::string path = std::format("./data/pheno_q_psig_var_reset/L={0:d}_p={1:.3f}_psig={2:.3f}_t={3:d}.csv", L,p,psig,t_r);
		file.open(path, std::ios_base::app);
		std::cout << path << '\n';


		unsigned long tot_count = 0;
		for (int n=0; n<N; n++) {
			RepCode<L> code;
			CA<L> ca;
			unsigned long count = 0;
			while (!code.hasLogErr()) {
				code.setNoise(p);
				ca.step(code, p, psig);
				count += 1;

				if (count % t_r == 0)
					ca.reset();
			}
			file << count << std::endl;
			tot_count += count;
		}

		double frac = double(tot_count) / N;
		std::cout << std::format("t_r = {0:d} -- {1:.4f}", t_r, frac) << '\n';
    }
}

unsigned long benchmark_Phenomenological(int N, double p) {
	std::ofstream file;
	// std::string path = std::format("./data/pheno/L={0:d}_p={1:.3f}.csv", L,p);
	std::string path = std::format("./data/pheno2/L={0:d}_p={1:.3f}.csv", L,p);
	// std::string path = std::format("./data/pheno2_early_reset/L={0:d}_p={1:.3f}.csv", L,p);
	// std::string path = std::format("./data/pheno_very_early_reset/L={0:d}_p={1:.3f}.csv", L,p);
	// std::string path = std::format("./data/pheno_early_reset/L={0:d}_p={1:.3f}.csv", L,p);
	// std::string path = std::format("./data/pheno_medium_reset/L={0:d}_p={1:.3f}.csv", L,p);
	std::cout << path << '\n';
	file.open(path, std::ios_base::app);


	unsigned long tot_count = 0;
	for (int n=0; n<N; n++) {
		RepCode<L> code;
		CA<L> ca;
		unsigned long count = 0;
		int n_errs = 0;
		while (!code.hasLogErr()) {
			code.setNoise(p);
			n_errs += code.getNErrs();
			ca.step(code);
			count += 1;
		}
		if (n_errs < code.getNQubits()/2+1)
			std::cout << "Sample " << n << ", #errs: " << n_errs << '\n';
		file << count << std::endl;
		tot_count += count;
	}
	return tot_count;
}

unsigned long benchmark_Phenomenological_q(int N, double q) {
	std::ofstream file;
	// std::string path = std::format("./data/pheno_q/L={0:d}_q={1:.3f}.csv", L,q);
	std::string path = std::format("./data/pheno_q2/L={0:d}_q={1:.3f}.csv", L,q);
	std::cout << path << '\n';
	file.open(path, std::ios_base::app);


	unsigned long tot_count = 0;
	for (int n=0; n<N; n++) {
		RepCode<L> code;
		CA<L> ca;
		unsigned long count = 0;
		while (!code.hasLogErr()) {
			// code.setNoise(p);
			ca.step(code, q);
			count += 1;
		}
		file << count << std::endl;
		tot_count += count;
	}
	return tot_count;
}

unsigned long benchmark_Phenomenological_pq(int N, double p, double q) {
	std::ofstream file;
	std::string path = std::format("./data/pheno_pq_early_reset/L={0:d}_p={1:.3f}_q={2:.3f}.csv", L,p,q);
	std::cout << path << '\n';
	file.open(path, std::ios_base::app);


	unsigned long tot_count = 0;
	for (int n=0; n<N; n++) {
		RepCode<L> code;
		CA<L> ca;
		unsigned long count = 0;
		while (!code.hasLogErr()) {
			code.setNoise(p);
			ca.step(code, q);
			count += 1;
		}
		file << count << std::endl;
		tot_count += count;
	}
	return tot_count;
}

unsigned long benchmark_Phenomenological_psig(int N, double p, double psig) {
    std::ofstream file;
    // std::string path = std::format("./data/pheno_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
    std::string path = std::format("./data/pheno2_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
    file.open(path, std::ios_base::app);
	std::cout << path << '\n';


    unsigned long tot_count = 0;
    for (int n=0; n<N; n++) {
		RepCode<L> code;
		CA<L> ca;
        unsigned long count = 0;
        while (!code.hasLogErr()) {
			code.setNoise(p);
			ca.step(code, 0.0, psig);
            count += 1;
        }
        file << count << std::endl;
        tot_count += count;
    }
    return tot_count;
}

unsigned long benchmark_Phenomenological_pq_psig(int N, double p, double psig) {
    std::ofstream file;
    // std::string path = std::format("./data/pheno_pq_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
    std::string path = std::format("./data/pheno2_pq_psig_early_reset/L={0:d}_p={1:.3f}_psig={2:.3f}.csv", L,p,psig);
    file.open(path, std::ios_base::app);
	std::cout << path << '\n';


    unsigned long tot_count = 0;
    for (int n=0; n<N; n++) {
		RepCode<L> code;
		CA<L> ca;
        unsigned long count = 0;
        while (!code.hasLogErr()) {
			code.setNoise(p); // data noise
			ca.step(code, p, psig); // meas. + sig noise
            count += 1;
        }
        file << count << std::endl;
        tot_count += count;
    }
    return tot_count;
}

void testCode()
{
	RepCode<L> code;
	std::cout << code.getNQubits() << '\n';

	double p = 0.5;
	code.setNoise(p);
	std::cout << code << '\n';

	std::array<bool,L> syndrome = code.getSyndrome();
	for (auto s : syndrome)
		std::cout << s << ' ';
	std::cout << '\n';

	std::array<bool,L> syndrome2 = code.getSyndrome(p);
	for (auto s : syndrome2)
		std::cout << s << ' ';
	std::cout << '\n';
}

void testCA() 
{
	RepCode<L> code;
	CA<L> ca;

	double p = 0.5;
	code.setNoise(p);
	code.print_qubits();

	for (int i=0; i<L; i++) 
	{
		std::cout << code;
		ca.step(code); // no meas. noise
		std::cout << ca << '\n';
	}
	// ca.step(code, p); // meas noise
}

int main()
{
	// testCode();
	// testCA();

	int N = 1000;
	// int N = 900;
	// double p = 0.5;
	// std::array<double, 18> ps { 9/10.,8/10.,7/10.,6/10.,5/10.,4/10.,3/10.,2/10.,1/10., 1/20., 1/30.,1/40.,1/50., 1/60., 1/70., 1/80., 1/90., 1/100.};
	// std::array<double, 9> ps { 9/10.,8/10.,7/10.,6/10.,5/10.,4/10.,3/10.,2/10.,1/10.};
	// std::array<double, 9> ps { 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1};
	// std::array<double, 9> ps { 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2};
	// std::array<double, 8> ps { 8e-3, 7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3};
	// std::array<double, 4> ps { 9e-3, 8e-3, 7e-3,6e-3};
	// std::array<double, 5> ps { 5e-3, 4e-3, 3e-3, 2e-3, 1e-3};
	// std::array<double, 2> ps {8e-3, 7e-3};
	// std::array<double, 4> ps {1e-2, 9e-3, 8e-3, 7e-3};
	// std::array<double, 9> ps {9e-3,8e-3,7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3};
	// std::array<double, 9> ps {9e-2,8e-2,7e-2,6e-2,5e-2,4e-2,3e-2,2e-2,1e-2};
	// std::array<double, 5> ps {5e-2,4e-2,3e-2,2e-2,1e-2};
	
	// signal noise
	std::array<double, 27> ps { 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3};
	// std::array<double, 18> ps { 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2};

	std::cout << "L=" << L << '\n';
	for (double p : ps)
	{
		std::cout << "p=" << p << '\n';
		// unsigned long n_errs = benchmark_CodeCapacity(N,p);
		// unsigned long n_errs = benchmark_Phenomenological(N,p);
		// unsigned long n_errs = benchmark_Phenomenological_q(N,p);
		benchmark_Phenomenological_var_reset(N,p);
		// benchmark_Phenomenological_q_var_reset(N,p);
		// benchmark_Phenomenological_psig_var_reset(N, 4e-2, p);
		// benchmark_Phenomenological_psig_var_reset(N, p, p); // p_sig = p
		// benchmark_Phenomenological_q_psig_var_reset(N, p, p); // p_sig = p
		// unsigned long n_errs = benchmark_Phenomenological_pq(N,p,p);
		// unsigned long n_errs = benchmark_Phenomenological_psig(N,1/50.,p);
		// unsigned long n_errs = benchmark_Phenomenological_pq_psig(N,p, 1e-2);
		// std::cout << "mu=" << n_errs / double(N) << "\n\n";
	}
}

