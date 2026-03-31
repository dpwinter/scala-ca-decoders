#include "RepCode.h"

RepCode::RepCode(int L) : L{L} {
	qubits = std::vector<bool>(L,0);
}

void RepCode::reset() {
	for (int i=0; i<L; i++) {
		qubits[i] = 0;
	}
}

bool RepCode::getStab(int i) {
	return qubits[i] ^ qubits[(i+1)%L];
}

const bool RepCode::getQubit(int i) {
	return qubits[i];
}

void RepCode::setQubit(int i, bool val) {
	qubits[i] = val;
}

std::vector<bool> RepCode::getSyndrome() {
	std::vector<bool> syndrome = std::vector<bool>(L,0);
	for (int i=0; i<L; i++) {
		syndrome[i] = getStab(i);
	}
	return syndrome;
}

std::vector<bool> RepCode::getSyndrome(double p) {
	std::vector<bool> syndrome = std::vector<bool>(L,0);
	for (int i=0; i<L; i++) {
		syndrome[i] = getStab(i);
		if (Random::get(0.0,1.0) < p)
			syndrome[i] = syndrome[i] ^ 1;
	}
	return syndrome;
}

bool RepCode::hasLogErr() {
	int sum = 0;
	for (int i=0; i<L; i++) {
		sum += qubits[i];
	}
	return (sum > L/2);
}

void RepCode::flip(int i, Location loc) {
	if (loc == Location::R) 
		qubits[(i+1)%L] = qubits[(i+1)%L] ^ 1;
	else if (loc == Location::L) 
		qubits[i] = qubits[i] ^ 1;
}

void RepCode::noise(double p) {
	for (int i=0; i<L; i++) {
		if (Random::get(0.0,1.0) < p)
			qubits[i] = qubits[i] ^ 1;
	}
}

std::ostream& operator<<(std::ostream &out, RepCode const& code) {
	for (int i=0; i<code.L; i++) {
		out << code.qubits[i] << " ";
	}
	out << '\n';
	return out;
}
