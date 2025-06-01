#include <iostream>
#include "vettore.h"

int main() {
	try {
		std::cout << "Performing various tests." << std::endl;
		Vettore v1(3);
		v1.stampa();
		v1.set_element(0, 1.);
		v1.set_element(1, 2.);
		v1.set_element(2, 3.);
		v1.stampa();
		std::cout << "Dimension" << v1.get_dimension() << std::endl;
		std::cout << "Elm 0: " << v1.get_element(0) << std::endl;
		Vettore v2(v1);
		v2.stampa();
		Vettore v3(3);
		v3 = v1;
		v3[0] = 999.;
		v3.stampa();
		Vettore v5 = v1 + v3;
		v5.stampa();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}

