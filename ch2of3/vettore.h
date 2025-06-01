/* Declaring the class Vettore */
#ifndef VETTORE_H
#define VETTORE_H

#include<iostream>

class Vettore
{
private:
	double* m_data;		// Elements of the vector
	int m_dimensione;	// Its dimension

public:
 	// Constructors
	Vettore(int dimensione);			// Costruttore //
	~Vettore();					// Distruttore //
	Vettore(const Vettore& altro);			// Costr. copia //

	// Operators //
	Vettore& operator=(const Vettore& altro);	// Operator = //
	double operator[](int indice) const;
	double& operator[](int indice);
	Vettore operator+(const Vettore& altro) const;

	// Methods
	int get_element(int) const;	// const means: object is unmodified
	int get_dimension() const;
	void stampa() const;
	void set_element(int, double);
};

#endif
