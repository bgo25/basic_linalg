#include "vettore.h"
#include <stdexcept>
#include <iostream>
#include <omp.h>

Vettore::~Vettore()
{
	delete[] m_data;
	m_data = nullptr;
	m_dimensione = 0;
}

Vettore::Vettore(int dimensione) : m_data(nullptr), m_dimensione(0)
{
	// If dimension is invalid, interrupt the object creation...
	if (dimensione < 0) {
		throw std::invalid_argument("Dimension must be positive!");
	} else {
		m_dimensione = dimensione;
		m_data = new double[dimensione];
		for (int i = 0; i < dimensione; ++i)
			m_data[i] = 0.0;
	}
}

// Costruttore di copia: contenuto corrente come copia del dato
Vettore::Vettore(const Vettore& altro): m_data(nullptr), m_dimensione(0)
{
	m_dimensione = altro.m_dimensione;
	if (m_dimensione > 0) {
		// Copy the full content
		m_data = new double[m_dimensione];
		for (int i = 0; i < m_dimensione; ++i)
			m_data[i] = altro.m_data[i];
	}
}



Vettore& Vettore::operator=(const Vettore& altro)
{
	if (this != &altro) {
		delete[] m_data;
		m_data = nullptr;
		m_dimensione = altro.m_dimensione;
		if (m_dimensione > 0) {
			m_data = new double[m_dimensione];
			for (int i = 0; i < m_dimensione; ++i)
				m_data[i] = altro.m_data[i];
		}
	}
	return *this;
}


double Vettore::operator[](int indice) const
{
	if ((indice >= 0) && (indice < m_dimensione)){
		return m_data[indice];
	} else {
		throw std::out_of_range("Out of range");
	}
}

double& Vettore::operator[](int indice)
{
	if ((indice >= 0) && (indice < m_dimensione)){
		return m_data[indice];
	} else {
		throw std::out_of_range("Out of range");
	}
}

int Vettore::get_element(int pos) const
{
	if ((pos >= 0) && (pos < m_dimensione)) {
		return m_data[pos];
	} else {
		throw std::out_of_range("Position out of bounds");
	}
}


int Vettore::get_dimension() const
{
	return m_dimensione;
}


void Vettore::stampa() const
{
	std::cout << "<";
	for(int i = 0; i < m_dimensione; ++i)
		std::cout << m_data[i] << " ";
	std::cout << ">" << std::endl;
}


void Vettore::set_element(int pos, double val)
{
	if ((pos >= 0) && (pos < m_dimensione)) {
		m_data[pos] = val;
	} else {
		throw std::out_of_range("Position out of bounds");
	}
}


Vettore Vettore::operator+(const Vettore& altro) const
{
	// Create a new Vector Object
	Vettore v(m_dimensione);
	#pragma omp parallel for
	for (int i = 0; i < m_dimensione; ++i) {
		v[i] = m_data[i] + altro.m_data[i];
	}
	return v;
}
