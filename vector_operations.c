#include <stdio.h>
#include <math.h>

/*
void vector_sum (double* a, double* b, double* res_c, int dim);
void print_vector(double* a, int dim);
void scalar_multiplier (double scalar, double* v_in, double* v_out, int dim);
double dot_product (double* a, double* b, int dim);
double norm(double* v, int dim);

int main() {
	double a[3] = {1., 2., 3.};
	double b[3] = {2., 1., 0.};
	double c[3] = {0., 0., 0.};

	print_vector(c, 3);
	vector_sum(a, b, c, 3);
	print_vector(c, 3);
	scalar_multiplier(10., b, c, 3);
	print_vector(c, 3);
	double r = dot_product(a, b, 3);
	printf("Scalar product: %.2f\n", r);

	double n = norm(b, 3);
	printf("Norm: %.2f\n", n);
	return 0;
}
*/

__declspec(dllexport) double dot_product (double* a, double* b, int dim)
{
	double sum = 0.;
	for (int i = 0; i < dim; ++i)
		sum += a[i] * b[i];
	return sum;
}

__declspec(dllexport) double norm(double* v, int dim)
{
	return sqrt(dot_product(v, v, dim));
}


__declspec(dllexport) void vector_sum (double* a, double* b, double* res_c, 
				       int dim)
{
	for (int i = 0; i < dim; ++i)
		res_c[i] = a[i] + b[i];	
}


__declspec(dllexport) void print_vector(double* a, int dim)
{
	for (int i = 0; i < dim; ++i)
		printf("%.2f ", a[i]);
	printf("\n");
}


__declspec(dllexport) void scalar_multiplier (double scalar, double* v_in, 
					      double* v_out, int dim)
{
	for (int i = 0; i < dim; ++i)
		v_out[i] = scalar * v_in[i];
}
