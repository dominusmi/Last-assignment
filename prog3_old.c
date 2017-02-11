#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>


struct band_mat{
	long ncol;        /* Number of columns in band matrix            */
	long nbrows;      /* Number of rows (bands in original matrix)   */
	long nbands_up;   /* Number of bands above diagonal           */
	long nbands_low;  /* Number of bands below diagonal           */
	double *array;    /* Storage for the matrix in banded format  */
	/* Internal temporary storage for solving inverse problem */
	long nbrows_inv;  /* Number of rows of inverse matrix   */
	double *array_inv;/* Store the inverse if this is generated */
	int *ipiv;        /* Additional inverse information         */
};
typedef struct band_mat band_mat;

int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns);
void finalise_band_mat( band_mat *bmat);
double *getp(band_mat *bmat, long row, long column);
double getv(band_mat *bmat, long row, long column);
void setv(band_mat *bmat, long row, long column, double val);
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b);
void print_bmat(band_mat *bmat);
void print_mat(band_mat *bmat);
void assertv(int check, char *message);

struct Params{
	long nX;		/* X grid points */
	long nY;		/* Y grid points */
	double t_f;		/* length of time to simulate for. */
	double t_d;		/* Diagnostic timestep */
	double x_r;		/* Right boundary of X domain */
	double y_h;		/* Height of Y domain */
	double gamma; 	/* Burn time constant */
	double T_c;		/* Ignition temperature */
	double T_w;		/* Ignition temperature cutoff */
};
typedef struct Params Params;

struct Grid{
	long length;	/* IxJ */
	long length2;	/* length square, total number of grid points */

	double dx;		/* Grid dx */
	double dy;		/* Grid dy */

	double* T;
	double* T_next;
	double* E;
	double* E_next;
	double* M;
	double* dE;
};
typedef struct Grid Grid;

/* Loads parameters */
int loadParams( Params *p, Grid *g);

/* Loads E and T functions */
int initialiseGrid( Grid *g, Params p );

/* Loads coarse functions */
int loadCoarseFunctions( Grid *g, Params p );

/* Finds the next value of the Energy */
void calculateE( Grid *g, Params p, double dt );

/* Finds the second part of the E eqn, gamma/2 * tanh(..) */
double calculateAlpha( long i, Params p, double* T );

/* Frees memory */
void cleanMemory( Grid *g );

/* Load operator */
int loadTOperator( Grid *g, Params p, double dt );

/* Calculates B = T-dE */
void calculateB( Grid *g );

/* Same old */
void swap_mem(double **array1, double **array2);

/* Pritns results */
void printResults( double t, Grid *g, FILE* output );

/* ##### MAIN ##### */

int main( int argc, char** argv ){

	int error;
	long i, iterations, temp;
	double cur_time, dt, temp_dt, next_diag;

	FILE* output = NULL;
	Params p;
	Grid g;
	band_mat mMat;

	/* Load parameters */
	error = loadParams( &p, &g);
	g.length 	= p.nX*p.nY;
	g.length2 	= g.length*g.length;

	/* Initialise grid, set up T and E functions */
	error = initialiseGrid( &g, p );

	dt = p.t_d/10;
	printf("%lf\n", dt);
	/* Set up temperature operator */
	error = loadTOperator( &g, p, dt);

	for( i=0; i<g.length2; i++){
		if( i%(p.nY*p.nX) == 0 )
			printf("\n");
		printf("%.2lf ", g.M[i] );
	}
	printf("\n");


	output = fopen( "output.dat", "w+" );

	if( error != 0 || output == NULL ){
		if( output == NULL ) { printf("Couldn't open output file\n"); }
		printf("Execution stopping, freeing memory\n");
	}

	init_band_mat( &mMat, g.length-1, g.length, g.length );

	for( i=0; i<g.length2; i++ ){
		temp = floor((double)i/g.length);
		setv(&mMat, temp, i%g.length, g.M[i]);
	}



	/* Sets up time dependent variables for the simulation */
	cur_time 	= 0.0;
	temp_dt 	= dt;
	next_diag	= 0.0;
	iterations 	= (long)ceil( p.t_f / dt );
	iterations 	= 10;

	printf("Iterations %ld\n", iterations);
	printResults( 0, &g, output );

	/* Need to be done for first loop to work coorectly */
	calculateE( &g, p, dt );
	calculateB( &g );

	/* #### MAIN LOOP #### */
	for( i=1; i<iterations; i++ ){
		/* If we're set to skip next diagnostic, temporarily modify dt */
		/*if( cur_time + dt > next_diag){
			temp_dt = next_diag - cur_time;
			iterations++;
		}*/


		/* Find next T */
		for( i=0; i<g.length; i++){
			printf("%lf %lf\n", g.T_next[i], g.T[i]);
		}
		printf("-------------------\n");

		solve_Ax_eq_b( &mMat, g.T_next, g.T );

		for( i=0; i<g.length; i++){
			printf("%lf %lf\n", g.T_next[i], g.T[i]);
		}
		finalise_band_mat( &mMat );
		cleanMemory( &g );
		exit(1);
		/* Prints results to file */
		printResults( (double)i, &g, output );
		/* Swap memories */
		swap_mem( &(g.E), &(g.E_next) );
		swap_mem( &(g.T), &(g.T_next) );

		/* Find next E */
		calculateE( &g, p, dt );
		/* Calculate T-dE, saves it in g.T */
		calculateB( &g );

		temp_dt = dt;
	}

	fclose( output );
	finalise_band_mat( &mMat );
	cleanMemory( &g);
	return 0;
}

/*
    Swap pointers for efficient value exchange

    @param array1:  pointer to be swapped
    @param array2:  -
*/
void swap_mem(double **array1, double **array2) {
    double *tmp;
    tmp     = *array1;
    *array1 = *array2;
    *array2 = tmp;
}

/*
	Frees allocated memory
*/
void cleanMemory( Grid *g){

	free( g->T );
	free( g->T_next );
	free( g->E );
	free( g->E_next );

}

/*
	Loads operator, including boundary conditions

*/
int loadTOperator( Grid *g, Params p, double dt){

	long i, length, row, index, index1, index2;
	double cy, cx;

	length = p.nY*p.nX;

	g->M = (double*) malloc( sizeof(double) * length * length );

	if( g->M == NULL )
		return 1;

	for( i=0; i<length*length; i++){
		g->M[i] = 0.0;
	}

	index = 0;
	index1=0;
	index2=0;
	row=0;
	cx = dt / (g->dx*g->dx);
	cy = dt / (g->dy*g->dy);

	for( i=0; i<length; i++){

		/* 	coefficients 4/3 and -1/3 come from qudratic boundary condition */

		/* 1st horizontal bdd */
		if( i < p.nX && i!=0 && i != p.nX-1 ){
			row		= i;
			index1 	= p.nX + i;
			index2	= 2*p.nX +i;
			g->M[ row * length + index1 ] = 4.0/3;
			g->M[ row * length + index2 ] = -1.0/3;
			printf("%ld %ld\n", index1, index2);
		}

		/* 1st vertical bdd */
		else if( i%p.nX==0 && i!= 0 && i!=length-p.nX ){
			row		= i;
			index1 	= i+1;
			index2	= i+2;
			g->M[ row * length + index1 ] = 4.0/3;
			g->M[ row * length + index2 ] = -1.0/3;
		}

		/* 2nd horizontal bdd */
		else if( i > p.nX*(p.nX-1)-1 && i!=length-p.nX && i != length-1 ){
			row		= i;
			index1 	= i-p.nX;
			index2	= i-2*p.nX;
			g->M[ row * length + index1 ] = 4.0/3;
			g->M[ row * length + index2 ] = -1.0/3;

		}
		/* 2nd vertical bdd */
		else if( (i+1)%p.nX==0 && i!=0 && i!=length-1 && i!=p.nX-1){
			row		= i;
			index1 	= i-1;
			index2	= i-2;
			g->M[ row * length + index1 ] = 4.0/3;
			g->M[ row * length + index2 ] = -1.0/3;
		}

		else if( i!= 0 && i%p.nX != 0 && (i+1)%p.nX!=0 && i!=length-1 ){
			index = i*length+i;
			g->M[ index ] 		= 1+2*cx+2*cy;
			g->M[ index-1 ] 	= -cx;
			g->M[ index+1 ] 	= -cx;
			g->M[ index+p.nX ]	= -cy;
			g->M[ index-p.nX ]	= -cy;
		}


		/* Corners */
		/*if( i == 0 ){
			M[1]=0.5;
			M[nX]=0.5;
		}
		if( i == ny-1 ){
			M[]
		}*/
	}


	return 0;
}

/*
	Sets up the arrays T,T_next, E, E_next.
	Initialises T & T_next to same values, and E & E_next to same values
*/
int initialiseGrid( Grid *g, Params p ){

	int i;
	double tempT, tempE;
	FILE* input = NULL;

	/* Malloc arrays & error check */
	g->T 		= (double*) malloc( sizeof( double ) *g->length );
	g->T_next 	= (double*) malloc( sizeof( double ) *g->length );
	g->E 		= (double*) malloc( sizeof( double ) *g->length );
	g->E_next 	= (double*) malloc( sizeof( double ) *g->length );
	g->dE 		= (double*) malloc( sizeof( double ) *g->length );

	if( g->T==NULL || g->T_next==NULL || g->E==NULL || g->E_next == NULL ){
		printf("The memory for an array could not be allocated in initialiseGrid\n");
		return 1;
	}

	/* Open file & error check */
	input = fopen("coefficients.txt", "r");
	if( input == NULL ){
		printf("Could not open coefficients.txt\n");
		return 1;
	}

	/* Loop through all points, array per array */
	long tot = p.nX*p.nY;
	long x;
	long y;
	long index;
	for( i=0; i<tot; i++ ){
		fscanf( input, "%lf %lf", &tempT, &tempE );

		x = i%p.nX;
		y = floor((double)i/p.nX);

		index = x+y*p.nX;

		g->T[index] 		= tempT;
		g->T_next[index] 	= tempT;
		g->E[index] 		= tempE;
		g->E_next[index]	= tempE;
		g->dE[index]		= 0;
	}

	fclose( input );
	return 0;
}

/*
	Finds next value of energy
*/
void calculateE( Grid *g, Params p, double dt ){

	long i;
	long N;
	double alpha;

	N = p.nX * p.nY;

	for( i=0; i<N; i++ ){
		alpha = calculateAlpha( i, p, g->T );
		/* Calculating dE/dt */
		g->dE[i] 	= -1*g->E[i] * alpha;
		/* Isolating E(t+1) from the equation */
		g->E_next[i]= -1*g->E[i] * alpha * dt + g->E[i];
	}
}

/*
	Calculates the second part of the energy equation, gamma/2 * tanh()
*/
double calculateAlpha( long i, Params p, double* T ){
	double t = p.gamma / 2;
	double tngh = tanh( (T[i]-p.T_c) / p.T_w );

	return tngh * t;
}

/*
	Calculates B = T-de
*/
void calculateB( Grid *g ){
	long i;
	for( i=0; i<g->length; i++){
		g->T[i] = g->T[i] - g->dE[i];
	}
}
/*
	Loads parameters from input.txt

	@param p: structure to load
*/
int loadParams( Params *p, Grid *g ){
	int i;
	int error;
	long tempL;
	double tempD;
	FILE *input = fopen("input.txt", "r");

	/* Checks existence of file */
	if( input == NULL ){
		printf("Error: file input.txt could not be opened\n");
		exit(1);
	}

	/* Load params */
	for( i = 0; i < 9; i++ ){

		/*
			I and J are scanned as long
			and the rest as double
			This is done by using different temporary variables:
			tempD, tempL (Double, Long)
		*/

		if( i == 2 || i == 3 ){ 	error = fscanf(input, "%ld", &tempL);}
		else{						error = fscanf(input, "%lf", &tempD);}

		/* Check for reading file error */
		if( error != 1 ){
			printf("Error: fscanf returned %d while reading input.txt\n", error);
			printf("=> This may mean there weren't enough parameters\n");
			return 1;
		}
		switch(i){
			case 0: p->t_f 	= tempD; break;
			case 1: p->t_d 	= tempD; break;
			case 2: p->nX 	= tempL; break;
			case 3: p->nY 	= tempL; break;
			case 4: p->x_r 	= tempD; break;
			case 5: p->y_h 	= tempD; break;
			case 6: p->gamma= tempD; break;
			case 7: p->T_c 	= tempD; break;
			case 8: p->T_w 	= tempD; break;
		}
	}
	g->dx = (p->x_r) / (double)(p->nX-1);
	g->dy = (p->y_h) / (double)(p->nY-1);


	/* Bad parameters handling */
	if( p->t_f <= 0 ){
		printf("Error: length of simulation must be >0\n");
		return 1;
	}
	else if( p->t_d <= 0 ){
		printf("Error: diagnostic timestep must be >0\n");
		return 1;
	}
	else if( p->gamma <= 0 ){
		printf("Error: burn time constant\n");
		return 1;
	}
	else if( p->T_c <= 0 ){
		printf("Warning: ignition temperature is not >0, the simulation will continue in case this was purposefully done\n");
	}
	else if( p->T_c <= 0 ){
		printf("Warning: ignition temperature cutoff width is not >0, the simulation will continue in case this was purposefully done\n");
	}
	/* Clean exit */
	fclose( input );
	return 0;
}

int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
	bmat->nbrows 	= nbands_lower + nbands_upper + 1;
	bmat->ncol   	= n_columns;
	bmat->nbands_up = nbands_upper;
	bmat->nbands_low= nbands_lower;
	bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
	bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
	bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
	bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
	if (bmat->array==NULL||bmat->array_inv==NULL) {
		return 0;
	}
	/* Initialise array to zero */
	long i;
	for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
		bmat->array[i] = 0.0;
	}
	return 1;
};

/* Finalise function: should free memory as required */
void finalise_band_mat( band_mat *bmat) {
	free( bmat->array );
	free( bmat->array_inv );
	free( bmat->ipiv );
}

/* Get a pointer to a location in the band matrix, using
the row and column indexes of the full matrix.           */
double *getp(band_mat *bmat, long row, long column) {
	int bandno = bmat->nbands_up + row - column;
	if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol || bandno<0 || bandno>=bmat->nbrows ) {
		printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
		exit(1);
	}
	if (bandno<0 || bandno>=bmat->nbrows) {
		return NULL;
	}
	return &bmat->array[bmat->nbrows*column + bandno];
}

/* Retrun the value of a location in the band matrix, using
the row and column indexes of the full matrix.           */
double getv(band_mat *bmat, long row, long column) {
	double *valr = getp(bmat,row,column);
	if (valr==NULL) return 0.0;
	return *valr;
}

void setv(band_mat *bmat, long row, long column, double val) {
	double *valr = getp(bmat,row,column);
	if (valr==NULL) {
		printf("Attempt to set values out of band in setv: %ld %ld %ld %ld\n",row,column,bmat->ncol,bmat->nbrows);
		exit(1);
	}
	*valr = val;
}

/* Solve the equation Ax = b for a matrix a stored in band format
and x and b real arrays                                          */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
	/* Copy bmat array into the temporary store */
	int i,bandno;
	for(i=0;i<bmat->ncol;i++) {
		for (bandno=0; bandno<bmat->nbrows; bandno++) {
			bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
		}
		x[i] = b[i];
	}

	long nrhs = 1;
	long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
	int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
	return info;
}

/* Print the bands of a banded matrix */
void print_bmat(band_mat *bmat) {
	long i,j;
	for(i=0; i<bmat->ncol;i++) {
		printf("Col: %ld Row \n",i);
		for(j=0; j<bmat->nbrows; j++) {
			printf("%ld, %g \n",j,bmat->array[bmat->nbrows*i + j]);
		}
	}
}

/* Pretty print the full matrix. */
void print_mat(band_mat *bmat) {
	long row,column;
	printf("       ");
	for(column=0; column<bmat->ncol; column++) {
		printf("%11ld ",column);
	}
	printf("\n");
	printf("       ");
	for(column=0; column<bmat->ncol; column++) {
		printf(" ---------- ");
	}
	printf("\n");

	for(row=0; row<bmat->ncol;row++) {
		printf("%4ld : ",row);
		for(column=0; column<bmat->ncol; column++) {
			double flval = 0.0;
			int bandno = bmat->nbands_up + row - column;
			if( bandno<0 || bandno>=bmat->nbrows ) {
			} else {
				flval = getv(bmat,row,column);
			}
			printf("%11.4g ",flval);
		}
		printf("\n");
	}
}

/* Prints resutls */
void printResults( double t, Grid *g, FILE* output ){
	long i;
	for( i=0; i<g->length; i++){
		fprintf( output, "%g %g %g\n", t, g->T_next[i], g->E_next[i] );
		fflush( output );
	}
}

/*  Catch errors: exit with specified message */
void assertv(int check, char *message) {
	if(!check) {
		printf("Fatal error: stopping. Message: %s \n",message);
		exit(1);
	}
	return;
}
