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
int loadParams( Params *p, Params *cP, Grid *g, Grid *cG);

/* Loads E and T functions */
int initialiseGrid( Grid *g, Grid *cG, Params p, Params pG );

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
void calculateB( Grid *g, double dt );

/* Same old */
void swap_mem(double **array1, double **array2);

/* Pritns results */
void printResults( double t, Grid *g, FILE* output, long nx );

/* ##### MAIN ##### */

int main( int argc, char** argv ){

	long i, j;
	double cur_time, dt, temp_dt, next_diag, errorest;

	FILE* output = NULL;
	Params p, coarseP;
	Grid g;
	Grid coarseG;
	band_mat mMat, coarseMat;

	/* Load parameters */
	if ( loadParams( &p, &coarseP, &g, &coarseG) ){
		printf("Simulation ended\n");
		exit(1);
	}

	/* Initialise grid, set up T and E functions */
	initialiseGrid( &g, &coarseG, p, coarseP );
	output = fopen( "output.txt", "w+" );

	printResults( 0, &coarseG, output, coarseP.nX );
	fclose( output );

	/* Takes dt = min( t_d, (dx)^2/2, (dy)^2/2, 0.1 ) */
	dt = p.t_d;
	if( (g.dx*g.dx)/2 < dt )
		dt = (g.dx*g.dx)/2;
	if( (g.dy*g.dy)/2 < dt )
		dt = (g.dy*g.dy)/2;
	if( 0.1 < dt )
		dt = 0.1;

	printf("%lf\n", dt);
	/* Set up temperature operator */
	loadTOperator( &g, p, dt);
	loadTOperator( &coarseG, coarseP, dt);


	/* Print matrix for debug */
	/*for( i=0; i<g.length2; i++){
		if( i%(p.nY*p.nX) == 0 )
			printf("\n");
		printf("%.2lf ", g.M[i] );
	}
	printf("\n");
	for( i=0; i<coarseG.length2; i++){
		if( i%(coarseP.nY*coarseP.nX) == 0 )
			printf("\n");
		printf("%.2lf ", coarseG.M[i] );
	}
	printf("\n");*/




	if( output == NULL ){
		printf("Couldn't open output file\n");
		printf("Execution stopping, freeing memory\n");
		cleanMemory( &g );
	}

	init_band_mat( &mMat, p.nX, p.nX, g.length );
	init_band_mat( &coarseMat, coarseP.nX, coarseP.nX, coarseG.length );

	/* Set up the bands for band matrix */
	for( i=0; i<g.length; i++ ){
		long index = i*g.length+i;
		for( j=-p.nX; j<=p.nX; j++){
			if( i+j>=0 && i+j<g.length){
				setv( &mMat, i, index%g.length+j, g.M[index+j] );
			}
		}
	}

	/* Set up the bands for coarse band matrix */
	for( i=0; i<coarseG.length; i++ ){
		long index = i*coarseG.length+i;
		for( j=-coarseP.nX; j<=coarseP.nX; j++){
			if( i+j>=0 && i+j<coarseG.length){
				setv( &coarseMat, i, index%coarseG.length+j, coarseG.M[index+j] );
			}
		}
	}


	/* Sets up time dependent variables for the simulation */
	cur_time 	= 0.0;
	temp_dt 	= dt;
	next_diag	= p.t_d;


	/* Needs to be done for first loop to work coorectly */
	printResults( 0, &g, output, p.nX );
	calculateE( &g, p, dt );
	calculateB( &g, dt );

	calculateE( &coarseG, coarseP, dt );
	calculateB( &coarseG, dt );


	/* #### MAIN LOOP #### */
	while( next_diag <= p.t_f ){
		/* If we're set to skip next diagnostic, temporarily modify dt */
		if( cur_time + dt > next_diag){
			temp_dt = next_diag - cur_time;
		}

		/* Update current time */
		cur_time += temp_dt;

		/* Find next T */
		int info = solve_Ax_eq_b( &mMat, g.T_next, g.T );
		if( info != 0 ){
			cleanMemory(&g);
			fclose(output);
			finalise_band_mat( &mMat );
			printf("Lapacke returned an error: %d\n", info);
			exit(1);
		}
		solve_Ax_eq_b( &coarseMat, coarseG.T_next, coarseG.T );


		/* Prints results to file */
		if( fabs(cur_time - next_diag) < 0.00001  ){
			printResults( cur_time, &g, output, p.nX );
			next_diag = cur_time + p.t_d;
		}

		/* Swap memories */
		swap_mem( &(g.E), &(g.E_next) );
		swap_mem( &(g.T), &(g.T_next) );
		swap_mem( &(coarseG.E), &(coarseG.E_next) );
		swap_mem( &(coarseG.T), &(coarseG.T_next) );

		/* Find next E */
		calculateE( &g, p, dt );
		calculateE( &coarseG, coarseP, dt );
		/* Calculate b for Ax=b, b=T-dt*dE, saves it in g.T */
		calculateB( &g, dt );
		calculateB( &coarseG, dt );


		temp_dt = dt;
	}

	/* Calculate error rest */
	errorest = 0;
	long x,y,index, indexC;
	for( i=0; i<g.length; i+=2 ){

		x = i%p.nX;
		y = (i-x)/p.nX;

		if( x%2 == 0 && y%2 == 0 ){
			index 	= x   +  y*p.nX;
			indexC 	= x/2 + (y/2)*coarseP.nX;
			errorest += coarseG.T[indexC]-g.T[index];
		}
	}
	errorest = errorest*errorest;
	errorest /= coarseG.length;
	FILE* error = fopen("errorest.txt","w+");
	fprintf(error, "%lf", errorest);
	fclose(error);


	/* Clean memory */
	fclose( output );
	finalise_band_mat( &mMat );
	cleanMemory( &g );
	finalise_band_mat( &coarseMat );
	cleanMemory( &coarseG );
	/* Clean exit */
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

/* Transforms matrix to index coord (i,j) -> (i*nX, j) */
long toIndex( long i, long j, long nx ){
	return i*nx+j;
}

/*
	Loads operator, including boundary conditions

*/
int loadTOperator( Grid *g, Params p, double dt){

	long i, length, row, index, index1, index2;
	double cy, cx;

	length = g->length;

	g->M = (double*) malloc( sizeof(double) * length * length );

	if( g->M == NULL ){
		printf("Could not allocate memory for the temperature operator\n");
		exit(1);
	}

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

		double diag = 1+2*cx+2*cy;
		/* On diagonal, all the same */
		index = i*length+i;
		g->M[index] = diag;

		/* 1st horizontal bdd */
		if( i < p.nX && i!=0 && i != p.nX-1 ){

			row		= i;
			index 	= row*length + i;

			g->M[ index-1 ] 	= -cx;
			g->M[ index+1 ] 	= -cx;
			g->M[ index+p.nX ]	= -2*cy;
		}

		/* 1st vertical bdd */
		else if( i%p.nX==0 && i!= 0 && i!=length-p.nX ){
			row		= i;
			index = row*length + i;

			g->M[ index+1 ] 	= -2*cx;
			g->M[ index+p.nX ]	= -cy;
			g->M[ index-p.nX ]	= -cy;
		}

		/* 2nd horizontal bdd */
		else if( i > p.nY*(p.nX-1) && i!=length-p.nX-1 && i != length-1 ){
			row		= i;
			index = row*length + i;

			g->M[ index-1 ] 	= -cx;
			g->M[ index+1 ] 	= -cx;
			g->M[ index-p.nX ]	= -2*cy;

		}
		/* 2nd vertical bdd */
		else if( (i+1)%p.nX==0 && i!=0 && i!=length-1 && i!=p.nX-1){
			row		= i;
			index = row*length + i;

			g->M[ index-1 ] 	= -2*cx;
			g->M[ index+p.nX ] 	= -cy;
			g->M[ index-p.nX ]	= -cy;
		}

		else if( i%p.nX != 0 && (i+1)%p.nX!=0 && i!=length-1 ){
			index = i*length+i;
			g->M[ index-1 ] 	= -cx;
			g->M[ index+1 ] 	= -cx;
			g->M[ index+p.nX ]	= -cy;
			g->M[ index-p.nX ]	= -cy;
		}


		/* Corners */
		if( i == 0 ){
		 	g->M[i+1]	=-2*cx;
			g->M[p.nX]	=-2*cy;
		}
		else if( i == p.nX-1 ){
			index = i*length+i;
			g->M[index-1]		=-2*cx;
			g->M[index+p.nX]	=-2*cy;
		}
		else if( i == g->length-p.nX ){
			index = i*length+i;
			g->M[index+1]		=-2*cx;
			g->M[index-p.nX]	=-2*cy;
		}
		else if( i == g->length-1 ){
			index = i*length+i;
			g->M[index-1]		=-2*cx;
			g->M[index-p.nX]	=-2*cy;
		}
	}


	return 0;
}

/*
	Sets up the arrays T,T_next, E, E_next.
	Initialises T & T_next to same values, and E & E_next to same values
*/
int initialiseGrid( Grid *g, Grid *cG, Params p, Params pG ){

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

	cG->T 		= (double*) malloc( sizeof( double ) *cG->length );
	cG->T_next 	= (double*) malloc( sizeof( double ) *cG->length );
	cG->E 		= (double*) malloc( sizeof( double ) *cG->length );
	cG->E_next 	= (double*) malloc( sizeof( double ) *cG->length );
	cG->dE 		= (double*) malloc( sizeof( double ) *cG->length );

	if( cG->T==NULL || cG->T_next==NULL || cG->E==NULL || cG->E_next == NULL ){
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
	int keep=0;
	long x;
	long y;
	long index, index1, index2;
	for( i=0; i<g->length; i++ ){
		int info = fscanf( input, "%lf %lf", &tempT, &tempE );

		if( info != 2 ){
			printf("There was an error reading coefficients.txt\n%d inputs were read instead of 2 in a single line\n", info);
			exit(1);
		}

		x = i%p.nX;
		y = (i-x)/p.nX;

		index = x+y*p.nX;

		g->T[index] 		= tempT;
		g->T_next[index] 	= tempT;
		g->E[index] 		= tempE;
		g->E_next[index]	= tempE;
		g->dE[index]		= 0;

		/*
			Only keep pair indeces in both x and y direction
			Also handle initial condition on non coarse grid points
		*/
		if( x%2 == 0 && y%2 == 0){
			index = x/2 + (y/2)*pG.nX;
			keep=1;
		}else{
			if( tempT >0 || tempE>0 ){
				if( x%2==0 ){
					index = x/2 + ((y-1)/2)*pG.nX;
					if( index < 0 )
						index = x/2 + ((y+1)/2)*pG.nX;
				}else if( y%2 == 0 ){
					index = (x-1)/2 + (y/2)*pG.nX;
					if( index < 0 )
						index = (x+1)/2 + (y/2)*pG.nX;
				}
				keep=1;
			}
		}
		if( keep ){
			cG->T[index] 		+= tempT;
			cG->T_next[index]   += tempT;
			cG->E[index] 		+= tempE;
			cG->E_next[index]	+= tempE;
			cG->dE[index]		= 0;
		}
		keep=0;
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

	return (1+tngh) * t;
}

/*
	Calculates B = T-de
*/
void calculateB( Grid *g, double dt ){
	long i;
	for( i=0; i<g->length; i++){
		g->T[i] = g->T[i] - g->dE[i]*dt;
	}
}
/*
	Loads parameters from input.txt

	@param p: structure to load
*/
int loadParams( Params *p, Params *cP, Grid *g, Grid *cG){
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
			case 0: p->t_f 	= tempD; cP->t_f 	= tempD; 		break;
			case 1: p->t_d 	= tempD; cP->t_d  	= tempD; 		break;
			case 2: p->nX 	= tempL; cP->nX 	= (tempL+1)/2; 	break;
			case 3: p->nY 	= tempL; cP->nY 	= (tempL+1)/2; 	break;
			case 4: p->x_r 	= tempD; cP->x_r 	= tempD;		break;
			case 5: p->y_h 	= tempD; cP->y_h 	= tempD;		break;
			case 6: p->gamma= tempD; cP->gamma	= tempD;		break;
			case 7: p->T_c 	= tempD; cP->T_c 	= tempD;		break;
			case 8: p->T_w 	= tempD; cP->T_w 	= tempD; 		break;
		}
	}
	g->dx = (p->x_r) / (double)(p->nX-1);
	g->dy = (p->y_h) / (double)(p->nY-1);

	cG->dx = (cP->x_r) / (double)(cP->nX-1);
	cG->dy = (cP->y_h) / (double)(cP->nY-1);

	g->length 	= p->nX*p->nY;
	g->length2 	= g->length*g->length;

	cG->length 	= cP->nX*cP->nY;
	cG->length2 = cG->length*cG->length;

	/* Bad parameters handling */
	if( p->t_f <= 0 ){
		printf("Error: length of simulation must be >0\n");
		return 1;
	}
	else if( p->t_d <= 0 ){
		printf("Error: diagnostic timestep must be >0\n");
		return 1;
	}
	else if( p->gamma < 0 ){
		printf("Error: burn time constant < 0\n");
		return 1;
	}
	else if( p->T_c <= 0 ){
		printf("Warning: ignition temperature is not >0, the simulation will continue in case this was purposefully done\n");
	}
	else if( p->T_w <= 0 ){
		printf("Warning: ignition temperature cutoff width is not >0\n");
		return 1;
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
void printResults( double t, Grid *g, FILE* output, long nx ){
	long i, x, y;
	for( i=0; i<g->length; i++){
		x = i%nx;
		y = (i-x)/nx;
		fprintf( output, "%11.4g %11ld %11ld %11.4g %11.4g\n", t, x, y, g->T_next[i], g->E_next[i] );
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
