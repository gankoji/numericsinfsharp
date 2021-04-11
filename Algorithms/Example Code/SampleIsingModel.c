#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>

#define L 256

#define FOR_ALL_SITES for (i = 0; i < L; i++) for (j = 0; j < L; j++)

int main(void) { 
    // Allocate RNG and file pointer on stack
    gsl_rng * RNG; 
    FILE *out; 

    // Allocate all other local variables
    char name[32];
    int i, j, t, m, S[L][L], n[L], p[L]; 
    double u, T = 3., E;

    // Instantiate the RNG
    RNG = gsl_rng_alloc(gsl_rng_ranlux389); 

    // p and n seem to be a really neat trick to wrap boundaries (i.e. periodic boundary conditions)
    p[0] = L - 1; 
    n[L - 1] = 0;

    for (i = 0; i < L - 1; i++) { 
        n[i] = i + 1; 
        p[i + 1] = i; 
    }

    // Initialize our S matrix randomly to either 0 or 1 at each location
    FOR_ALL_SITES S[i][j] = gsl_rng_get(RNG) % 2;

    for (t = 0; t < 500; t++) { 
        // Create name for output file of this run
        sprintf(name, "%03d.pgm", t);
        // Then open it
        out = fopen(name, "w"); 
        // Write one line, with some header info
        fprintf(out, "P5 %d %d 255\n", L, L);

        // Then, for each element in S, write it out to the file
        FOR_ALL_SITES fputc(255 * S[i][j], out); 
        // Close the file when we're done
        fclose(out);

        // Why 81920 runs of this loop? Thats 10*2^13, odd for sure
        // Each 81920 runs is what he's calling '1 tau' in his plots. 
        for (m = 0; m < 81920; m++) {

            // Sample i and j
            i = gsl_rng_get(RNG) % L; 
            j = gsl_rng_get(RNG) % L;

            // At the place we've sampled, calculate E based on neighbors
            E = 2. * (double)(S[n[i]][j] + S[p[i]][j] + S[i][n[j]] + S[i][p[j]] - 2);

            // Update S at the sampled location.
            // If the RNG we get is lower than the threshold set by E and T
            // Set S as true, otherwise false (in C, that means 1 or 0)
            S[i][j] = (gsl_rng_uniform(RNG) < 1. / (1 + exp(-2. * E / T))); 
        } 
    }

    // Of course, clean up after ourselves when we're done
    gsl_rng_free(RNG); 
    return 0; 
}