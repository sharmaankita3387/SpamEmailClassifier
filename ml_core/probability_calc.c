/**
 * File: probability_calc.c
 * Programmer: Ankita Sharma  
 * Program Description: Implementation of probability calculation utilities
 * Date: November 23, 2025
 * 
 * Implements mathematical safety functions for Naive Bayes:
 * - safe_log(): Prevents log(0) errors by returning small values
 * - Enables stable multiplication of many small probabilities
 * - Critical for handling unknown words in classification
 */

#include <stdio.h>
#include <math.h>
#include "probability_calc.h"

//Help for probability calculation module
void print_probability_calc_help(void) {
    printf("\n=== PROBABILITY CALCULATION MODULE HELP ===\n");
    printf("Mathematical utilities for stable probability computations\n\n");
    
    printf("FUNCTIONS:\n");
    printf("  double safe_log(double x)\n");
    printf("    - Safe logarithm that handles very small probabilities\n");
    printf("    - Prevents log(0) = -infinity errors\n");
    printf("    - Returns: log(x) or -1000.0 for x <= 0\n\n");
    
    printf("PURPOSE:\n");
    printf("  • Prevents numerical underflow in Bayesian calculations\n");
    printf("  • Enables multiplication of many small probabilities\n");
    printf("  • Essential for stable spam classification\n\n");
    
    printf("MATHEMATICAL CONTEXT:\n");
    printf("  Naive Bayes multiplies many P(word|spam) values\n");
    printf("  These can be very small (e.g., 0.0001 × 0.0002 × ...)\n");
    printf("  Using logarithms: log(a×b) = log(a) + log(b)\n");
    printf("  This prevents underflow to zero\n");
}

// Safe version of logarithm for probability calculations
// When probabilities get very small, log(0) would be negative infinity
// This prevents numerical issues by returning a very small number instead
double safe_log(double x) {
    if (x <= 0.0) {
        return -1000.0;  // Return a very small number instead of -infinity
    }
    return log(x);
}