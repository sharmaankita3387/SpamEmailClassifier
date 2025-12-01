/**
 * File: probability_calc.h
 * Programmer: Ankita Sharma  
 * Program Description: Mathematical utility functions for probability calculations
 * Date: November 23, 2025
 * 
 * Provides safe mathematical operations for probability computations:
 * - Safe logarithm to handle zero probabilities
 * - Prevents numerical underflow in Bayesian calculations
 * - Essential for stable spam probability predictions
 */

#ifndef PROBABILITY_CALC_H
#define PROBABILITY_CALC_H

// Safe logarithm that handles very small numbers
// Regular log(0) would give infinity and break our calculations
// This returns a very small number instead for values <= 0
double safe_log(double x);

// Help system
void print_probability_calc_help(void);

#endif
