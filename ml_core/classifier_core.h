/**
 * File: classifier_core.h
 * Programmer: Ankita Sharma  
 * Program Description: Classifier wrapper interface for spam detection system
 * Date: November 22, 2025
 * 
 * Provides a high-level interface for the spam classifier:
 * - Combines ML model with classification settings
 * - Manages training and prediction workflow
 * - Tracks performance metrics and accuracy
 * - Simplifies integration with other system components
 */


#ifndef CLASSIFIER_CORE_H
#define CLASSIFIER_CORE_H

#include "naive_bayes.h"

// Wrapper that combines the ML model with classification settings
// Makes it easier to use our spam detection system
typedef struct {
    SpamModel *model;                   // The actual ML model
    double classification_threshold;    // Decision boundary (usually 0.5)
    int total_predictions;              // Track how many predictions we've made
    int correct_predictions;            // Track how many were correct
} Classifier;



// Creates a new classifier with given threshold
Classifier* create_classifier(double threshold);

// Cleans up classifier memory
void free_classifier(Classifier *classifier);

// Training and prediction with tokens
void classifier_train_tokens(Classifier *classifier, char ***tokenized_emails, int *labels, int email_count);
int classifier_predict_tokens(Classifier *classifier, char **tokens, int token_count);

// Performance tracking
double get_classifier_accuracy(Classifier *classifier);
void reset_classifier_stats(Classifier *classifier);

// Help system
void print_classifier_core_help(void);

#endif