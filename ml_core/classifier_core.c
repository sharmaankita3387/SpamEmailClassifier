/**
 * File: classifier_core.c
 * Programmer: Ankita Sharma  
 * Program Description: Implementation of classifier wrapper and management
 * Date: November 22, 2025
 * 
 * Implements the classifier interface that wraps the core ML model:
 * - Creates and manages classifier instances
 * - Coordinates training and prediction processes
 * - Maintains accuracy tracking for evaluation
 * - Provides clean API for system integration
 */

#include <stdio.h>
#include <stdlib.h>
#include "classifier_core.h"


// Help for classifier core module
void print_classifier_core_help(void) {
    printf("\n=== CLASSIFIER CORE MODULE HELP ===\n");
    printf("High-level interface for spam classification system\n\n");
    
    printf("CLASSIFIER STRUCTURE:\n");
    printf("  typedef struct {\n");
    printf("    SpamModel *model;              // ML model\n");
    printf("    double classification_threshold; // Decision boundary\n");
    printf("    int total_predictions;         // Performance tracking\n");
    printf("    int correct_predictions;       // Accuracy tracking\n");
    printf("  } Classifier;\n\n");
    
    printf("CORE FUNCTIONS:\n");
    printf("  Classifier* create_classifier(double threshold)\n");
    printf("    - Creates a ready-to-use classifier\n");
    printf("    - threshold: Typically 0.5 (50%% spam probability)\n");
    printf("    - Returns: Pointer to Classifier, NULL on failure\n\n");
    
    printf("  void classifier_train_tokens(Classifier *classifier, char ***tokenized_emails, int *labels, int email_count)\n");
    printf("    - Trains classifier on tokenized email data\n");
    printf("    - Wrapper around train_naive_bayes_tokens()\n\n");
    
    printf("  int classifier_predict_tokens(Classifier *classifier, char **tokens, int token_count)\n");
    printf("    - Predicts if email is spam using trained model\n");
    printf("    - Uses classification_threshold for decision\n");
    printf("    - Returns: 1 (spam) or 0 (not-spam)\n\n");
    
    printf("  double get_classifier_accuracy(Classifier *classifier)\n");
    printf("    - Calculates accuracy if labels were provided during prediction\n");
    printf("    - Returns: Accuracy between 0.0 and 1.0\n\n");
    
    printf("INTEGRATION GUIDE:\n");
    printf("  1. create_classifier(0.5)\n");
    printf("  2. classifier_train_tokens() with Data Engineer's tokens\n");
    printf("  3. classifier_predict_tokens() for new emails\n");
    printf("  4. free_classifier() when done\n\n");
    
    printf("TYPICAL THRESHOLDS:\n");
    printf("  0.5 - Balanced (default)\n");
    printf("  0.7 - Conservative (fewer false positives)\n");
    printf("  0.3 - Aggressive (catch more spam)\n");
}

// Creates a ready-to-use classifier
Classifier* create_classifier(double threshold) {
    Classifier *classifier = malloc(sizeof(Classifier));
    if (!classifier) return NULL;
    
    // Create the underlying ML model
    classifier->model = create_model();
    if (!classifier->model) {
        free(classifier);
        return NULL;
    }
    
    // Set classification threshold (0.5 = equal cost for false positives/negatives)
    classifier->classification_threshold = threshold;
    classifier->total_predictions = 0;
    classifier->correct_predictions = 0;
    
    return classifier;
}

// Clean up memory, IMPORTANT to prevent leaks!
void free_classifier(Classifier *classifier) {
    if (classifier) {
        free_model(classifier->model);  // Free the ML model
        free(classifier);               // Free the wrapper
    }
}

// Train with tokenized data from Data Engineer
void classifier_train_tokens(Classifier *classifier, char ***tokenized_emails, int *labels, int email_count) {
    if (!classifier || !tokenized_emails) return;
    train_naive_bayes_tokens(classifier->model, tokenized_emails, labels, email_count);
}

// Predict with tokenized input
int classifier_predict_tokens(Classifier *classifier, char **tokens, int token_count) {
    if (!classifier || !tokens) return 0;
    
    int prediction = classify_email_tokens(classifier->model, tokens, token_count, 
                                         classifier->classification_threshold);
    classifier->total_predictions++;
    
    return prediction;
}

double get_classifier_accuracy(Classifier *classifier) {
    if (!classifier || classifier->total_predictions == 0) return 0.0;
    return (double)classifier->correct_predictions / classifier->total_predictions;
}

void reset_classifier_stats(Classifier *classifier) {
    if (classifier) {
        classifier->total_predictions = 0;
        classifier->correct_predictions = 0;
    }
}
