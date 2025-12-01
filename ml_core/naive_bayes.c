/**
 * File: naive_bayes.c
 * Programmer: Ankita Sharma  
 * Program Description: Core implementation of Naive Bayes spam classifier
 * Date: November 24, 2025
 * 
 * Implements the training and prediction logic for spam detection:
 * - Model creation and memory management
 * - Vocabulary building with word frequency counting
 * - Probability calculations with Laplace smoothing
 * - Email classification using Bayes theorem
 * 
 * Mathematical basis: P(spam|email) ∝ P(spam) × Π P(word|spam)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "naive_bayes.h"

/**
 * Detailed help for Naive Bayes module
 */
void print_naive_bayes_help(void) {
    printf("\n=== NAIVE BAYES CORE MODULE HELP ===\n");
    printf("Implements the Naive Bayes classification algorithm for spam detection\n\n");
    
    printf("MATHEMATICAL BASIS:\n");
    printf("  P(spam|email) ∝ P(spam) × Π P(word|spam)\n");
    printf("  Uses word frequencies with Laplace smoothing for probability estimates\n\n");
    
    printf("CORE FUNCTIONS:\n");
    printf("  SpamModel* create_model(void)\n");
    printf("    - Creates a new empty spam classification model\n");
    printf("    - Returns: Pointer to allocated SpamModel, NULL on failure\n\n");
    
    printf("  void train_naive_bayes_tokens(SpamModel *model, char ***tokenized_emails, int *labels, int email_count)\n");
    printf("    - Trains model on pre-tokenized email data\n");
    printf("    - tokenized_emails: Array of NULL-terminated token arrays\n");
    printf("    - labels: Array of 1 (spam) and 0 (not-spam)\n");
    printf("    - email_count: Number of training emails\n\n");
    
    printf("  double predict_spam_probability_tokens(SpamModel *model, char **tokens, int token_count)\n");
    printf("    - Predicts spam probability (0.0 to 1.0) for tokenized email\n");
    printf("    - tokens: NULL-terminated array of words\n");
    printf("    - token_count: Number of tokens in the array\n");
    printf("    - Returns: Probability between 0.0 and 1.0\n\n");
    
    printf("  int classify_email_tokens(SpamModel *model, char **tokens, int token_count, double threshold)\n");
    printf("    - Classifies email as spam (1) or not-spam (0)\n");
    printf("    - threshold: Decision boundary (typically 0.5)\n");
    printf("    - Returns: 1 for spam, 0 for not-spam\n\n");
    
    printf("FEATURES:\n");
    printf("  • Laplace smoothing for unknown words\n");
    printf("  • Log probabilities for numerical stability\n");
    printf("  • Dynamic vocabulary expansion\n");
    printf("  • Memory efficient storage\n");
    printf("  • Handles 5000+ word vocabulary\n\n");
    
    printf("USAGE EXAMPLE:\n");
    printf("  SpamModel *model = create_model();\n");
    printf("  char *tokens[] = {\"free\", \"money\", NULL};\n");
    printf("  train_naive_bayes_tokens(model, &tokens, &labels, 1);\n");
    printf("  double prob = predict_spam_probability_tokens(model, tokens, 2);\n");
    printf("  free_model(model);\n");
}

 // Quick help for ML module
void print_ml_help(void) {
    printf("\n=== SPAM DETECTION ML MODULE ===\n");
    printf("Quick Usage: classifier_train_tokens() + classifier_predict_tokens()\n\n");
    
    printf("ESSENTIAL FUNCTIONS:\n");
    printf("  Classifier* create_classifier(0.5)\n");
    printf("  classifier_train_tokens(classifier, tokens, labels, count)\n");
    printf("  classifier_predict_tokens(classifier, tokens, count)\n");
    printf("  free_classifier(classifier)\n\n");
    
    printf("DATA FORMAT:\n");
    printf("  Input: NULL-terminated token arrays from Data Engineer\n");
    printf("  Labels: 1 = SPAM, 0 = NOT-SPAM\n");
    printf("  Output: 1 = SPAM, 0 = NOT-SPAM\n\n");
    
    printf("Run '--naive-bayes-help' for detailed algorithm info\n");
}

// Creates a new empty model: like giving our program a blank brain
SpamModel* create_model(void) {
    SpamModel *model = malloc(sizeof(SpamModel));
    if (!model) return NULL;
    
    // Allocates initial space for vocabulary (list of words we'll learn)
    model->vocabulary = malloc(INITIAL_VOCAB_SIZE * sizeof(WordProbability));
    if (!model->vocabulary) {
        free(model);
        return NULL;
    }
    
    // Initialize everything to empty/zero state
    model->vocab_size = 0;
    model->vocab_capacity = INITIAL_VOCAB_SIZE;
    model->total_spam_emails = 0;
    model->total_not_spam_emails = 0;
    model->prior_spam = 0.0;
    model->prior_not_spam = 0.0;
    
    return model;
}

// Cleans up memory, IMPORTANT in C to prevent memory leaks
void free_model(SpamModel *model) {
    if (model) {
        free(model->vocabulary);
        free(model);
    }
}

// Helper: Finds a word in our vocabulary, returns NULL if not found
WordProbability* find_word(SpamModel *model, const char *word) {
    for (int i = 0; i < model->vocab_size; i++) {
        if (strcmp(model->vocabulary[i].word, word) == 0) {
            return &model->vocabulary[i];  // Return pointer to the word's data
        }
    }
    return NULL;  // Word not found in vocabulary
}

// Helper: Adds a word to vocabulary or updates counts if it exists
//Optimized for a larger dataset
int add_word_to_vocab(SpamModel *model, const char *word, int is_spam) {
    // Check if word already exists
    WordProbability *existing_word = find_word(model, word);
    if (existing_word) {
        // Word exists - just update the counts
        if (is_spam == 1) {
            existing_word->spam_count++;
        } else {
            existing_word->not_spam_count++;
        }
        return 1;  // Success
    }
    
    // Resize if more space is needed for the new word
    if (model->vocab_size >= model->vocab_capacity) {
        int new_capacity = model->vocab_capacity * 2;
        WordProbability *new_vocab = realloc(model->vocabulary, new_capacity * sizeof(WordProbability));
        if (!new_vocab) return -1;  // Expansion failed
        model->vocabulary = new_vocab;
        model->vocab_capacity = new_capacity;
    }
    
    // Adds the new word to the vocabulary
    strncpy(model->vocabulary[model->vocab_size].word, word, MAX_WORD_LENGTH - 1);
    model->vocabulary[model->vocab_size].word[MAX_WORD_LENGTH - 1] = '\0';
    
    // Set initial counts based on whether this came from spam or not-spam
    if (is_spam == 1) {
        model->vocabulary[model->vocab_size].spam_count = 1;
        model->vocabulary[model->vocab_size].not_spam_count = 0;
    } else {
        model->vocabulary[model->vocab_size].spam_count = 0;
        model->vocabulary[model->vocab_size].not_spam_count = 1;
    }
    
    // Initialized probabilities to 0, it will be calculated in probabilities_calc.c file
    model->vocabulary[model->vocab_size].prob_spam = 0.0;
    model->vocabulary[model->vocab_size].prob_not_spam = 0.0;
    
    model->vocab_size++;
    return 1;  // Success
}

// MAIN TRAINING FUNCTION that teaches our model to recognize spam
//Uses pre-tokenized data 
void train_naive_bayes_tokens(SpamModel *model, char ***tokenized_emails, int *labels, int email_count) {
    if (!model || !tokenized_emails || !labels || email_count <= 0) return;
    
    printf("Training on %d tokenized emails\n", email_count);
    
    // Reseting counters
    model->total_spam_emails = 0;
    model->total_not_spam_emails = 0;
    
    // Process each email
    for (int i = 0; i < email_count; i++) {
        // Count email type
        if (labels[i] == 1) {
            model->total_spam_emails++;
        } else {
            model->total_not_spam_emails++;
        }
        
        // Add each token to vocabulary
        char **tokens = tokenized_emails[i];
        for (int j = 0; tokens[j] != NULL; j++) {
            add_word_to_vocab(model, tokens[j], labels[i]);
        }
    }
    
    printf("Learned %d unique words\n", model->vocab_size);
    printf("Spam emails: %d, Not-spam emails: %d\n", 
           model->total_spam_emails, model->total_not_spam_emails);
    
    // Calculate priors
    int total_emails = model->total_spam_emails + model->total_not_spam_emails;
    if (total_emails > 0) {
        model->prior_spam = (double)model->total_spam_emails / total_emails;
        model->prior_not_spam = (double)model->total_not_spam_emails / total_emails;
    }
    
    printf("Prior probabilities: P(spam)=%.3f, P(not_spam)=%.3f\n", 
           model->prior_spam, model->prior_not_spam);
    
    // Calculate probabilities with Laplace smoothing
    double alpha = 1.0;
    for (int i = 0; i < model->vocab_size; i++) {
        model->vocabulary[i].prob_spam = (model->vocabulary[i].spam_count + alpha) / 
                                        (model->total_spam_emails + alpha * model->vocab_size);
        
        model->vocabulary[i].prob_not_spam = (model->vocabulary[i].not_spam_count + alpha) / 
                                           (model->total_not_spam_emails + alpha * model->vocab_size);
    }
    
    printf("Training completed!\n");
}

// Predict spam probability for tokenized email
double predict_spam_probability_tokens(SpamModel *model, char **tokens, int token_count) {
    if (!model || !tokens || model->vocab_size == 0) return 0.0;
    
    // Use log probabilities for numerical stability
    double spam_score = log(model->prior_spam);
    double not_spam_score = log(model->prior_not_spam);
    
    // Process each token
    for (int i = 0; i < token_count && tokens[i] != NULL; i++) {
        WordProbability *word_prob = find_word(model, tokens[i]);
        
        if (word_prob) {
            spam_score += log(word_prob->prob_spam);
            not_spam_score += log(word_prob->prob_not_spam);
        } else {
            // Unknown word: use Laplace smoothing
            double unknown_prob = 1.0 / (model->vocab_size + 1);
            spam_score += log(unknown_prob);
            not_spam_score += log(unknown_prob);
        }
    }
    
    // Convert to probability using softmax
    double max_score = (spam_score > not_spam_score) ? spam_score : not_spam_score;
    double exp_spam = exp(spam_score - max_score);
    double exp_not_spam = exp(not_spam_score - max_score);
    
    return exp_spam / (exp_spam + exp_not_spam);
}

// Classify tokenized email
int classify_email_tokens(SpamModel *model, char **tokens, int token_count, double threshold) {
    double spam_prob = predict_spam_probability_tokens(model, tokens, token_count);
    return (spam_prob >= threshold) ? 1 : 0;
}

// Display model statistics
void print_model_stats(SpamModel *model) {
    if (!model) return;
    
    printf("\n=== MODEL STATISTICS ===\n");
    printf("Vocabulary size: %d words\n", model->vocab_size);
    printf("Training data: %d spam, %d not-spam emails\n", 
           model->total_spam_emails, model->total_not_spam_emails);
    printf("Prior probabilities: P(spam)=%.3f, P(not_spam)=%.3f\n", 
           model->prior_spam, model->prior_not_spam);
}

// Show top spam words
void print_top_spam_words(SpamModel *model, int count) {
    if (!model) return;
    
    printf("\nTop %d spam words:\n", count);
    int shown = 0;
    
    for (int i = 0; i < model->vocab_size && shown < count; i++) {
        if (model->vocabulary[i].spam_count > 2) {  // Only words seen multiple times
            double spam_ratio = (double)model->vocabulary[i].spam_count / 
                              (model->vocabulary[i].spam_count + model->vocabulary[i].not_spam_count);
            if (spam_ratio > 0.7) {
                printf("   '%s': %.0f%% spam (%d spam, %d not-spam)\n", 
                       model->vocabulary[i].word, spam_ratio * 100,
                       model->vocabulary[i].spam_count, model->vocabulary[i].not_spam_count);
                shown++;
            }
        }
    }
    
    if (shown == 0) {
        printf("   (No strong spam indicators found)\n");
    }
}

int get_vocabulary_size(SpamModel *model) {
    return model ? model->vocab_size : 0;
}