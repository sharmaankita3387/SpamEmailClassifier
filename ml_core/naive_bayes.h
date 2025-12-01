/**
 * File: naive_bayes.h
 * Programmer: Ankita Sharma
 * Program Description: Naive Bayes spam classification algorithm implementation
 * Date: November 24, 2025
 *  
 * This module implements a Naive Bayes classifier for email spam detection.
 * It trains on labeled email data and predicts spam probability for new emails.
 * Uses Laplace smoothing and log probabilities for numerical stability.
 * 
 * Features:
 * - Binary classification: SPAM (1) vs NOT-SPAM (0)
 * - Word frequency-based probability calculations
 * - Configurable classification threshold
 * - Memory-efficient vocabulary storage
 */

#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#define MAX_WORD_LENGTH 100
#define INITIAL_VOCAB_SIZE 5000 //Increased for large dataset
#define MAX_EMAIL_LENGTH 10000

// Structure to store probability info for each word
// For each word, we track how often it appears in spam vs not-spam emails
typedef struct {
    char word[MAX_WORD_LENGTH];  // The word itself (e.g., "free")
    int spam_count;              // How many SPAM emails contain this word
    int not_spam_count;          // How many NOT-SPAM emails contain this word
    double prob_spam;            // P(word|spam) - probability word appears in spam
    double prob_not_spam;        // P(word|not_spam) - probability word appears in not-spam
} WordProbability;

// The main model that stores everything our classifier learns
typedef struct {
    WordProbability *vocabulary;  // Array of all words we have learned
    int vocab_size;               // How many unique words we know
    int vocab_capacity;           // How much space we have allocated (for resizing)
    int total_spam_emails;        // Total spam emails in training data
    int total_not_spam_emails;    // Total not-spam emails in training data
    double prior_spam;            // P(spam) - overall probability any email is spam
    double prior_not_spam;        // P(not_spam) - overall probability any email is not-spam
} SpamModel;

// ===== CORE ML FUNCTIONS =====
SpamModel* create_model(void);
void free_model(SpamModel *model);

// Training with tokenized input (from Data Engineer)
void train_naive_bayes_tokens(SpamModel *model, char ***tokenized_emails, int *labels, int email_count);

// Prediction functions
double predict_spam_probability_tokens(SpamModel *model, char **tokens, int token_count);
int classify_email_tokens(SpamModel *model, char **tokens, int token_count, double threshold);

// ===== MODEL STATS =====
void print_model_stats(SpamModel *model);
int get_vocabulary_size(SpamModel *model);
void print_top_spam_words(SpamModel *model, int count);

// ===== HELP SYSTEM =====
void print_naive_bayes_help(void);
void print_ml_help(void);

#endif