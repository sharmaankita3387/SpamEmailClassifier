/**
 * File: test_mlCode.c
 * Programmer: Ankita Sharma
 * Program Description: Test suite for ML spam detection with tokenized data
 * Date: November 25, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "naive_bayes.h"
#include "classifier_core.h"
#include "probability_calc.h"

// Helper function to create tokenized test data
char*** create_test_tokenized_emails(int *email_count) {
    // Sample tokenized emails (what Data Engineer will provide)
    *email_count = 6;
    
    // Allocate memory for emails
    char ***emails = malloc(*email_count * sizeof(char**));
    
    // Email 1: Spam
    emails[0] = malloc(6 * sizeof(char*));
    emails[0][0] = "congratulations"; emails[0][1] = "you"; emails[0][2] = "won"; 
    emails[0][3] = "free"; emails[0][4] = "lottery"; emails[0][5] = NULL;
    
    // Email 2: Not-spam
    emails[1] = malloc(5 * sizeof(char*));
    emails[1][0] = "meeting"; emails[1][1] = "tomorrow"; emails[1][2] = "10am"; 
    emails[1][3] = "conference"; emails[1][4] = NULL;
    
    // Email 3: Spam
    emails[2] = malloc(5 * sizeof(char*));
    emails[2][0] = "urgent"; emails[2][1] = "account"; emails[2][2] = "suspended"; 
    emails[2][3] = "verify"; emails[2][4] = NULL;
    
    // Email 4: Not-spam
    emails[3] = malloc(4 * sizeof(char*));
    emails[3][0] = "lunch"; emails[3][1] = "restaurant"; emails[3][2] = "noon"; 
    emails[3][3] = NULL;
    
    // Email 5: Spam
    emails[4] = malloc(6 * sizeof(char*));
    emails[4][0] = "winner"; emails[4][1] = "claim"; emails[4][2] = "prize"; 
    emails[4][3] = "money"; emails[4][4] = "now"; emails[4][5] = NULL;
    
    // Email 6: Not-spam
    emails[5] = malloc(4 * sizeof(char*));
    emails[5][0] = "homework"; emails[5][1] = "assignment"; emails[5][2] = "due"; 
    emails[5][3] = NULL;
    
    return emails;
}

// Helper to free tokenized data
void free_test_tokenized_emails(char ***emails, int email_count) {
    for (int i = 0; i < email_count; i++) {
        free(emails[i]);
    }
    free(emails);
}

// Count tokens in an array
int count_tokens(char **tokens) {
    int count = 0;
    while (tokens[count] != NULL) {
        count++;
    }
    return count;
}

int main(int argc, char *argv[]) {
    
    // ===== HELP SYSTEM =====
    // Check if user wants help
    if (argc > 1) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_ml_help();
            return 0;
        }
        else if (strcmp(argv[1], "--naive-bayes-help") == 0) {
            print_naive_bayes_help();
            return 0;
        }
        else if (strcmp(argv[1], "--classifier-help") == 0) {
            print_classifier_core_help();
            return 0;
        }
        else if (strcmp(argv[1], "--probability-help") == 0) {
            print_probability_calc_help();
            return 0;
        }
        else {
            printf("Unknown option: %s\n", argv[1]);
            printf("Use --help for available options\n");
            return 1;
        }
    }
    
    printf("Testing SpamCheck ML with Tokenized Data...\n\n");
    
    // Create classifier
    Classifier *classifier = create_classifier(0.5);
    if (!classifier) {
        printf("Failed to create classifier\n");
        return 1;
    }
    
    // Create sample tokenized training data (like Data Engineer would provide)
    int email_count;
    char ***training_emails = create_test_tokenized_emails(&email_count);
    int labels[] = {1, 0, 1, 0, 1, 0};  // 1=spam, 0=not-spam
    
    // Train the model
    printf("Training on %d tokenized emails...\n", email_count);
    classifier_train_tokens(classifier, training_emails, labels, email_count);
    
    // Show what the model learned
    print_model_stats(classifier->model);
    print_top_spam_words(classifier->model, 5);
    
    // Test predictions with tokenized emails
    printf("\nTesting predictions with tokenized data:\n");
    
    // Test email 1: Should be SPAM
    char *test1[] = {"free", "money", "winner", NULL};
    int token_count1 = count_tokens(test1);
    int prediction1 = classifier_predict_tokens(classifier, test1, token_count1);
    double prob1 = predict_spam_probability_tokens(classifier->model, test1, token_count1);
    printf("Tokens: 'free money winner'\n");
    printf("  Prediction: %s (confidence: %.1f%%)\n\n", 
           prediction1 ? "SPAM" : "NOT-SPAM", prob1 * 100);
    
    // Test email 2: Should be NOT-SPAM
    char *test2[] = {"meeting", "project", "update", NULL};
    int token_count2 = count_tokens(test2);
    int prediction2 = classifier_predict_tokens(classifier, test2, token_count2);
    double prob2 = predict_spam_probability_tokens(classifier->model, test2, token_count2);
    printf("Tokens: 'meeting project update'\n");
    printf("  Prediction: %s (confidence: %.1f%%)\n\n", 
           prediction2 ? "SPAM" : "NOT-SPAM", prob2 * 100);
    
    // Test email 3: Should be SPAM
    char *test3[] = {"urgent", "verify", "account", NULL};
    int token_count3 = count_tokens(test3);
    int prediction3 = classifier_predict_tokens(classifier, test3, token_count3);
    double prob3 = predict_spam_probability_tokens(classifier->model, test3, token_count3);
    printf("Tokens: 'urgent verify account'\n");
    printf("  Prediction: %s (confidence: %.1f%%)\n\n", 
           prediction3 ? "SPAM" : "NOT-SPAM", prob3 * 100);
    
    // Test email 4: Unknown words
    char *test4[] = {"unknown", "words", "test", NULL};
    int token_count4 = count_tokens(test4);
    int prediction4 = classifier_predict_tokens(classifier, test4, token_count4);
    double prob4 = predict_spam_probability_tokens(classifier->model, test4, token_count4);
    printf("Tokens: 'unknown words test'\n");
    printf("  Prediction: %s (confidence: %.1f%%)\n\n", 
           prediction4 ? "SPAM" : "NOT-SPAM", prob4 * 100);
    
    // Test with single token
    char *test5[] = {"free", NULL};
    int token_count5 = count_tokens(test5);
    int prediction5 = classifier_predict_tokens(classifier, test5, token_count5);
    double prob5 = predict_spam_probability_tokens(classifier->model, test5, token_count5);
    printf("Tokens: 'free'\n");
    printf("  Prediction: %s (confidence: %.1f%%)\n\n", 
           prediction5 ? "SPAM" : "NOT-SPAM", prob5 * 100);
    
    // Show help
    printf("\n");
    print_ml_help();
    
    // Cleanup
    free_test_tokenized_emails(training_emails, email_count);
    free_classifier(classifier);
    
    printf("\n ML Tokenized Data Test Completed Successfully!\n");
    return 0;
}