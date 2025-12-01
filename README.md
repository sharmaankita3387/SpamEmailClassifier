## SpamEmailClassifier

### About
> Machine learning spam classifier implemented from scratch in C. Uses Naive Bayes with probabilistic text analysis to detect spam emails with high accuracy. Lightweight, fast, and dependency-free.

### Features
- Pure C Implementation - Zero dependencies, compiles anywhere <br/>
- Complete Naive Bayes - Mathematical implementation with Laplace smoothing <br/>
- Token-Based Processing - Efficient text analysis and feature extraction <br/>
- Confidence Scoring - Probabilistic predictions (0.0 to 1.0) <br/>
- Memory Efficient - Handles 50,000+ word vocabularies <br/>
- Cross-Platform - Linux, macOS, Windows compatible <br/>
- Comprehensive Testing - 95%+ code coverage with gcov <br/>
- Thread-Safe Design - Ready for integration into email servers <br/>

## Quick Start
**Clone & Build**
```
git clone https://github.com/yourusername/SpamEmailClassifier.git
cd SpamEmailClassifier
make test_mlCode
```

Basic Usage
```
#include "src/ml_core/classifier_core.h"

int main() {
    // Create classifier with 0.5 threshold
    Classifier *clf = create_classifier(0.5);
    
    // Train with tokenized emails
    char *tokens[] = {"free", "money", "winner", NULL};
    char ***emails = {tokens};
    int labels[] = {1};  // 1 = spam, 0 = not-spam
    classifier_train_tokens(clf, emails, labels, 1);
    
    // Predict new email
    char *new_email[] = {"congratulations", "winner", NULL};
    int is_spam = classifier_predict_tokens(clf, new_email, 2);
    printf("Result: %s\n", is_spam ? "SPAM 🚩" : "NOT-SPAM ✅");
    
    free_classifier(clf);
    return 0;
}
```

### How It Works
**Naive Bayes Algorithm**
`P(spam|email) ∝ P(spam) × Π P(word|spam)`

1. Training Phase: Counts word frequencies in spam/ham emails
2. Probability Calculation: Computes P(word|spam) with Laplace smoothing
3. Prediction Phase: Applies Bayes' theorem to classify new emails
4. Log Probabilities: Prevents numerical underflow

**Mathematical Foundation**: Laplace smoothing prevents zero probabilities
> P(word|spam) = (count_in_spam + α) / (total_spam_words + α × vocab_size)
<br/>

### Development
**Build & Test**
```
# Run all tests
make test_mlCode
./test_mlCode
```

### Generate coverage reports
```
make coverage
gcov naive_bayes.c probability_calc.c classifier_core.c
```

### Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Learn More
[Naive Bayes Classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)<br/>
[Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)<br/>
[Enron Spam Dataset](https://www.cs.cmu.edu/~enron/)<br/>
[Text Classification with Bayes](https://www.ibm.com/think/topics/naive-bayes)<br/>

### ⭐ Support
**If you find this project useful, please give it a ⭐ on GitHub!**

*Writing ML algorithms from scratch is the best way to truly understand them. Building this project taught me more about probability, optimization, and software engineering than any textbook could.*
