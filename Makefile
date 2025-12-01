test_mlCode: test_mlCode.c naive_bayes.c probability_calc.c classifier_core.c naive_bayes.h probability_calc.h classifier_core.h
	gcc -Wall -g test_mlCode.c naive_bayes.c probability_calc.c classifier_core.c -o test_mlCode -lm

coverage: test_mlCode.c naive_bayes.c probability_calc.c classifier_core.c naive_bayes.h probability_calc.h classifier_core.h
	gcc -Wall -g test_mlCode.c naive_bayes.c probability_calc.c classifier_core.c -o test_mlCode -lm --coverage

clean:
	rm -f test_mlCode *.gcda *.gcno *.gcov
