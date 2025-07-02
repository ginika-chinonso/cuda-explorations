#include <iostream>

/*
    The main function takes two or more arguments
*/
int main(int argc, char **argv){

    int first_number{12};

    int second_number{10};

    int sum = first_number + second_number;

    std::cout << "The sum of the two numbers is: " << sum << std::endl;

    return 0;
}