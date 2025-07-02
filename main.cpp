#include <iostream>

/*
    Reading data from the command line
*/
int main(int argc, char **argv){

    std::string name;
    int age;

    std::getline(std::cin, name);

    std::cin >> age;

    std::cout << "Hello " << name << ", you are " << age << "years old." << std::endl;

    return 0;
}