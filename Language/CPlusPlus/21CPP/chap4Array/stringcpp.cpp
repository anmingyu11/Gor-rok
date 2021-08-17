#include<iostream>
#include<string>

using namespace std;

int main(){
    string G("abcd");
    cout<< G <<endl;


    string line;

    getline(cin , line);
    cout << line << endl;


    string concat;
    concat = line + G;

    cout << concat << endl;

    string copy = concat;


    cout << copy << endl;


    cout << concat.length() << endl;

    return 0;
}
