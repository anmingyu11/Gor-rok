#include <iostream>


using namespace std;


int main(){
    char hello[] =  {'a','b','c','\0'};

    cout << hello << endl;
    cout << "size of : " << sizeof(hello) << endl;
    cout << "strlen of : " << strlen(hello) << endl;


    hello[1] = '\0';


    cout << hello << endl;
    cout << "size of : " << sizeof(hello) << endl;
    cout << "strlen of : " << strlen(hello) << endl;


}
