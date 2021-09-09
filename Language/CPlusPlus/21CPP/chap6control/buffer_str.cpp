#include <iostream>
#include <string>

using namespace std;

int main(){
    char buffer[20] = {'\0'};

    cout << "line : " << endl;

    string lineEntered;
    getline(cin,lineEntered);

    if(lineEntered.length() < 20){
        strcpy(buffer,lineEntered.c_str());
        cout << buffer<<endl;
    }

    return 0;
}
