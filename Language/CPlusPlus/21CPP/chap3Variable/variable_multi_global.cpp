#include<iostream>

using namespace std;

int a;
int b;

void multi(){
    cout << "input a:" << endl; 
    cin >> a;

    cout << "input b:" << endl; 
    cin >> b;

}

int main(){
    multi();
    cout << "res : " << a*b <<endl;
}
