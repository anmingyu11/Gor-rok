#include <iostream>
#include <bitset>

using namespace std;

int main() {
    int set[]={1,1,1,4,1,1,1};
    int a = set[0];

    for(int i =1 ; i < sizeof(set)/sizeof(int);++i){
        a ^=set[i];
    }

    cout << a << endl;

    return 0;
}
