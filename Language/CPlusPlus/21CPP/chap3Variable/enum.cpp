#include<iostream>

using namespace std;

enum CandidateDirections{
    n=10,
    s,
    e,
    w
};

int main(){

    cout<< "n : " <<  n <<endl;
    cout<< "s : " <<  s <<endl;
    cout<< "e : " <<  e <<endl;
    cout<< "w : " <<  w <<endl;

    CandidateDirections cd = n;

    cout << "candidate direction : " << cd << endl;

    return 0;

}
