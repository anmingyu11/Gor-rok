#include <iostream>
#include <vector>

using namespace std;

int main(){
    vector<int> vec1(3);

    vec1[0] = 12;
    vec1[1] = 15;
    vec1[2] = 20;

    cout << "vec size :" << vec1.size() << endl;


    vec1.push_back(10);


    cout << "vec size :" << vec1.size() << endl;
    cout << "vec last int : " << vec1[vec1.size()-1] << endl;




}
