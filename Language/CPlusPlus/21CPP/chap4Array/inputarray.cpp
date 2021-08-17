# include<iostream>

using namespace std;

int main(){
    const int ARR_LEN = 5;
    int arrs[ARR_LEN] = {0};

    int eIdx = 0;
    cin >> eIdx;

    int eVal = 0;
    cin >> eVal;

    arrs[eIdx] = eVal;


    for(int i =0; i < ARR_LEN;++i){
        cout << "idx : " << i << " val : " << arrs[i];
    }

}
