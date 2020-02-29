#include "env.h"
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
using namespace std;


Env::Env(){
    bug = 0;
    memset(&serv_addr, 0, sizeof(serv_addr));
    sock = socket(PF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serv_addr.sin_port = htons(9999);

    if( connect(sock,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) == -1) {
        bug = 1;
    }
}


Env::~Env(){
    close(sock);
}

Protocol Env::send(){
    char message_send[] = "Hello world"; 	
    char message[30] = "";
    write(sock, message_send, sizeof(message_send));//send {state, action, }
    str_len = read(sock, message, sizeof(message) - 1);
    if (str_len == -1) {
        bug = 2;
    }
    
    return package(message); 
}


void Env::send(float* sensor, float* wheel, float* posInfo, float* carInfo, float**segInfo,int sensor_size, int wheel_size, int pos_size, int car_size, int seg_size_1, int seg_size_2, float current_dist, float total_dist, float rpm){
    UsrInfo usrInfo;
    memset(&usrInfo, 0, sizeof(UsrInfo));
    memcpy(usrInfo.sensor, sensor, sizeof(sensor[0]) * sensor_size);
    memcpy(usrInfo.wheel, wheel, sizeof(wheel[0]) * wheel_size);
    memcpy(usrInfo.pos, posInfo, sizeof(posInfo[0]) * pos_size);
    memcpy(usrInfo.car, carInfo, sizeof(carInfo[0]) * car_size);
    memcpy(usrInfo.seg, segInfo, sizeof(segInfo[0][0]) * seg_size_1 * seg_size_2);
    usrInfo.current_dist = current_dist;
    usrInfo.total_dist = total_dist;
    usrInfo.rpm = rpm;
    write(sock, (char*)&usrInfo, sizeof(UsrInfo));	//send posInfo, carInfo, segInfo
    str_len = read(sock, message, sizeof(message) - 1);
    return; 
}

SvrInfo Env::get(){
    SvrInfo svrInfo;
    memcpy((char*)&svrInfo, message, sizeof(SvrInfo));
    return svrInfo;
}

Protocol Env::package(char message[30]){
    Protocol x;
    memcpy(x.message, message, 29);
    return x;
}

int Env::restart(){
    int x = 0;
    /*ofstream fout("/home/qby/Documents/torcs-1.3.8-test1/src/drivers/qianbytest/debug.txt",ios::app);
    fout << "in restart" << endl;
    fout.close();*/
    close(sock);
    // x = system("pkill torcs; sleep 0.5; torcs & sh /home/lemon/Workspace/Torcs/torcs-src/src/drivers/lemon/test.sh");
    // return x;
    return 0;
}
