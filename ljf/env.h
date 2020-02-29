#ifndef _ENV_H_
#define _ENV_H_
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<arpa/inet.h>
#include<sys/socket.h>

struct Protocol{
    char message[30];
};

struct UsrInfo{
    float sensor[19];       //传感器
    float wheel[4];         //轮子转速
    float pos[4];           //angle, tomid, toleft, toright
    float car[5];           //fuel, mass, speedXYZ
    float seg[4][4];        
        /*segment information
        float trkType;    //L,R,Straight
        float trkRadius;    //trk-radius: 1/r
        float trkSurface;   //\mu maybe is unimportant
        float trkLength;    //trk->length
    */
    float current_dist;     //距离开始点距离
    float total_dist;       //一共的距离
    float rpm;              //功率
};

struct SvrInfo{
    float steer, accelerate, brake, time;
};

class Env{
    public:
        Env();
        ~Env();
        int restart();
        Protocol send();
        void send(float*, float*, float*, float*, float**, int, int, int, int, int, int, float, float, float);
        SvrInfo get();
        Protocol package(char[30]);
    private:
        int sock;
        struct sockaddr_in serv_addr;
        char message[30];
        int str_len;
        int bug;
};

#endif // _ENV_H_
