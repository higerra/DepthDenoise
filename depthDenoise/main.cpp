#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "mrf.h"
#include "ICM.h"
#include <eigen3/Eigen/Eigen>

#define IMGWIDTH 640
#define IMGHEIGHT 480
#define RANGE 3
using namespace std;
using namespace cv;
using namespace Eigen;

char prefix[100];
bool mask[IMGWIDTH*IMGHEIGHT];
int depthNum = 30;
int curframe = 0;
int maxv=-1, minv=99999;
vector<vector<int>> depthdata;
vector<int>nLabels;
//Camera Parameter
vector <Matrix4f> camPara;
Matrix4f intrinsic;
int lambda = 2;

MRF::CostVal dCostFn(int pix, int value)
{
    int diff = depthdata[curframe][pix] - value;
    return diff * diff;
}

void readPara()
{
    char buffer[100];
    for(int i=0;i<depthNum;i++)
    {
        sprintf(buffer, "%s/Geometry/frame%03d.txt",prefix,i);
        Matrix4f temp;
        ifstream camin(buffer);
        for(int y=0;y<4;y++)
        {
            for(int x=0;x<4;x++)
                camin>>temp(y,x);
        }
        temp(3,3) = 1.0;
        camPara.push_back(temp.transpose());
        camin.close();
    }
}


Vector2i getPoint(Matrix4f oriext,Matrix4f dstext,Vector2i imgPoint)
{
    Vector4f imgPoint1_homo(imgPoint[0],imgPoint[1],1.0,0.0);
    Vector4f worldPoint_homo = oriext.inverse()*intrinsic.inverse()*imgPoint1_homo;
    Vector4f imgPoint2_homo =intrinsic*dstext*worldPoint_homo;
    Vector2i imgpt(static_cast<int>(imgPoint2_homo[0]/imgPoint2_homo[2]),static_cast<int>(imgPoint2_homo[1]/imgPoint2_homo[2]));
    return imgpt;
}

bool isValid(Vector2i imgPoint)
{
    if(imgPoint[0]>=0&&imgPoint[0]<IMGWIDTH&&imgPoint[1]>=0&&imgPoint[1]<IMGHEIGHT)
        return 1;
    else
        return 0;
}

int main()
{
    sprintf(prefix, "/Users/Jach/Documents/research/kinect/TextureMapping/depth5.18");
    
    //Set camera Parameter
    intrinsic<<575,0.0,320,0.0,0.0,575,240,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0;
    readPara();
    
    //read depth data
    cout<<"Loading..."<<endl;
    for(curframe = 0;curframe<depthNum;curframe++)
    {
        char buffer[100];
        //read the depth data
        vector<int>curdepthdata;
        nLabels.push_back(-1);
        sprintf(buffer, "%s/depth/frame_depth%03d.txt",prefix,curframe);
        ifstream fin(buffer);
        float temp;
        int itemp;
        fin>>temp;
        curdepthdata.push_back(static_cast<int>(temp*1000));
        for(int i=1;i<IMGWIDTH*IMGHEIGHT;++i)
        {
            fin>>temp;
            itemp = static_cast<int>(temp*1000);
            minv = (itemp<minv&&itemp>0)?itemp:minv;
            maxv = itemp>maxv?itemp:maxv;
            nLabels[curframe] = itemp>nLabels[curframe]?itemp:nLabels[curframe];
            curdepthdata.push_back(itemp);
        }
        fin.close();
        depthdata.push_back(curdepthdata);
    }
    
    //temporal filter
    cout<<"Filtering..."<<endl;
    for(curframe = 0;curframe<depthdata.size();curframe++)
    {
        cout<<"Frame "<<curframe<<endl;
        for(int pix=0;pix<IMGWIDTH*IMGHEIGHT;++pix)
        {
            if(depthdata[curframe][pix] == 0)
            {
                int start = curframe>=RANGE?curframe-RANGE:0;
                int end = curframe<depthdata.size()-RANGE?curframe+RANGE:(int)depthdata.size()-1;
                int sumdepth = 0,count = 0;
                for(int windowind = start;windowind<=end;windowind++)
                {
                    Vector2i ori(pix%IMGWIDTH,pix/IMGWIDTH);
                    Vector2i dst = getPoint(camPara[curframe], camPara[windowind], ori);
                    if(isValid(dst))
                    {
                        if(depthdata[windowind][dst[1]*IMGWIDTH+dst[0]] != 0)
                        {
                            sumdepth += depthdata[windowind][dst[1]*IMGWIDTH+dst[0]];
                            count++;
                        }
                    }
                }
                if(count>0)
                    depthdata[curframe][pix] = static_cast<int>(sumdepth / count);
            }
        }
        
    }
        
    /*for(curframe = 0;curframe<depthNum;curframe++)
    {
        //set the MRF
        cout<<"frame "<<curframe<<endl;
        EnergyFunction *energy = new EnergyFunction(new DataCost(dCostFn),new SmoothnessCost(2, nLabels[curframe]*nLabels[curframe], lambda));
    
        MRF *mrf = new ICM(IMGWIDTH,IMGHEIGHT,nLabels[curframe],energy);
        mrf->initialize();
        mrf->clearAnswer();
        for(int i=0;i<IMGWIDTH*IMGHEIGHT;++i)
                mrf->setLabel(i, depthdata[curframe][i]);
        
        float t;
        cout<<"Optimizing..."<<endl;
        mrf->optimize(5,t);
        MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
        MRF::EnergyVal E_data   = mrf->dataEnergy();
        cout<<"Total Energy: " << E_smooth+E_data<<endl;
        cout<<"Time usage: "<<t<<endl;
    
        //save the result
        delete mrf;
        delete energy;
    }*/
    
    //saving image
    cout<<"Saving..."<<endl;
    for(curframe = 0;curframe<depthdata.size();curframe++)
    {
        Mat outimg = Mat(IMGHEIGHT,IMGWIDTH,CV_8UC3);
        for(int y=0;y<IMGHEIGHT;++y)
        {
            for(int x=0;x<IMGWIDTH;++x)
            {
                int curpix = depthdata[curframe][y*IMGWIDTH+x]*255/(maxv - minv);
                outimg.at<Vec3b>(y,x)[0] = curpix;
                outimg.at<Vec3b>(y,x)[1] = curpix;
                outimg.at<Vec3b>(y,x)[2] = curpix;
            }
        }
        char savebuffer[100];
        sprintf(savebuffer, "%s/depth/depth_denoise%03d.png",prefix,curframe);
        imwrite(string(savebuffer), outimg);
    }
    return 0;
}