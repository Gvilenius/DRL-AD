/***************************************************************************

    file                 : ljf.cpp
    created              : 2019年 05月 22日 星期三 20:00:06 CST
    copyright            : (C) 2002 qby

 ***************************************************************************/

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h> 
#include <string.h> 
#include <math.h>

#include <tgf.h> 
#include <track.h> 
#include <car.h> 
#include <raceman.h> 
#include <robottools.h>

#include <robot.h>
#include <iostream>
#include <fstream>

//#include <tensorflow/c/c_api.h>

#include "env.h"
//#include "agent.h"
using namespace std;

#define HALF_PI PI/2

static tTrack	*curTrack;

static void initTrack(int index, tTrack* track, void *carHandle, void **carParmHandle, tSituation *s); 
static void newrace(int index, tCarElt* car, tSituation *s); 
static void drive(int index, tCarElt* car, tSituation *s); 
static void endrace(int index, tCarElt *car, tSituation *s);
static void shutdown(int index);
static int  InitFuncPt(int index, void *pt); 

float sensor_calc_str(tTrackSeg *seg, float x, float y, float sensor_angle, float remaining_range);
float sensor_calc_lft_rgt(tTrackSeg *seg, float x, float y, float sensor_angle, float remaining_range);
		
bool check_max_circle_intersect(float angle, float radius_max, float radius_x, float y_dn, float y_up, float remaining_range, float &dist_returned);
bool check_min_circle_intersect(float angle, float radius_min, float radius_x, float y_dn, float y_up, float remaining_range, float &dist_returned);
bool check_up_border_intersect(float angle, float radius_min, float radius_x, float y_up, float remaining_range, float &dist_returned, float &angle_returned, float &x_returned);
bool check_down_border_intersect(float angle, float radius_min, float radius_x, float y_dn, float remaining_range, float &dist_returned, float &angle_returned, float &x_returned);

const float MAX_UNSTUCK_SPEED = 5.0;   /* [m/s] */
const float MIN_UNSTUCK_DIST = 3.0;    /* [m] */
const float MAX_UNSTUCK_ANGLE = 30.0/180 * PI;
const float MAX_UNSTUCK_COUNT = 30.0;
const float SHIFT = 0.98f;							///< [-] (% of rpmredline) When do we like to shift gears.

static int stuck = 0;

static int lock = 0;

static int count = 0;               //wait at first 3 seconds

static int hz = 0;                  //count for 0~10 steps

static int cut = 5;                 //every 10 steps make one decision

static SvrInfo last_svrInfo;        //last_decision
static int start_id = 0;            //start_id
static float total_length = 0.0;    //total_length
static float current_length = 0.0;  //current_length
static int sensor_range = 200;

const float FULL_ACCEL_MARGIN = 1.0;   /* [m/s] */

Env* env = NULL;
//Agent* agent = NULL;

bool train = TRUE;
// bool train = FALSE;

/* 
 * Module entry point  
 */ 
extern "C" int 
ljf(tModInfo *modInfo) 
{
    memset(modInfo, 0, 10*sizeof(tModInfo));

    modInfo->name    = strdup("ljf");		/* name of the module (short) */
    modInfo->desc    = strdup("");	/* description of the module (can be long) */
    modInfo->fctInit = InitFuncPt;		/* init function */
    modInfo->gfId    = ROB_IDENT;		/* supported framework version */
    modInfo->index   = 1;

    return 0; 
} 

/*check if stuck*/

float getDistToSegEnd(tCarElt* car)
{
    if (car->_trkPos.seg->type == TR_STR) {
        return car->_trkPos.seg->length - car->_trkPos.toStart;
    } else {
        return (car->_trkPos.seg->arc - car->_trkPos.toStart)*car->_trkPos.seg->radius;
    }
}

bool isStuck(tCarElt* car)
{
    float angle = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw;
    NORM_PI_PI(angle);
    if (fabs(angle) > MAX_UNSTUCK_ANGLE &&
        car->_speed_x < MAX_UNSTUCK_SPEED &&
        fabs(car->_trkPos.toMiddle) > MIN_UNSTUCK_DIST) {
        if (stuck > MAX_UNSTUCK_COUNT && car->_trkPos.toMiddle*angle < 0.0) {
            return true;
        } else {
            stuck++;
            return false;
        }
    } else {
        stuck = 0;
        return false;
    }
}

float getAllowedSpeed(tTrackSeg *segment)
{
    if (segment->type == TR_STR) {
        return FLT_MAX;
    } else {
        float mu = segment->surface->kFriction;
        return sqrt(mu*G*segment->radius);
    }
}

float getAccel(tCarElt* car)
{
    float allowedspeed = getAllowedSpeed(car->_trkPos.seg);
    float gr = car->_gearRatio[car->_gear + car->_gearOffset];
    float rm = car->_enginerpmRedLine;
    if (allowedspeed > car->_speed_x + FULL_ACCEL_MARGIN) {
        return 1.0;
    } else {
        return allowedspeed/car->_wheelRadius(REAR_RGT)*gr /rm;
    }
}

float getBrake(tCarElt* car)
{
    tTrackSeg *segptr = car->_trkPos.seg;
    float currentspeedsqr = car->_speed_x*car->_speed_x;
    float mu = segptr->surface->kFriction;
    float maxlookaheaddist = currentspeedsqr/(2.0*mu*G);
	float lookaheaddist = getDistToSegEnd(car);
    float allowedspeed = getAllowedSpeed(segptr);
    if (allowedspeed < car->_speed_x) return 0.2;
        segptr = segptr->next;
    while (lookaheaddist < maxlookaheaddist) {
        allowedspeed = getAllowedSpeed(segptr);
        if (allowedspeed < car->_speed_x) {
            float allowedspeedsqr = allowedspeed*allowedspeed;
            float brakedist = (currentspeedsqr - allowedspeedsqr) / (2.0*mu*G);
                if (brakedist > lookaheaddist) {
                return 0.2;
            }
        }
        lookaheaddist += segptr->length;
        segptr = segptr->next;
    }
    return 0.0;
}

/* Module interface initialization. */
static int 
InitFuncPt(int index, void *pt) 
{ 
    tRobotItf *itf  = (tRobotItf *)pt; 

    itf->rbNewTrack = initTrack; /* Give the robot the track view called */ 
				 /* for every track change or new race */ 
    itf->rbNewRace  = newrace; 	 /* Start a new race */
    itf->rbDrive    = drive;	 /* Drive during race */
    itf->rbPitCmd   = NULL;
    itf->rbEndRace  = endrace;	 /* End of the current race */
    // itf->rbShutdown = shutdown;	 /* Called before the module is unloaded */
	itf->rbShutdown = NULL;	 /* Called before the module is unloaded */

    itf->index      = index; 	 /* Index used if multiple interfaces */
    return 0; 
} 

/* Called for every track change or new race. */
static void  
initTrack(int index, tTrack* track, void *carHandle, void **carParmHandle, tSituation *s) 
{ 
    curTrack = track;
    *carParmHandle = NULL;
    env = new Env();
    last_svrInfo.steer = 0;
    last_svrInfo.accelerate = 0;
    last_svrInfo.brake = 0;
//    agent = new Agent();
} 

/* Start a new race. */
static void  
newrace(int index, tCarElt* car, tSituation *s) 
{
    	/* initialize the list of opponents */
	// opponents = new Opponents(s, this);
	// opponent = opponents->getOpponentPtr();
} 


/* Called before the module is unloaded */
static void
shutdown(int index)
{
//    ofstream fout("/home/qby/Documents/torcs-1.3.8-test1/src/drivers/ljf/debug.txt",ios::app);
//    fout << "end race"  << endl;
//    fout << TF_Version() << endl;
//    fout << "end race" << endl;
//    fout.close();
    env->restart();
    delete env;
//    delete agent;
}

  /**
       \brief Estimate torque from max torque and max power ratings.

       Uses a piecewise linear model to estimate torque, given the value
       of torque at the max torque and power points and assuming that the
       torque drops to 0 at 0 revs and that it decays linearly after the
       maximum power at a rate 4 times than that of the decay between
       maximum torque and maximum power.
    */
float EstimateTorque (float rpm, tCarElt* car)
{
	#if 0
	float MaxPw = 350000;
	float rpmMaxPw = 7000.0f * (2.0f*M_PI/60.0f);
	float MaxTq = 536.0;
	float rpmMaxTq = 5000.0f * (2.0f*M_PI/60.0f);
	#else
	float MaxPw = car->_engineMaxPw;
	float rpmMaxPw = car->_enginerpmMaxPw;
	float MaxTq = car->_engineMaxTq;
	float rpmMaxTq = car->_enginerpmMaxTq;
	#endif
	float rpmMax = car->_enginerpmMax;
	float TqAtMaxPw = MaxPw/rpmMaxPw;
	float PwAtMaxRevs = .5*MaxPw;
	float TqAtMaxRevs = PwAtMaxRevs/rpmMax;
	//printf ("pw-estimated Tq at maxrevs: %f@%f\n", TqAtMaxRevs, car->_enginerpmMax);

	float t [] = {
	    0.0,
	    MaxTq,
	    TqAtMaxPw,
	    TqAtMaxRevs,
	    0.0
	};
	float a [] = {		
	    0.0,
	    rpmMaxTq,
	    rpmMaxPw,
	    rpmMax,
	    rpmMax*2.0
	};
	int N = 5;
	for (int i=0; i<N - 1; i++) {
	    if (rpm>a[i] && rpm<=a[i+1]) {
		float d = (a[i+1]-a[i]);
		float dr = (rpm - a[i]);
		float D = (dr/d);
		float est_1 = t[i] * (1-D) + D * t[i+1];
		//float est_2 = (t[i]*a[i]*(1-D) + D*t[i+1]*a[i+1])/rpm; 
		return est_1;

	    }
	}
	// if rpm < 0, or rpm >maxrpm
	return 0.0;
}
int getGear(tCarElt* car)
{
	if (car->_gear <= 0) {
	    return 1;
	}
	float gr_up = car->_gearRatio[car->_gear + car->_gearOffset];
	float omega = car->_enginerpmRedLine/gr_up;
	float wr = car->_wheelRadius(2);
	//float rpm = car->_enginerpm;
	int next_gear = car->_gear + 1;
	if (next_gear> car->_gearNb) {
	    next_gear = car->_gear;
	}

	float next_ratio = car->_gearRatio[next_gear + car->_gearOffset];
	float next_rpm = next_ratio * car->_speed_x / wr;
	float rpm = gr_up * car->_speed_x / wr;//car->_enginerpm;
	if (omega*wr*SHIFT < car->_speed_x) {
	    return car->_gear + 1;
	} else if (EstimateTorque(next_rpm, car)*next_ratio > EstimateTorque(rpm, car)*gr_up) {
	    return car->_gear + 1;
	} else {
	    float gr_down = car->_gearRatio[car->_gear + car->_gearOffset - 1];
	    float prev_rpm = gr_down * (car->_speed_x) / wr;

	    if (prev_rpm < car->_enginerpmMaxPw * SHIFT
		&& car->_gear > 1
		&& (EstimateTorque (prev_rpm, car) * gr_down
		    > EstimateTorque (rpm, car) * gr_up)) {
		return car->_gear - 1;
	    }
	}
	return car->_gear;
}
//sensor
float sensor_calc_str(tTrackSeg *seg, float x_coord, float y_coord, float angle, float remaining_range) 
{
	float x_sx = x_coord;						//meters
	float x_dx = (seg->startWidth) - x_coord;	//meters
	float y_up = (seg->length) - y_coord;		//meters
	float y_dn = y_coord;						//meters

	if (cos(angle) == 0)
	{
		if (angle > 0)
		{
			if (x_sx < remaining_range) return x_sx;
			else return remaining_range;
		}
		else 
		{
			if (x_dx < remaining_range) return x_dx;
			else return remaining_range;
		}
	}
	else
		if (cos(angle) > 0 && sin(angle) >= 0)
		{
			if ((x_sx/y_up) > tan(angle)) //potrei incontrare il bordo superiore, controllare il range 
			{
				float partial = y_up/cos(angle);
				if (partial >= remaining_range) return remaining_range;
				else
				{
					float x = x_sx - (partial * sin(angle));
					float partial_returned=0.1;
					switch (seg->next->type) 
					{
						case 3:	partial_returned = sensor_calc_str(seg->next, x, 0, angle, (remaining_range - partial));
								break;
						case 2:	partial_returned = sensor_calc_lft_rgt(seg->next, x, 0, angle, (remaining_range - partial));
								break;
						case 1:	partial_returned = sensor_calc_lft_rgt(seg->next, x, 0, angle, (remaining_range - partial));
								break;
					}
					return (partial_returned + partial);
				}
			}
			else //potrei incontrare il bordo sinistro, controllare il range
			{
				float temp_val = x_sx / cos((HALF_PI) - angle);
				if (temp_val >= remaining_range) return remaining_range;
				else return temp_val;
			}
		}
	else
		if (cos(angle) > 0 && sin(angle) < 0)
		{
			if ((x_dx/y_up) > -tan(angle)) //potrei incontrare il bordo superiore, controllare il range
			{
				float partial = y_up/cos(angle);
				if (partial >= remaining_range) return remaining_range;
				else
				{
					float x = x_sx + (partial * sin(-angle));
					float partial_returned=0.1;
					switch (seg->next->type)
					{
						case 3:	partial_returned = sensor_calc_str(seg->next, x, 0, angle, (remaining_range - partial));
								break;
						case 2:	partial_returned = sensor_calc_lft_rgt(seg->next, x, 0, angle, (remaining_range - partial));
								break;
						case 1:	partial_returned = sensor_calc_lft_rgt(seg->next, x, 0, angle, (remaining_range - partial));
								break;
					}
					return (partial_returned + partial);
				}
			}
			else //potrei incontrare il bordo destro, controllare il range
			{
				float temp_val = x_dx / cos((HALF_PI) + angle);
				if (temp_val >= remaining_range) return remaining_range;
				else return temp_val;
			}
		}
	else
		if (cos(angle) < 0 && sin(angle) >= 0)
		{
			if ((x_sx/y_dn) > tan(PI-angle)) //potrei incontrare il bordo inferiore, controllare il range
			{
				float partial = y_dn/-cos(angle);
				if (partial >= remaining_range) return remaining_range;
				else
				{
					float x = x_sx - (partial * sin(PI-angle));
					float partial_returned=0.1;
					switch (seg->prev->type) 
					{
						case 3:	partial_returned = sensor_calc_str(seg->prev, x, seg->prev->length, angle, (remaining_range - partial));
								break;
						case 2:	partial_returned = sensor_calc_lft_rgt(seg->prev, x, seg->prev->arc, angle, (remaining_range - partial));
								break;
						case 1:	partial_returned = sensor_calc_lft_rgt(seg->prev, x, seg->prev->arc, angle, (remaining_range - partial));
								break;
					}
					return (partial_returned + partial);
				}
			}
			else //potrei incontrare il bordo sinistro, controllare il range
			{
				float temp_val = x_sx / cos(angle - (HALF_PI));
				if (temp_val >= remaining_range) return remaining_range;
				else return temp_val;
			}
		}
	else
		if (cos(angle) < 0 && sin(angle) < 0)
		{
			if ((x_dx/y_dn) > tan(PI+angle)) //potrei incontrare il bordo inferiore, controllare il range
			{
				float partial = y_dn / cos(PI + angle);
				if (partial >= remaining_range) return remaining_range;
				else
				{
					float x = x_sx + partial * sin(PI + angle);
					float partial_returned=0.1;
					switch (seg->prev->type) 
					{
						case 3:	partial_returned = sensor_calc_str(seg->prev, x, seg->prev->length, angle, (remaining_range - partial));
								break;
						case 2:	partial_returned = sensor_calc_lft_rgt(seg->prev, x, seg->prev->arc, angle, (remaining_range - partial));
								break;
						case 1:	partial_returned = sensor_calc_lft_rgt(seg->prev, x, seg->prev->arc, angle, (remaining_range - partial));
								break;
					}
					return (partial_returned + partial);
				}
			}
			else //potrei incontrare il bordo destro, controllare il range
			{
				float temp_val = x_dx / cos(-angle - (HALF_PI));
				if (temp_val >= remaining_range) return remaining_range;
				else return temp_val;
			}
		}
	else return remaining_range;
}


float sensor_calc_lft_rgt(tTrackSeg *seg, float x_coord, float y_coord, float angle, float remaining_range) 
{
	float x_sx = x_coord;						//meters
	float x_dx = (seg->startWidth) - x_sx;		//meters
	float y_up = (seg->arc) - y_coord;			//radians
	float y_dn = y_coord;						//radians
	float radius_max, radius_min, radius_x, temp_angle;

	if (seg->type == 2)
	{
		radius_max = seg->radiusr;
		radius_min = seg->radiusl;
		radius_x = radius_min + x_sx;
		temp_angle = angle;
	}
	else
	{
		radius_max = seg->radiusl;
		radius_min = seg->radiusr;
		radius_x = radius_min + x_dx;
		temp_angle = -angle;
	}
	
	float dist_returned;
	float angle_returned;
	float x_returned;
	if (check_max_circle_intersect(temp_angle, radius_max, radius_x, y_dn, y_up, remaining_range, dist_returned)) return dist_returned;
	else if	(check_min_circle_intersect(temp_angle, radius_min, radius_x, y_dn, y_up, remaining_range, dist_returned)) return dist_returned;
	else if (check_up_border_intersect(temp_angle, radius_min, radius_x, y_up, remaining_range, dist_returned, angle_returned, x_returned))
	{
		float partial_returned=0.1;
//		printf ("\nRecursive Call for next segment\n");
		
		// Correction of outputs if we are currently in a right turn (Daniele Loiacono - 3/2009)
		if (seg->type == 1)
		{
			x_returned = seg->startWidth - x_returned;
			angle_returned = -angle_returned;
		}

		if (seg->next->type==3) 
		{
			partial_returned = sensor_calc_str(seg->next, x_returned, 0, angle_returned, (remaining_range - dist_returned));
		} 
		else
		{
			partial_returned = sensor_calc_lft_rgt(seg->next, x_returned, 0, angle_returned, (remaining_range - dist_returned));

		}
		return (partial_returned + dist_returned);
	}
	else if (check_down_border_intersect(temp_angle, radius_min, radius_x, y_dn, remaining_range, dist_returned, angle_returned, x_returned))
	{
		float partial_returned=0.1;
//		printf ("\nRecursive Call for prev segment\n");

		// Correction of outputs if we are currently in a right turn (Daniele Loiacono - 3/2009)
		if (seg->type == 1)
		{
			x_returned = seg->startWidth - x_returned;
			angle_returned = -angle_returned;
		}

		if (seg->prev->type==3) 
		{
			partial_returned = sensor_calc_str(seg->prev, x_returned, seg->prev->length, angle_returned, (remaining_range - dist_returned));
					
		}
		else
		{
			partial_returned = sensor_calc_lft_rgt(seg->prev, x_returned, seg->prev->arc, angle_returned, (remaining_range - dist_returned));
		}		
		return (partial_returned + dist_returned);
	}
	else return remaining_range;
}


bool check_max_circle_intersect(float angle, float radius_max, float radius_x, float y_dn, float y_up, float remaining_range, float &dist_returned)
{
	if (angle < 0)
	{
		if (angle >= -(HALF_PI))
		{
			float check = radius_x * cos(-angle);
			float beta = acos(check/radius_max);
			float gamma = beta + angle;
			if (gamma <= y_up) //il sensore potrebbe incontrare il cerchio max in questo segmento
			{
				float check_dist = radius_x * sin(-angle);
				float z = radius_max * sin(beta);
				float dist = z - check_dist;
				if (dist <= remaining_range)
				{ 
					dist_returned = dist;
//					printf ("\nMax Circle Intersection (-90 < angle < 0)\n");
					return true;
				}
				else return false;
			}
			else return false;
		}
		else // if (angle < -(HALF_PI))
		{
			float check = radius_x * cos(PI + angle);
			float beta = acos(check/radius_max);
			float gamma = beta - angle - PI;
			if (gamma <= y_dn) //il sensore potrebbe incontrare il cerchio max in questo segmento
			{
				float check_dist = radius_x * sin(PI + angle);
				float z = radius_max * sin(beta);
				float dist = z - check_dist;
				if (dist <= remaining_range)
				{ 
					dist_returned = dist;
//					printf ("\nMax Circle Intersection (-180 < angle < -90)\n");
					return true;
				}
				else return false;
			}
			else return false;
		}
	}
	
	else if (angle > 0)
	{
		if (angle <= (HALF_PI))
		{
			float check = radius_x * cos(angle);
			float beta = acos(check/radius_max);
			float gamma = beta + angle;
			if (gamma <= y_up) //il sensore potrebbe incontrare il cerchio max in questo segmento
			{
				float check_dist = radius_x * sin(angle);
				float z = radius_max * sin(beta);
				float dist = z + check_dist;
				if (dist <= remaining_range)
				{ 
					dist_returned = dist;
//					printf ("\nMax Circle Intersection (0 < angle < 90)\n");
					return true;
				}
				else return false;
			}
			else return false;
		}
		else // if (angle > (HALF_PI))
		{
			float check = radius_x * cos(PI + angle);
			float beta = acos(check/radius_max);
			float gamma = beta + angle + PI;
			if (gamma <= y_up) //il sensore potrebbe incontrare il cerchio max in questo segmento
			{
				float check_dist = radius_x * sin(PI + angle);
				float z = radius_max * sin(beta);
				float dist = z + check_dist;
				if (dist <= remaining_range)
				{ 
					dist_returned = dist;
//					printf ("\nMax Circle Intersection (90 < angle < 180)\n");
					return true;
				}
				else return false;
			}
			else return false;
		}
	}
	else return false;
}


bool check_min_circle_intersect(float angle, float radius_min, float radius_x, float y_dn, float y_up, float remaining_range, float &dist_returned)
{
	if (angle > 0 && angle <= (HALF_PI))
	{
		float check = radius_x * cos(angle);
		float beta = acos(check/radius_min);
		float gamma = angle - beta;
		if (gamma <= y_up) //il sensore potrebbe incontrare il cerchio min in questo segmento
		{
			float check_dist = radius_x * sin(angle);
			float z = radius_min * sin(beta);
			float dist = check_dist - z;
			if (dist <= remaining_range)
			{ 
				dist_returned = dist;
//				printf ("\nMin Circle Intersection (0 < angle < 90)\n");
				return true;
			}
			else return false;
		}
		else return false;
	}
	else if (angle > 0 && angle > (HALF_PI))
	{
		float check = radius_x * cos(PI - angle);
		float beta = acos(check/radius_min);
		float gamma = PI - beta - angle;
		if (gamma <= y_dn && gamma >= 0.0) //il sensore potrebbe incontrare il cerchio min in questo segmento
		{
			float check_dist = radius_x * sin(PI - angle);
			float z = radius_min * sin(beta);
			float dist = check_dist - z;
			if (dist <= remaining_range)
			{ 
				dist_returned = dist;
//				printf ("\nMin Circle Intersection (90 < angle < 180)\n");
				return true;
			}
			else return false;
		}
		else return false;
	}
	else return false;
}


bool check_up_border_intersect(float angle, float radius_min, float radius_x, float y_up, float remaining_range, float &dist_returned, float &angle_returned, float &x_returned)
{
	if (angle < (HALF_PI) && angle > -(HALF_PI))
	{
		if (angle >= y_up)
		{
			float check_dist = radius_x * sin(y_up);
			float dist = check_dist / cos(angle - y_up);
			if (dist < remaining_range) //il sensore incontra il bordo superiore di questo segmento
			{
				float z = dist * sin(angle - y_up);
				float check = radius_x * cos(y_up);
				dist_returned = dist;
				angle_returned = angle - y_up;
				x_returned =  check - z - radius_min;
//				printf ("\nUp Boarder Intersection\n");
				return true;
			}
			else return false;
		}
		else
		{
			float check_dist = radius_x * sin(y_up);
			float dist = check_dist / cos(y_up - angle);
			if (dist < remaining_range) //il sensore incontra il bordo superiore di questo segmento
			{
				float z = dist * sin(y_up - angle);
				float check = radius_x * cos(y_up);
				dist_returned = dist;
				angle_returned = - (y_up - angle);
				x_returned =  check + z - radius_min;
//				printf ("\nUp Boarder Intersection\n");
				return true;
			}
			else return false;
		}
	}
	else return false;
}


bool check_down_border_intersect(float angle, float radius_min, float radius_x, float y_dn, float remaining_range, float &dist_returned, float &angle_returned, float &x_returned)
{
	if (angle > (HALF_PI) || angle < -(HALF_PI))
	{
		if (angle > 0 && angle <= (PI - y_dn))
		{
			float check_dist = radius_x * sin(y_dn);
			float dist = check_dist / cos(PI - angle - y_dn);
			if (dist < remaining_range) //il sensore incontra il bordo inferiore di questo segmento
			{
				float z = dist * sin(PI - angle - y_dn);
				float check = radius_x * cos(y_dn);
				dist_returned = dist;
				angle_returned = angle + y_dn;
				x_returned =  check - z - radius_min;
//				printf ("\nDown Boarder Intersection\n");
				return true;
			}
			else return false;
		}
		else if (angle > 0 && angle > (PI - y_dn))
		{
			float check_dist = radius_x * sin(y_dn);
			float dist = check_dist / cos(y_dn + angle - PI);
			if (dist < remaining_range) //il sensore incontra il bordo inferiore di questo segmento
			{
				float z = dist * sin(y_dn + angle - PI);
				float check = radius_x * cos(y_dn);
				dist_returned = dist;
				angle_returned = -((2 * PI) - angle - y_dn);
				x_returned =  check + z - radius_min;
//				printf ("\nDown Boarder Intersection\n");
				return true;
			}
			else return false;
		}
		else
		{
			float check_dist = radius_x * sin(y_dn);
			float dist = check_dist / cos(y_dn + angle + PI);
			if (dist < remaining_range) //il sensore incontra il bordo inferiore di questo segmento
			{
				float z = dist * sin(y_dn + angle + PI);
				float check = radius_x * cos(y_dn);
				dist_returned = dist;
				angle_returned = -(-angle - y_dn);
				x_returned =  check + z - radius_min;
//				printf ("\nDown Boarder Intersection\n");
				return true;
			}
			else return false;
		}
	}
	else return false;
}


/* Drive during race. */
static void  
drive(int index, tCarElt* car, tSituation *s) 
{	
	
    //position information(trk)
    float angle;       //angle
    float trkdist;     //dist-to-trkstart
    float disttomid;    //dist-to-middle
    float disttoleft;    //dist-to-left
    float disttoright;    //dist-to-right

    float posInfo[4];

    //car's information
    float fuel;        //car-fuel    
    float mass;        //car-mass
    float speedX;      //car->speedX
    float speedY;      //car->speedY
    float speedZ;      //car->speedZ

    float carInfo[5];

    /*segment information
        float trkType;    //L,R,Straight
        float trkRadius;    //trk-radius: 1/r
        float trkSurface;   //\mu maybe is unimportant
        float trkLength;    //trk->length
    */
    float segInfo[4][4];

    float sensor[19];

    float wheel[4];     //spinwheel

    float rpm;

    SvrInfo svrInfo;

    memset(&car->ctrl, 0, sizeof(tCarCtrl));
    const float SC = 1.0;

    rpm = car->_enginerpm*10;

    for(int i = 0; i < 4; i++){
        wheel[i] = car->priv.wheel[i].spinVel; 
    }

    for (int i = 0; i < 19; i++){
        float sensor_angle = -90 + 10*i;
        float relative_sensor_angle = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw + sensor_angle;
        NORM_PI_PI(relative_sensor_angle);
		
        switch (car->_trkPos.seg->type) {
		    case 3: sensor[i] = sensor_calc_str(car->_trkPos.seg, car->_trkPos.toLeft, car->_trkPos.toStart, -relative_sensor_angle, sensor_range);
				    break;
		    case 2:	sensor[i] = sensor_calc_lft_rgt(car->_trkPos.seg, car->_trkPos.toLeft, car->_trkPos.toStart, -relative_sensor_angle, sensor_range);
				    break;
		    case 1:	sensor[i] = sensor_calc_lft_rgt(car->_trkPos.seg, car->_trkPos.toLeft, car->_trkPos.toStart, -relative_sensor_angle, sensor_range);
				    break;
	    }
    }

    /*
    if (isStuck(car)){
        angle = -RtTrackSideTgAngleL(&(car->_trkPos)) + car->_yaw;
        NORM_PI_PI(angle); // put the angle back in the range from -PI to PI
        car->ctrl.steer = angle / car->_steerLock;
        car->ctrl.gear = -1; // reverse gear
        car->ctrl.accelCmd = 0.5; // 30% accelerator pedal
        car->ctrl.brakeCmd = 0.0; // no brakes
        return;
    }*/

    tTrackSeg *segptr = car->_trkPos.seg;  //get segment
    tTrackSeg *segptr_last = NULL;  //last segment

    //init-information    
    if (lock < 1){
        lock = lock + 1;
        start_id = car->_trkPos.seg->id;
//      segptr = car->_trkPos.seg;
        if (segptr->type == TR_STR){
            total_length = segptr->length;
        }
        else{
            total_length = segptr->arc*segptr->radius;
        }
        segptr = segptr->next;
//        ofstream fout("/home/qby/Documents/torcs-1.3.8-test1/src/drivers/ljf/debug.txt",ios::app);
        while (segptr != car->_trkPos.seg){
            float tmp_l = 0;
            float tmp_r = 0;
            if (segptr->type == TR_STR){
                tmp_l = segptr->length;
                tmp_r = 0;
                total_length += tmp_l;
            }
            else{
                tmp_r = segptr->radius;
                tmp_l = segptr->arc*segptr->radius;
                total_length += tmp_l;
            }
//            fout << segptr->type << ' ' << tmp_l << ' ' << tmp_r  << ' ' << segptr->id << endl;
            segptr = segptr->next;
        }
//        fout << "total_length: " << total_length << endl;
//        fout.close();
    }
    segptr = car->_trkPos.seg;

    //--------------------position information(trk)-----------------------
    //angle
    angle = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw;
    NORM_PI_PI(angle); // put the angle back in the range from -PI to PI
    posInfo[0] = angle;

    //dist-to-start
    if (car->_trkPos.seg->type == TR_STR) {
	    trkdist = car->_trkPos.toStart;
    }
    else{
        trkdist =  car->_trkPos.toStart*segptr->radius;
    }
    //posInfo[1] = trkdist;

    //dist-to-middle
    disttomid = car->_trkPos.toMiddle;
    posInfo[1]= disttomid;

    //dist-to-left and right
    disttoleft = car->_trkPos.toLeft;
    disttoright = car->_trkPos.toRight;

    posInfo[2]= disttoleft;
    posInfo[3]= disttoright;
    
    //------------------ car's information----------------------------------
    //car-fuel
    fuel = car->_fuel;

    //car-mass
    float CARMASS = GfParmGetNum(car->_carHandle, SECT_CAR, PRM_MASS, NULL, 1000.0);
    mass = CARMASS + car->_fuel;

    //car->speedX,Y,Z
    speedX = car->_speed_x;
    speedY = car->_speed_y;
    speedZ = car->_speed_z;

    carInfo[0] = fuel;
    carInfo[1] = mass;
    carInfo[2] = speedX;
    carInfo[3] = speedY;
    carInfo[4] = speedZ;

   
    //--------------------segment information------------------------------
    int i = 0;
    while (i < 4){
        /*trkType = segment type:(need to be decoded by server)
			 - TR_RGT
			 - TR_LFT
			 - TR_STR
			*/
        if (segptr_last == NULL){
            if (segptr->type == TR_LFT){
                segInfo[i][0] = 1;
            }
            else if (segptr->type == TR_RGT){
                segInfo[i][0] = 2;
            }
            else
                segInfo[i][0] = 3;
            
            //radius
            if (segptr->type == TR_STR){
                segInfo[i][1] = 0;
            }
            else
                segInfo[i][1] = 1.0/segptr->radius;

            //Surface
            segInfo[i][2] = segptr->surface->kFriction;

            //length
            if (segptr->type == TR_STR){
                segInfo[i][3] = segptr->length;
            }
            else{
                segInfo[i][3] = segptr->arc*segptr->radius;
            }
        }
        else{
            if (segptr_last->type == segptr->type && segptr->surface->kFriction == segptr_last->surface->kFriction){
                if (segptr->type == TR_STR){
                    segInfo[i][3] = segInfo[i][3] + segptr->length;
                }
                else{
                    segInfo[i][3] = segInfo[i][3] + segptr->arc*segptr->radius;
                }
            }
            else {
                i = i + 1;
                if (segptr->type == TR_LFT){
                    segInfo[i][0] = 1;
                }
                else if (segptr->type == TR_RGT){
                    segInfo[i][0] = 2;
                }
                else
                    segInfo[i][0] = 3;
                
                //radius
                if (segptr->type == TR_STR){
                    segInfo[i][1] = 0;
                }
                else
                    segInfo[i][1] = 1.0/segptr->radius;

                //Surface
                segInfo[i][2] = segptr->surface->kFriction;

                //length
                if (segptr->type == TR_STR){
                    segInfo[i][3] = segptr->length;
                }
                else{
                    segInfo[i][3] = segptr->arc*segptr->radius;
                }
            }
        }
        segptr_last = segptr;
        segptr = segptr->next;
    }
    //dist-to-end
    segInfo[0][3] = segInfo[0][3] - trkdist; 
    
    current_length = car->race.distFromStartLine;


    //start training stop at first 1 seconds
    if (count < 100){
        car->ctrl.steer = 0;
        car->ctrl.gear = 0; // first gear
        car->ctrl.accelCmd = 0.0; // 30% accelerator pedal
        car->ctrl.brakeCmd = 0.0; // no brakes
        count++;
        return;
    }

    if (train){
        env->send(sensor, wheel, posInfo, carInfo, (float **)segInfo, 19, 4, 4, 5, 4, 4, current_length, total_length, rpm);
        svrInfo = env->get();

        //don't know why but have to add this line
        ofstream fout("./debug.txt",ios::app);
        fout << "end race" << endl;
        fout.close();
        float t = svrInfo.time;
        car->ctrl.steer = svrInfo.steer;
        car->ctrl.accelCmd = svrInfo.accelerate;
//         car->ctrl.brakeCmd = svrInfo.brake;
        car->ctrl.brakeCmd = getBrake(car);
        car->ctrl.gear = getGear(car);
        last_svrInfo = svrInfo;
        return;
    }

    /*  
     * add the driving code here to modify the 
     * car->_steerCmd 
     * car->_accelCmd 
     * car->_brakeCmd 
     * car->_gearCmd 
     * car->_clutchCmd 
     */ 
}

/* End of the current race */
static void
endrace(int index, tCarElt *car, tSituation *s)
{
    // env->send();
}

