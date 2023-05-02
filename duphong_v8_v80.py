import cv2
import time
def Controller(edges, PID, current_speed, current_angle, check_err, xmax, xmin, conf, cls, right, left, straight, S, notleft, notright, lefttest, righttest):
    angle_check=0
    rightmin=0
    leftmin=0
    cuagap=0
    od = (xmax + xmin)/2
    """detect duong thang"""
    straightarr = []
    lineStraight = edges[4,:]
    for x,y in enumerate(lineStraight):
        if y==255:
            straightarr.append(x)
    try: 
        straightmax=max(straightarr)
        straight_detect = 1
    except Exception as er:
        straight_detect = 0
        print('ko co duong thang')
        pass
    """"""
    """detect trai"""
    leftarr = []
    lineLeft = edges[:,13]
    for x,y in enumerate(lineLeft):
        if y==255:
            leftarr.append(x)
    try:
        leftmin=min(leftarr)
        if (leftmin>25):
            left_detect = 0
        else: left_detect = 1
        # print('left', left_detect)
    except Exception as er:
        left_detect = 5
        pass
    """"""
    """detect phai"""
    rightarr = []
    lineRight = edges[:,147]
    for x,y in enumerate(lineRight):
        if y==255:
            rightarr.append(x)
    try:
        rightmin=min(rightarr)
        if (rightmin>25):
            right_detect = 0
        else: right_detect = 1
        # print('right', right_detect)
    except Exception as er:
        right_detect = 5
        pass
    """DETECT LANE"""
    line = 30
    arr = []
    lineRow = edges[line,:]
    for x,y in enumerate(lineRow):
        if y==255:
            arr.append(x)
    if not arr:
        print('000000000000000000000000000000000000000000')
        line_check = 45
        arr_check = []
        lineRow_check = edges[line_check,:]
        for x,y in enumerate(lineRow_check):
            if y==255:
                arr_check.append(x)
        arrmax=max(arr_check)
        arrmin=min(arr_check)
        center = int((arrmax + arrmin)/2)  
        error = int(edges.shape[1]/2) - center
        angle_check = -PID(error, 0.35, 0.000, 0.065)#0.3
        if (angle_check>0):
            angle_check = angle_check + 1
        if (angle_check<0):
            angle_check = angle_check - 1
        # if (right>1):
        #     angle_check = 25
        #     print('angle==========================25')
        return angle_check, 100, check_err, right, left, straight, notleft, notright, lefttest, righttest
    arrmax=max(arr)
    arrmin=min(arr)
    arrmax_turn=max(arr)
    arrmin_turn=min(arr)
    # line_turn = 40
    # arr_turn = []
    # lineTurn = edges[line_turn,:]
    # for x,y in enumerate(lineTurn):
    #     if y==255:
    #         arr_turn.append(x)
    # arrmax_turn=max(arr_turn)
    # arrmin_turn=min(arr_turn)
    # print(arrmax, arrmin, xmax, xmin, arrmax_turn, arrmin_turn)
    """ CAR """
    if (conf>0.7):
        if (cls==6 and S>6500):
            print('---------------------------------------------------------------', float(current_speed))
            if (od<320):
                arrmin = 60
            else: 
                arrmax = 100

    # if (check_err==1):
    #     t = time.time()
    #     while((time.time()-t)<0.135):
    #         angle = 25
    #         print('++++++++++++++++++++++++++++++++++++++')
    #     check_err=0

    # if (check_err==2):
    #     t = time.time()
    #     while((time.time()-t)<0.135):
    #         angle = -25
    #         print('-------------------------------------------')
    #     check_err=0
    
    # if (arrmax>150 and arrmin>120):
    #     angle = 25
    #     check_err=1
    #     print('1111111111111111111111111111111')
    # if (arrmax<30 and arrmin<10):
    #     angle = -25
    #     check_err=2
    #     print('2222222222222222222222222222222')

    center = int((arrmax + arrmin)/2)  
    error = int(edges.shape[1]/2) - center
    # print(error)
    angle = -PID(error, 0.35, 0.000, 0.065)#0.3
    if (angle>11 and left==0 and right==0 and conf==0 and float(current_speed)>62):
        angle=25
        cuagap=1
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ +25')
    if(angle<-11 and left==0 and right==0 and conf==0 and float(current_speed)>62):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ -25')
        angle=-25
        cuagap=1
    if (cls==6 and S>2500 and conf>0.8):
        set_speed_OD = 77
    elif (cls==2 and S>2500 and conf>0.8):
        set_speed_OD = 77
    if (conf>0.6 and cls!=7 and S>2800):
        set_speed_OD=77
    if (right==1 or left==1):
        set_speed_OD = 75
    if (righttest==1 or lefttest==1):
        set_speed_OD = 72
    # elif (notleft==1 or notright==1):
    #     set_speed_OD=70
    else: set_speed_OD=0
    """TIME DELAY TURN RIGHT"""
    # CUA 0.7S
    if (right==3):
        print('speed:......................................................................', float(current_speed))  
        right=0
        t1 = time.time()
        while ((time.time()-t1)<1.1):
            None
    # DI THANG 0.1s
    if (right==2):
        print('speed:......................................................................', float(current_speed))  
        right=3
        t1 = time.time()
        # tdelay_right=0.01
        # if(float(current_speed)<65):
        #     tdelay_right=0.368-0.005*(float(current_speed)) 
        tdelay_right=0.36-0.005*(float(current_speed)) + 0.0007*(70-int(current_speed))*(70-float(current_speed))
        if (float(current_speed)>69.5):
            tdelay_right = tdelay_right-0.0055
        if (float(current_speed)<69.5 and float(current_speed)>66):
            tdelay_right = tdelay_right+0.0025
        if (float(current_speed)<66 and float(current_speed)>59.5):
            tdelay_right = tdelay_right+0.009
        if (float(current_speed)<59.5 and float(current_speed)>50):
            tdelay_right = tdelay_right-0.015
        if (float(current_speed)<50 and float(current_speed)>40):
            tdelay_right = tdelay_right-0.025
        if (float(current_speed)<40):
            tdelay_right = tdelay_right-0.035
        if (righttest==1 and float(current_speed)>70):
            tdelay_right = 0
            print('                                                      ')
            print('da tru delayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
            print('                                                      ')
        while ((time.time()-t1)<tdelay_right):
            None
        angle = 25
        speed = 140
        if (righttest==1 and float(current_speed)>70):
            angle = 25
            speed = 140
        righttest = 0
    """TIME DELAY TURN LEFT"""
    # CUA 0.7S
    if (left==3):
        print('speed:..................................................left....................', float(current_speed))  
        left=0
        t2 = time.time()
        while ((time.time()-t2)<1.2):
            None
    # DI THANG 0.1s
    if (left==2):
        print('speed:.....................................................left.................', float(current_speed))  
        left=3
        t2 = time.time()
        # tdelay_left=0.01
        # if (float(current_speed)<65): #-0.021x+1.437
        #     tdelay_left=0.368-0.005*(float(current_speed))
        # else: tdelay_left=0.36-0.005*(float(current_speed))
        tdelay_left=0.36-0.005*(float(current_speed)) + 0.0007*(70-int(current_speed))*(70-int(current_speed))
        if (float(current_speed)>69.5):
            tdelay_left = tdelay_left-0.005
        elif (float(current_speed)<69.5 and float(current_speed)>66):
            tdelay_left = tdelay_left+0.0025
        elif (float(current_speed)<66 and float(current_speed)>59.5):
            tdelay_left = tdelay_left+0.009
        elif (float(current_speed)<59.5 and float(current_speed)>50):
            tdelay_left = tdelay_left-0.015
        elif (float(current_speed)<50 and float(current_speed)>40):
            tdelay_left = tdelay_left-0.025
        elif (float(current_speed)<40):
            tdelay_left = tdelay_left-0.035
        if (lefttest==1 and float(current_speed)>70):
            tdelay_left = 0
            print('                                                      ')
            print('da tru delayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
            print('                                                      ')
        while ((time.time()-t2)<tdelay_left):
            None
        angle = -25
        speed = 140
        if (lefttest==1 and float(current_speed)>70):
            angle = -25
            speed = 140
        lefttest = 0
    """TIME DELAY STRAIGHT"""
    if (straight==2):
        print('speed:.....................................................straight.................', float(current_speed), arrmax, arrmin)  
        t3 = time.time()
        while ((time.time()-t3)<1.2):
            None
        straight=0
    ############################################################################################################################################
    """STRAIGHT"""
    if (conf > 0.8 and S>4000): # can sua lai
        if (cls==2):
            straight=1
    if (straight==1 and S==0):
        angle = 0
        speed = 150
        straight=2
        notleft=0
        notright=0
    # if (straight==1):
    #     if (left_detect==0 and right_detect==1):
    #         if (arrmax>145):
    #             angle = 0
    #             speed = 150
    #             straight=2
    #     elif (left_detect==1 and right_detect==0):
    #         if (arrmin<15):
    #             angle = 0
    #             speed = 150
    #             straight=2
    #     else: straight=0
    """TURN RIGHT"""
    if (conf > 0.8 and S>2900): # can sua lai
        if (cls==1):
            right=1
    if (right==1):  
        if (arrmax_turn>152):
            angle = 0
            speed = 150
            right = 2
            notleft=0
        else: 
            if (left_detect==1):
                if(notleft==1):
                    angle = 0.055
                    print('                                                      ')
                    print('                                 righttttttttttttttttttttttttttttttttttttttttttttttt')
                    print('                                                      ')
                else: angle = 0.03
    """TURN LEFT"""
    if (conf > 0.8 and S>2900): # can sua lai
        if (cls==8):
            left=1
    if (left==1): 
        if (arrmin_turn<8):
            angle = 0
            speed = 150
            left = 2 
            notright=0
        else: 
            if (right_detect==1):
                if (notright==1):
                    print('                                                      ')
                    print('                                 leftttttttttttttttttttttttttttttttttttttttttttttttttttt')
                    print('                                                      ')
                    angle = -0.055
                else: angle = -0.03
    """NO TURN RIGHT"""
    if (conf > 0.8 and S>2900): # can sua lai
        if (cls==4):
            notright=1
    if (notright==1):
        if (left_detect==1):
            left=1
            lefttest = 1
        elif (straight_detect==1):
            straight=1
    """NO TURN LEFT"""
    if (conf > 0.8 and S>2900): # can sua lai
        if (cls==3):
            notleft=1
    if (notleft==1):
        if (right_detect==1):
            right=1
            righttest = 1
        elif (straight_detect==1):
            straight=1
    #err duong xe dang ben phai va can sang trai nen de am de nguoc lai, goc duong la dg bi sang phai
    if (right<2 and left<2 and straight<2):
        if (set_speed_OD>0):
            if (set_speed_OD>0):
                set_speed =  set_speed_OD
            if (set_speed>0 and float(current_speed)>set_speed):
                er = float(current_speed) - set_speed 
                if (float(current_speed)>75.5):
                    speed = -2
                # speed = -er*0.8 + 5
                else: speed = 0
                if (cls==6 and S>2000):
                    speed = 10
            else: 
                if(abs(float(current_speed) - set_speed)<4):
                    speed = 130
                else: speed = 150
        else:
            if (abs(angle)<4):
                set_speed=80
            else: set_speed = 77 - abs(error)/6
            if (float(current_speed)<set_speed):
                speed=150
            else: 
                if (float(current_speed)>81):
                    speed = 0
                else: 
                    if (cuagap==1):
                        speed = -2.4*abs((error)) + 150
                    elif (righttest==1 or lefttest==1):
                        speed = -2.1*abs((error)) + 150
                    elif (float(current_speed)>70):
                        if (cls>0 and S>2800):
                            speed = -1.7*abs((error)) + 150
                        else: speed = -1.5*abs((error)) + 150
                    elif (float(current_speed)>65):
                        speed = -1.3*abs((error)) + 150
                    elif (float(current_speed)<60):
                        speed = 150
                    else: speed = -1*abs((error)) + 150
    print(arrmax, arrmin, left_detect, right_detect)
    # cv2.circle(edges,(arrmin,line),5,(0,0,0),3)
    # cv2.circle(edges,(arrmax,line),5,(0,0,0),3)
    # # cv2.circle(edges,(arrmin_turn,line_turn),5,(0,0,0),3)
    # # cv2.circle(edges,(arrmax_turn,line_turn),5,(0,0,0),3)
    # # cv2.line(edges,(center,line),(int(edges.shape[1]/2),edges.shape[0]),(0,0,0),3)
    # cv2.imshow("IMG", edges)
    # key = cv2.waitKey(1)
    return angle, speed, check_err, right, left, straight, notleft, notright, lefttest, righttest
    
    
