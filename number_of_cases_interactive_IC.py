# Calculate the number of cases with a decreasing R-number, 2 different variants and vaccination
# For information only. Provided "as-is" etc.

# https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py
# https://github.com/rcsmit/COVIDcases/blob/main/number_of_cases_interactive.py

# Sorry for all the commented out code, maybe I will combine the old and new version(s) later

# Import our modules that we are using
import math
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
_lock = RendererAgg.lock
from scipy.integrate import odeint
import pandas as pd


# VARIABLES
# startdate in mm/dd/yyyy
# variable1 = old variant
# variable2 = new variant
# variable 12 = old + new
# variablex = with vaccination

DATE_FORMAT = "%m/%d/%Y"
b = datetime.today().strftime('%m/%d/%Y')

#values 01/13/2021, according to https://www.bddataplan.nl/corona/
st.sidebar.title('Key Parameters')
numberofpositivetests = st.sidebar.number_input('Total number of positive tests',None,None,4600)
newrnumber = st.sidebar.slider('New R-number 1', 0.0, 3.0, 1.1)
ic_dayzero = st.sidebar.number_input('Aantal IC dag 0',None,None,500)


st.sidebar.markdown("<br><br><br><hr>", unsafe_allow_html=True)
st.sidebar.title('Parameters')
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',"03/10/2021")

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
except:
    st.error("Please make sure that the date is in format mm/dd/yyyy")
    st.stop()

NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 720, 180)
global numberofdays_
numberofdays_ = NUMBEROFDAYS


Rnew_2_ = st.sidebar.slider('R-number', 0.1, 6.0, 1.00)
#correction = st.sidebar.slider('Correction factor', 0.0, 2.0, 1.00)
correction = 1


# https://www.medrxiv.org/content/10.1101/2020.09.13.20193896v1.full.pdf / page 4
showcummulative =  True



showimmunization =  True





turning = st.sidebar.checkbox("Turning point", True)
turning_2 = False

if turning:
   
    
    turningpointdate = st.sidebar.text_input('Turning point date (mm/dd/yyyy) 1', "03/15/2021")
    turningdays = st.sidebar.slider('Number of days needed to reach new R values 1', 1, 90, 3)
    try:
        starty = dt.datetime.strptime(turningpointdate,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()

    d1 = datetime.strptime(a, '%m/%d/%Y')
    d2 = datetime.strptime(turningpointdate,'%m/%d/%Y')
    if d2<d1:
        st.error("Turning point cannot be before startdate")
        st.stop()
    turningpoint =  abs((d2 - d1).days)

    #Rnew3 = st.sidebar.slider('R-number target British variant', 0.1, 2.0, 0.8)
    newrnumber2 = st.sidebar.slider('New R-number 2', 0.0, 3.0, 0.9)
    #turning_2point = st.sidebar.slider('Startday turning_2', 1, 365, 30)
    turning_2pointdate = st.sidebar.text_input('Turning point date 2 (mm/dd/yyyy)', "02/02/2025")
    small = ("Set date in 2030 if turning point if not needed")
    st.sidebar.write(small)
    turning_2days = st.sidebar.slider('Number of days needed to reach new R values 2', 1, 90, 3, key="test")
    try:
        starty = dt.datetime.strptime(turning_2pointdate,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()


    d3 = datetime.strptime(turning_2pointdate,'%m/%d/%Y')
    if d3<d2:
        st.error("Second Turning point cannot be before First turning point")
        st.stop()
    turning_2point =  abs((d3 - d1).days)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.write('IC')

ic_days_stay = st.sidebar.number_input('Aantal dagen op IC',None,None,13)
from_test_to_ic =  st.sidebar.number_input('Aantal dagen test -> IC',None,None,5)
percentage_test_ic = st.sidebar.number_input('Percentage test/IC',None,None,0.7)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

show_hospital = False # st.sidebar.checkbox("Hospital", False)
if show_hospital:
    st.sidebar.write('HOSPITAL')
    hospital_days_stay = st.sidebar.number_input('Aantal dagen in ziekenhuis',None,None,21)
    from_test_to_hospital = st.sidebar.number_input('Aantal dagen test -> ziekenhuis',None,None,5)
    percentage_test_hospital = st.sidebar.number_input('Percentage test/ziekenhuis',None,None,4)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
else:
    hospital_dayzero = 1447
    hospital_days_stay = 21
    from_test_to_hospital = 5
    percentage_test_hospital = 4
    


st.sidebar.write ("No need to touch this")

Tg = st.sidebar.slider('Generation time', 2.0, 11.0, 4.0)
global Tg_
Tg_=Tg

if showcummulative or showSIR or showimmunization:
    totalimmunedayzero_ = (st.sidebar.text_input('Total immune persons day zero', 3_600_000))
    totalpopulation_ = (st.sidebar.text_input('Total population', 17_500_000))
    
    testimmunefactor = st.sidebar.slider('Test/immunityfactor', 0.0, 5.0, 2.5)
    try:
        totalimmunedayzero = int(totalimmunedayzero_)
    except:
        st.error("Please enter a number for the number of immune people on day zero")
        st.stop()
    
    try:
        totalpopulation = int(totalpopulation_)
    except:
        st.error("Please enter a number for the number of population")
        st.stop()

if showcummulative or showSIR:
    numberofcasesdayz = (st.sidebar.text_input('Number infected persons on day zero', 130000))

    try:
        numberofcasesdayzero = int(numberofcasesdayz)
    except:
        st.error("Please enter a number for the number of active cases on day zero")
        st.stop()
lambdaa = st.sidebar.slider('Lambda / heterogeneity', 1.0, 6.0, 1.0)
averagedayssick = (st.sidebar.slider('Average days infectious', 1, 30, 20))

Rnew_1_ = st.sidebar.slider('R-number first variant', 0.1, 10.0, 0.84)
Rnew1_= round(Rnew_1_ * correction,2)
Rnew2_= round(Rnew_2_ * correction,2)
percentagenewversion = (st.sidebar.slider('Percentage second variant at start', 0.0, 100.0, 100.0)/100)
vaccination = st.sidebar.checkbox("Vaccination")

if vaccination:
    VACTIME = st.sidebar.slider('Number of days needed for vaccination', 1, 730, 365)

# I wanted to link the classical SIR model of Kermack & McKendrik but the R_0 in that 
# model isnt the same as the R_0 = beta / gamma from that model.
# See https://www.reddit.com/r/epidemiology/comments/lfk83s/real_r0_at_the_start_not_the_same_as_given_r0/


numberofpositivetests1 = numberofpositivetests*(1-percentagenewversion)
numberofpositivetests2 = numberofpositivetests*(percentagenewversion)
numberofpositivetests12 = numberofpositivetests

# Some manipulation of the x-values (the dates)
then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1))
# x = dagnummer gerekend vanaf 1 januari 1970 (?)
# y = aantal gevallen
# z = dagnummer van 1 tot NUMBEROFDAYS
z  = np.array(range(NUMBEROFDAYS))

a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
b_ = dt.datetime.strptime(b,'%m/%d/%Y').date()
datediff = ( abs((a_ - b_).days))

# TODO:  Transform this in a multi dimensional list

positivetests1 = []
positivetests2 = []
positivetests12 = []
positivetestsper100k = []
cummulative1 = []
cummulative2 = []
cummulative12 = []
#walkingcummulative = []
ratio=[]
walkingR=[]
actualR=[]
totalimmune=[]
hospital = []
infected = []
ic_opnames  = []
ic_cumm = []
hospital_cumm = []
#if vaccination:
ry1x = []
ry2x = []

terugk= []
ry2 = Rnew2_
terugk.append(numberofpositivetests)

 # prevent an [divide by zero]-error

if ry2 == 1:
    ry2 = 1.000001
if ry2 <= 0:
    ry2 = 0.000001
for t in range (1, numberofdays_):

    thalf2 = Tg * math.log(0.5) / math.log(ry2)
    pt2 = (terugk[t-1] * (0.5**(-1/thalf2)))

    terugk.append((pt2))
#print (terugk)
pss_ = 0
for pss in range (0,30):
   # st.write(f"{str(terugk[pss])} - {str(terugk[pss]*0.7/100)} ")
    pss_ += terugk[pss]
#st.write ("--"+ str(pss_) )
blabla2 = (pss_ *0.7/100)

suspectible =[]
recovered = []
if showcummulative or showSIR or showimmunization:
    suspectible.append(totalpopulation -totalimmunedayzero)
    recovered.append(totalimmunedayzero )
if turning == False:
    #label1= 'First variant (R='+ str(Rnew1_) + ')'
    label2= '(R='+ str(Rnew2_) + ')'
else:
    #label1= 'First variant'
    label2= '_'
blabla=0
# START CALCULATING --------------------------------------------------------------------
positivetests1.append (numberofpositivetests1)
positivetests2.append (numberofpositivetests2)
positivetests12.append (numberofpositivetests12)
positivetestsper100k.append ((numberofpositivetests12/25))

if showcummulative:
    cummulative1.append(numberofcasesdayzero*(1-percentagenewversion))
    cummulative2.append(numberofcasesdayzero*(percentagenewversion))
    cummulative12.append(numberofcasesdayzero)
    infected.append(numberofcasesdayzero)
ratio.append(percentagenewversion*100 )
#walkingcummulative.append(1)
#if vaccination:
ry1x.append(Rnew1_)
ry2x.append(Rnew2_)
immeratio_=[]
immeratio_.append(1)

hospital.append(None)
ic_opnames.append(terugk[(-1*from_test_to_ic+1)] * percentage_test_ic / 100)
ic_cumm.append(ic_dayzero)
hospital_cumm.append(hospital_dayzero)
walkingR.append((Rnew1_**(1-percentagenewversion))*(Rnew2_**(percentagenewversion)))
if showimmunization:
    totalimmune.append(totalimmunedayzero)
#ry1 = 5
#ry1__ = 5
ry1__ = 999
fractionlist = []
lijstje = []
erbij_=[]
eraf_=[]
erbij_.append(ic_dayzero)
eraf_.append(0)
for t in range(0, NUMBEROFDAYS):
    lijstje.append(0)
#print (lijstje)




for t in range(1, NUMBEROFDAYS):
    erbij = 0
    eraf = 0   
    delta_hospital = 0
    delta_ic = 0 
    if not turning:
        if showimmunization:
            immeratio = (1-( (totalimmune[t-1]-totalimmune[0])/(totalpopulation-totalimmune[0])))
            ry1_ = ry1x[0]*(immeratio**lambdaa)
            ry2_ = ry2x[0]*(immeratio**lambdaa)
            immeratio_.append(immeratio)
            
            r1 = ry2_
            ry2__ = ry2_
        else:
            ry1_ = ry1x[0]
            ry2_ = ry2x[0]
            pass

    if turning :
        r1 = ry2x[0]
        r2 = newrnumber 
        tp = turningpoint
        tptd = turningpoint + turningdays
        ry2__ = ry2x[0]

        tp2 = turning_2point
        tptd2 = turning_2point + turning_2days
        
        r3 = newrnumber2 
        if t<=tp:
            immeratio = (1-( (totalimmune[t-1]-totalimmune[0])/(totalpopulation-totalimmune[0])))
            ry2__ = ry2x[0]*immeratio
            pass
        if ((t>tp) and t<=(tptd)):
            if turningdays==0:   
                ry2__ = r2
            else:
                fraction =  (((t-tp)/turningdays)) 
                                 
                ry2__ = ry2x[tp] + ((r2 -ry2x[tp] )*fraction)
                fractionlist.append(fraction)
        
        if (t>tptd) and t<=tp2:
            if totalimmune[tptd] > totalpopulation  or totalpopulation == totalimmune[tptd]:
                    immeratio = 0.0001
            else:
                immeratio = (1-( (totalimmune[t-1]-totalimmune[tptd])/(totalpopulation-totalimmune[tptd])))
                ry2__ = ry2x[tptd]*(immeratio**lambdaa)

        if t> tp2 and t<= tptd2:
            if turning_2days == 0 :
                ry2__ = r3
            else:
                fraction2 =  (((t-tp2)/turning_2days)) 
                ry2__ = ry2x[tp2] + ((r3 -ry2x[tp2] )*fraction2)
                
        if t>tptd2: 
                if totalimmune[tptd2] > totalpopulation  or totalpopulation == totalimmune[tptd2]:
                    immeratio = 0.0001
                else:
                    immeratio = (1-( (totalimmune[t-1]-totalimmune[tptd2])/(totalpopulation-totalimmune[tptd2])))
                ry2__ = ry2x[tptd2]*(immeratio**lambdaa)
        immeratio_.append(immeratio)
        
      
    
    if vaccination:
        if t>7:
            if t<(VACTIME+7) :
         
                ry2 = ry2__ * ((1-((t-7)/(VACTIME))))
            else:
                # vaccination is done, everybody is immune
               
                ry2 = ry2__ * 0.0000001
        else:
            # it takes 7 days before vaccination works
            ry1 = ry1__
            ry2 = ry2__
    else:
        ry1 = ry1__
        ry2 = ry2__

    # prevent an [divide by zero]-error
    if ry1 == 1:    
        ry1 = 1.000001
    if ry2 == 1:
        ry2 = 1.000001
    if ry1 <= 0:
        ry1 = 0.000001
    if ry2 <= 0:
        ry2 = 0.000001

    thalf1 = Tg * math.log(0.5) / math.log(ry1)
    thalf2 = Tg * math.log(0.5) / math.log(ry2)
    
    pt1 = (positivetests1[t-1] * (0.5**(1/thalf1)))
    pt2 = (positivetests2[t-1] * (0.5**(1/thalf2)))
    positivetests1.append(pt1)
    positivetests2.append(pt2)

    # This formula works also and gives same results
    # https://twitter.com/hk_nien/status/1350953807933558792
    # positivetests1a.append(positivetests1a[t-1] * (ry1**(1/Tg)))
    # positivetests2a.append(positivetests2a[t-1] * (ry2**(1/Tg)))
    # positivetests12a.append(positivetests2a[t-1] * (ry2**(1/Tg))+ positivetests1[t-1] 
    #                                              * (ry1**(1/Tg)))

    positivetests12.append(pt1+pt2)

    if showcummulative:
        cpt1 = (cummulative1[t-1]+  pt1)
        cpt2 = (cummulative2[t-1]+  pt2 )
        cpt12 =  (cummulative12[t-1]+ pt1 + pt2)
        
        if cpt1>=totalpopulation:
            cpt1 = totalpopulation
        if cpt2>=totalpopulation:
            cpt2 = totalpopulation
        if cpt12>=totalpopulation:
            cpt12 = totalpopulation

        cummulative1.append   (cpt1)
        cummulative2.append   (cpt2 )
        cummulative12.append   (cpt12)

    if (pt1+pt2)>0:
        ratio_ =  ((pt2/(pt1+pt2)))
    else:
        ratio_ = 1

    ratio.append   (100*ratio_)
    positivetestsper100k.append((pt1+pt2)/25)

    if showimmunization:
        totalimmune_ = totalimmune[t-1]+((pt1+pt2)*testimmunefactor)
        if totalimmune_>=totalpopulation:
            totalimmune_ = totalpopulation
        totalimmune.append(totalimmune_)   

    if showcummulative:
        if t>averagedayssick:
            infected.append (infected[t-1]+(((pt1+pt2))*testimmunefactor) -
                               (( positivetests1[t-averagedayssick]+ positivetests2[t-averagedayssick])*testimmunefactor )
                             ) 
            suspectible.append(suspectible[t-1]-(((pt1+pt2))*testimmunefactor) )
            recovered.append(recovered[t-1]+ 
                          (( positivetests1[t-averagedayssick]+positivetests2[t-averagedayssick])
                               * testimmunefactor ) ) 
        else:
            infected.append ( infected[t-1]+((pt1+pt2)*testimmunefactor) - 
                              (infected[0]/averagedayssick))  
            suspectible.append(suspectible[t-1]-(((pt1+pt2))*testimmunefactor) )
            recovered.append(recovered[t-1]+  (infected[0]/averagedayssick))
            
    ry1x.append(ry1)
    ry2x.append(ry2)
    walkingR.append((ry1**(1-ratio_))*(ry2**(ratio_)))

    if t>=from_test_to_hospital:
        hospital.append(positivetests12[t-from_test_to_hospital]*(percentage_test_hospital/100))
    else:
        hospital.append(None)
    
    

    
    if t < hospital_days_stay:
        delta_hospital += -1 * (hospital_dayzero / hospital_days_stay) 
    else:
        hospital_temp = hospital[t-hospital_days_stay] 
        if hospital_temp == None:
            hospital_temp = 0
        delta_hospital += -1 * hospital_temp    
          
    if t> from_test_to_hospital:
        delta_hospital +=(positivetests12[t-from_test_to_hospital]*(percentage_test_hospital/100))

   # ERAF
    if t <= ic_days_stay:
        delta_ic += -1 * (ic_dayzero / ic_days_stay) 
        # xy = (t-ic_days_stay) * -1
        # delta_ic -=(terugk[xy]*(percentage_test_ic/100)) 
        # eraf =+ (terugk[xy]*(percentage_test_ic/100))
        eraf =   (ic_dayzero / ic_days_stay) 
        blabla +=delta_ic
        #print (delta_ic) 
        #print (str(xy) + " "+ str(delta_ic)) 
    else:
        ic_temp = ic_opnames[t-ic_days_stay] 
        if ic_temp == None:
            ic_temp = 0
        delta_ic += -1 * ic_temp  
        eraf += ic_temp  

    #delta_ic -=lijstje[t]
    # ERBIJ

    if t>from_test_to_ic:
        ic_opnames.append(positivetests12[t-from_test_to_ic]*(percentage_test_ic/100))
        delta_ic +=      (positivetests12[t-from_test_to_ic]*(percentage_test_ic/100))
        erbij +=         (positivetests12[t-from_test_to_ic]*(percentage_test_ic/100))
      
    else:
        #ic_opnames.append(None)
        xy = 1+((t-ic_days_stay) * -1)
        ic_opnames.append(terugk[xy]*(percentage_test_ic/100))
        delta_ic +=      (terugk[xy]*(percentage_test_ic/100)) 
        erbij +=         (terugk[xy]*(percentage_test_ic/100)) 
        #erbij += (terugk[xy]*(percentage_test_ic/100)) 
        
    

                 
    erbij_.append(erbij)
    eraf_.append(eraf)

    hospital_cumm.append(hospital_cumm[t-1]+delta_hospital)
    ic_cumm.append(ic_cumm[t-1]+delta_ic)

erbijmoved = []
# for xxx in range(0,NUMBEROFDAYS-5):
#     if xxx<13:
#         erbijmoved.append(None)
#     else:
#         erbijmoved.append(erbij_[xxx+5])

# print (ic_cumm)
#print (ic_cumm)
st.title('IC-bezetting in NL - NOT VALIDATED')
#st.write (f"Attention. In the days before there were {int(blabla2)} persons admitted (related to the R-number), but the IC had alaready {ic_dayzero} persons. This causes the graph not ending at 0.")


def th2r(rz):
    th = int( Tg_ * math.log(0.5) / math.log(rz))
    return th

def r2th(th):
    r = int(10**((Tg_*mat.log(2))/th))
    # HK is using  r = 2**(Tg_/th)
    return r

def getsecondax():
    # get second y axis
    # Door Han-Kwang Nienhuys - MIT License
    # https://github.com/han-kwang/covid19/blob/master/nlcovidstats.py
    ax2 = ax.twinx()
    T2s = np.array([-2, -4,-7, -10, -11,-14, -21, -60, 9999, 60, 21, 14, 11,10, 7, 4, 2])
    y2ticks = 2**(Tg_/T2s)
    y2labels = [f'{t2 if t2 != 9999 else "∞"}' for t2 in T2s]
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(y2labels)
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel('Halverings-/verdubbelingstijd (dagen)')

def configgraph(titlex):
    interval_ = int(numberofdays_ / 20)
    plt.xlabel('date')
    plt.xlim(x[0], x[-1])
    todaylabel = "Today ("+ b + ")"
    plt.axvline(x=x[0]+datediff, color='yellow', alpha=.6,linestyle='--',label = todaylabel)
    if turning:
        pass
        #plt.axvline(x=x[0]+tp, color='orange', alpha=.6,linestyle='--',label = "First turningpoint")
        #plt.axvline(x=x[0]+tptd, color='orange', alpha=.4,linestyle='--',label = "First turningpoint")
  
        #plt.axvline(x=x[0]+tp2, color='orange', alpha=.6,linestyle='--',label = "Second turningpoint")
        #plt.axvline(x=x[0]+tptd2, color='orange', alpha=.6,linestyle='--',label = "Second turningpoint")
    
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    plt.title(titlex , fontsize=10)

    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

################################################

# IC bezetting
with _lock:
    fig1g, ax = plt.subplots()
  
    plt.plot(x, ic_cumm, label='IC bezetting per dag')
    #plt.plot(x, erbij_, label='IC erbij')
    #plt.plot(x, eraf_, label='IC eraf')
    #plt.plot(x, erbijmoved, label='IC erbijmoved')
    # Add X and y Label and limits

    plt.ylabel('Ziekenhuis- en IC-bezetting per day')
    plt.ylim(bottom = 0)
    #plt.axhline(y=1300, color='blue', alpha=.6,linestyle='--' )
    #plt.axhline(y=750, color='yellow', alpha=.6,linestyle='--')
    #plt.axhline(y=1000, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=1700, color='red', alpha=.6,linestyle='--')
    plt.fill_between(x, 750, 1349, color='#FFB1B4', label='Uitstellen reg. zorg') #1000 normale capaciteit 1350 =+35%
    plt.fill_between(x, 1350, 1699, color='#FE8081', label='IC Overbelast')
    plt.fill_between(x, 1700, 10000, color='#B25A5A', label='Code zwart')
    # plt.axvline(x=x[0]+21, color='yellow', alpha=.4,linestyle='--')
    # plt.axvline(x=x[0]+35, color='yellow', alpha=.4,linestyle='--')


    # Add a title
    titlex = (f'IC ({percentage_test_ic}%) bezetting per day, R={Rnew_2_ }, start={ic_dayzero}')
    configgraph(titlex) 
    st.pyplot(fig1g)
try:
    st.write(f"Value for IC occupation t=21         : {int((ic_cumm[21]))}")
    st.write(f"Value for IC occupation t=35         : {int((ic_cumm[35]))}")
    st.write(f"Maximum value for IC occupation       : {int(max(ic_cumm))}")
    
except:
    pass
# st.write("ERBIJ")
# st.write(erbij_)
# st.write(f"erbij :  {sum(erbij_)}")
# st.write("ERAF")
# st.write( eraf_)
# st.write(f"eraf : { sum(eraf_)}")
# st.write(f"VERSCHIL :  { sum(erbij_)-sum(eraf_)}")
# st.write("IC BEZETTING")
# st.write(ic_cumm)
#  # Ziekenhuis bezetting
if show_hospital:
    with _lock:
        fig1h, ax = plt.subplots()
        plt.plot(x, hospital_cumm, label='Ziekenhuis bezetting per dag')
        # Add X and y Label and limits

        plt.ylabel('Ziekenhuisbezetting per day')
        plt.ylim(bottom = 0)
        # plt.axhline(y=1300, color='blue', alpha=.6,linestyle='--' )
        # plt.axhline(y=750, color='yellow', alpha=.6,linestyle='--')
        # plt.axhline(y=1000, color='orange', alpha=.6,linestyle='--')
        # plt.axhline(y=1500, color='red', alpha=.6,linestyle='--')
        
        plt.axvline(x=21, color='yellow', alpha=.4,linestyle='--')
        plt.axvline(x=35, color='yellow', alpha=.4,linestyle='--')

        # Add a title
        titlex = (f'Ziekenhuis ({percentage_test_hospital}%)  bezetting per day, R={Rnew_2_ }, start={hospital_dayzero}')
        configgraph(titlex) 
        st.pyplot(fig1h)
        st.write(f"Value for hospital occupation t=21   : {int((hospital_cumm[21]))}")
        st.write(f"Value for hospital occupation t=35   : {int((hospital_cumm[35]))}")
        st.write(f"Maximum value for hospital occupation : {int(max(hospital_cumm))}")


 

# # #########################
# # # Ziekenhuis opnames
with _lock:
    fig1g, ax = plt.subplots()
    plt.plot(x, hospital, label='Ziekenhuis per dag')
    plt.plot(x, ic_opnames, label='IC per dag')
    # Add X and y Label and limits

    plt.ylabel('Ziekenhuis- en IC-opnames per day')
    plt.ylim(bottom = 0)
    plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    plt.axhline(y=12, color='yellow', alpha=.6,linestyle='--')
    plt.axhline(y=40, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=80, color='red', alpha=.6,linestyle='--')
    
    # https://twitter.com/YorickB/status/1369253144014782466/photo/1
    # Add a title
    titlex = ('Ziekenhuis ('+str(percentage_test_hospital)+ ' %) en IC ('+str(percentage_test_ic)+' %) opnames per day,\n7 dgn vertraging')
    configgraph(titlex)
    st.pyplot(fig1g)


# hospital = []
# ic_opnames = []


# # POS TESTS /day ################################
with _lock:
    fig1, ax = plt.subplots()
    #plt.plot(x, positivetests1, label=label1,  linestyle='--')
    plt.plot(x, positivetests2, label=label2,  linestyle='--')
    plt.plot(x, positivetests12, label='Total')
    
    # Add X and y Label and limits
    plt.ylabel('positive tests per day')
    plt.ylim(bottom = 0)

    plt.fill_between(x, 0, 875, color='#f392bd',  label='waakzaam')
    plt.fill_between(x, 876, 2500, color='#db5b94',  label='zorgelijk')
    plt.fill_between(x, 2501, 6250, color='#bc2165',  label='ernstig')
    plt.fill_between(x, 6251, 10000, color='#68032f', label='zeer ernstig')
    plt.fill_between(x, 10000, 20000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    # Add a title
    titlex = (
        'Pos. tests per day.\n'
        'Number of cases on '+ str(a) + ' = ' + str(numberofpositivetests) + '\n')
    configgraph(titlex)
    st.pyplot(fig1)

    
# Show the R number in time
with _lock:
    fig1f, ax = plt.subplots()
    plt.plot(x, ry2x, label='New variant',  linestyle='--')

    walkingR = []

    # Add X and y Label and limits
    plt.ylabel('R-number')
    #plt.ylim(bottom = 0)

    # Add a title
    titlex = ('R number in time.\n')
    plt.title(titlex , fontsize=10)
    configgraph(titlex)
    plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
    getsecondax()
    #secax = ax.secondary_yaxis('right', functions=(r2th,th2r))
    #secax.set_ylabel('Thalf')
    st.pyplot(fig1f)

with _lock:
    fig1i, ax = plt.subplots()
    plt.plot(x, suspectible, label='Suspectible',  linestyle='--')
    plt.plot(x, infected, label='Infected',  linestyle='--')
    plt.plot(x, recovered, label='Recovered',  linestyle='--')
    infected = []

    # Add  y Label and limits
    plt.ylabel('No of cases')
    plt.ylim(bottom = 0)

    # Add a title
    titlex = ('Suspectible - Infected - Recovered.\nBased on positive tests.\n'
                '(test/immunityfactor is taken in account)')
    configgraph(titlex)
    
    st.pyplot(fig1i)

positivetests1 = []
positivetests2 = []
positivetest12 = []
   



#####################################################
disclaimernew=('<style> .infobox {  background-color: lightyellow; padding: 10px;margin: 20-px}</style>'
               '<div class=\"infobox\"><h3>Disclaimer</h3><p>For illustration purpose only.</p>'
               '<p>Attention: these results are different from the official models'
               ' probably due to simplifications and different (secret) parameters.'
               '(<a href=\"https://archive.is/dqOjs\" target=\"_blank\">*</a>) '
                'The default parameters on this site are the latest known parameters of the RIVM'
                '</p><p>Forward-looking projections are estimates of what <em>might</em> occur. '
                'They are not predictions of what <em>will</em> occur. Actual results may vary substantially. </p>'
                 '<p>The goal was/is to show the (big) influence of (small) changes in the R-number. '
              'At the bottom of the page are some links to more advanced (SEIR) models.</p></div>')

st.markdown(disclaimernew,  unsafe_allow_html=True)

if showimmunization:
    disclaimerimm = ('<div class=\"infobox\"><p>The flattening  is very indicational. It is based on the principe R<sub>t</sub> = R<sub>start</sub> x (Suspectible / Population)<sup>λ</sup>. '
            'A lot of factors are not taken into account.'
        'The number of test is multiplied by ' +str(testimmunefactor)+ ' to get an estimation of the number of immune persons</div>'
        )

    st.markdown(disclaimerimm, unsafe_allow_html=True)
#        'Inspired by <a href=\'https://twitter.com/RichardBurghout/status/1357044694149128200\' target=\'_blank\'>this tweet</a>.<br> '


tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
    'Inspired by <a href=\"https://twitter.com/mzelst/status/1350923275296251904\" target=\"_blank\">this tweet</a> of Marino van Zelst.<br>'
    'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a>.</div>')

links = (
'<h3>Useful dashboards</h3><ul>'
'<li><a href=\"https://allecijfers.nl/nieuws/statistieken-over-het-corona-virus-en-covid19/\" target=\"_blank\">Allecijfers.nl</a></li>'
'<li><a href=\"https://datagraver.com/corona\" target=\"_blank\">https://www.datagraver/corona/</a></li>'
'<li><a href=\"https://www.bddataplan.nl/corona\" target=\"_blank\">https://www.bddataplan.nl/corona/</a></li>'
'<li><a href=\"https://renkulab.shinyapps.io/COVID-19-Epidemic-Forecasting/_w_ebc33de6/_w_dce98783/_w_0603a728/_w_5b59f69e/?tab=jhu_pred&country=France\" target=\"_blank\">Dashboard by  Institute of Global Health, Geneve, Swiss</a></li>'
'<li><a href=\"https://coronadashboard.rijksoverheid.nl/\" target=\"_blank\">Rijksoverheid NL</a></li>'
'<li><a href=\"https://www.corona-lokaal.nl/locatie/Nederland\" target=\"_blank\">Corona lokaal</a></li>'
'</ul>'


'<h3>Other (SEIR) models</h3><ul>'
'<li><a href=\"http://gabgoh.github.io/COVID/index.html\" target=\"_blank\">Epidemic Calculator </a></li>'
'<li><a href=\"https://www.covidsim.org" target=\"_blank\">COVID-19 Scenario Analysis Tool (Imperial College London)</a></li>'
'<li><a href=\"http://www.covidsim.eu/" target=\"_blank\">CovidSIM/a></li>'
'<li><a href=\"https://covid19-scenarios.org/\" target=\"_blank\">Covid scenarios</a></li>'
'<li><a href=\"https://share.streamlit.io/lcalmbach/pandemic-simulator/main/app.py\" target=\"_blank\">Pandemic simulator</a></li>'
'<li><a href=\"https://penn-chime.phl.io/\" target=\"_blank\">Hospital impact model</a></li>'
'<li><a href=\"http://www.modelinginfectiousdiseases.org/\" target=\"_blank\">Code from the book Modeling Infectious Diseases in Humans and Animals '
'(Matt J. Keeling & Pejman Rohani)</a></li></ul>'
'<h3>Other sources/info</h3>'
'<ul><li><a href=\"https://archive.is/dqOjs\" target=\"_blank\">Waarom bierviltjesberekeningen over het virus niet werken</a></li>'
'<li><a href=\"https://www.scienceguide.nl/2020/03/modellen-geven-geen-absolute-zekerheid-maar-ze-zijn-ontontbeerlijk/\" target=\"_blank\">Modellen geven geen absolute zekerheid, maar ze zijn onontbeerlijk</a></li>'
'<li><a href=\"https://www.nature.com/articles/d41586-020-02009-ws\" target=\"_blank\">A guide to R — the pandemic’s misunderstood metric</a></li></ul>')

vaccinationdisclaimer = (
'<h3>Attention</h3>'
'The plot when having vaccination is very indicative and very simplified.'
' It assumes an uniform(?) distribution of the vaccins over the population, '
' that a person who had the vaccin can\'t be sick immediately, '
'that everybody takes the vaccin, the R is equal for everybody etc. etc.')

st.sidebar.markdown(tekst, unsafe_allow_html=True)
#st.sidebar.info(tekst)
if vaccination:
    st.markdown(vaccinationdisclaimer, unsafe_allow_html=True)
st.markdown(links, unsafe_allow_html=True)
