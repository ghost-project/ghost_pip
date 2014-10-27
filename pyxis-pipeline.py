import pyrap.tables
from pyrap.tables import table
from Pyxis.ModSupport import *
import mqt
import cal
import imager
import stefcal
import lsm
import ms
import std
import numpy as np
import pylab as plt
import pyfits

#Setting Pyxis global variables
v.OUTDIR = '.'
#v.MS = 'KAT7_1445_1x16_12h.ms'
v.MS = '3C147_spw0_1.MS'
v.CALORNOT = ''
v.DESTDIR_Template = "${OUTDIR>/}"
v.OUTFILE_Template = "${DESTDIR}${MS:BASE}"
imager.DIRTY_IMAGE_Template = "${OUTFILE}${_<CALORNOT}.dirty.fits"
imager.RESTORED_IMAGE_Template = "${OUTFILE}.restored.fits"
imager.RESIDUAL_IMAGE_Template = "${OUTFILE}.residual.fits"
imager.MASK_IMAGE_Template = "${OUTFILE}.mask.fits"
imager.MODEL_IMAGE_Template = "${OUTFILE}.model.fits"
v.LSM_Template = "${OUTFILE}${_<CALORNOT}.lsm.html"

#Setting standard pyxis imager settings
def image_settings(npix=2048,cellsize="11arcsec",mode ="channel",stokes="I",weight="natural",filter=None,wprojplanes=0,niter=1000,gain=0.1,threshold=0,clean_alg="hogbom"):
    imager.npix = npix
    imager.cellsize=cellsize
    imager.stokes=stokes
    imager.weight=weight
    imager.filter = filter
    imager.wprojplanes = wprojplanes
    imager.niter = niter
    imager.gain = gain
    imager.threshold = threshold
    imager.mode = mode
    imager.CLEAN_ALGORITHM = clean_alg

#Creating advanced pyxis imager settings
def image_advanced_settings(img_nchan=1,img_chanstart=0,img_chanstep=1):
    options = {}
    options["img_nchan"] = img_nchan
    options["img_chanstart"] = img_chanstart
    options["img_chanstep"] = img_chanstep
    return options

#generate flux values of sources (powe law distribution) 
def generate_flux(a = 1.2,num_sources = 10,plot = False):
    y = np.random.pareto(a, size=num_sources)
    if plot:
       count, bins, ignored = plt.hist(y, 100, normed=True)
       plt.xlabel("$x$")
       plt.ylabel("$p(x)$")
       plt.title("Pareto Distribition, $a = $%.2f"%(a))
       plt.show()
    return y

#generate positions values of sources (uniform distribution)
def generate_pos(fov = 3,num_sources=10):
    return np.random.uniform(low=-1*np.absolute(fov), high=np.absolute(fov), size = num_sources)*(np.pi/180)

#get the phase center for measurment set
def get_field_center():
    t = ms.ms(subtable="FIELD")
    phase_centre = (t.getcol("PHASE_DIR"))[0,0,:]
    t.close()
    return phase_centre[0], phase_centre[1] #ra0,dec0 in radians

#get the number of time slots
def get_time_slots():
    t = ms.ms()
    A1 = t.getcol("ANTENNA1")
    A2 = t.getcol("ANTENNA2")
    uvw=t.getcol("UVW")
    temp = uvw[(A1==0)&(A2==1),0]
    return(len(temp))

#compute the residual (CORR_DATA - MODEL)
def residual():
    t = ms.ms(write=True)
    corr_data = t.getcol("CORRECTED_DATA")
    data = t.getcol("DATA")
    t.putcol("CORRECTED_DATA",corr_data-data)
    t.close()

# converting from l and m coordinate system to ra and dec
def lm2radec(l,m):#l and m in radians
    rad2deg = lambda val: val * 180./np.pi
    ra0,dec0 = get_field_center() # phase centre in radians
    rho = np.sqrt(l**2+m**2)
    if rho==0:
       ra = ra0
       dec = dec0
    else:
       cc = np.arcsin(rho)
       ra = ra0 - np.arctan2(l*np.sin(cc), rho*np.cos(dec0)*np.cos(cc)-m*np.sin(dec0)*np.sin(cc))
       dec = np.arcsin(np.cos(cc)*np.sin(dec0) + m*np.sin(cc)*np.cos(dec0)/rho)
    return rad2deg(ra), rad2deg(dec)

# converting ra and dec to l and m coordiantes
def radec2lm(ra_d,dec_d):# ra and dec in degrees
    rad2deg = lambda val: val * 180./np.pi
    deg2rad = lambda val: val * np.pi/180
    ra0,dec0 = get_field_center() # phase centre in radians
    ra_r, dec_r = deg2rad(ra_d), deg2rad(dec_d) # coordinates of the sources in radians
    l = np.cos(dec_r)* np.sin(ra_r - ra0)
    m = np.sin(dec_r)*np.cos(dec0) - np.cos(dec_r)*np.sin(dec0)*np.cos(ra_r-ra0)
    return rad2deg(l),rad2deg(m)

# creating meqtrees skymodel (also creates a second sky model containing the brightest #num_cal_sources
def meqskymodel(point_sources,num_cal_sources=0):
    
    ind_sorted = np.argsort(point_sources[:,0])
    point_sources = point_sources[ind_sorted,:]
    point_sources = point_sources[::-1,:]
    #WRITE OUT ALL SOURCES
    str_out = "#format: name ra_d dec_d i\n"
    for i in range(len(point_sources)):
        #print "i = ",i
        amp, l ,m = point_sources[i,0], point_sources[i,1], point_sources[i,2]
        #m = np.absolute(m) if m < 0 else -1*m # changing the signs since meqtrees has its own coordinate system
        ra_d, dec_d = lm2radec(l,m)
        #print ra_d, dec_d
        #l_t,m_t = radec2lm(ra_d,dec_d)
        #print l_t, m_t
        name = "A"+ str(i)
        str_out += "%s %.10g %.10g %.4g\n"%(name, ra_d, dec_d,amp)
    file_name = II("${LSM:BASE}")+".txt"
    print "file_name = ",file_name
    simmodel = open(file_name,"w")
    simmodel.write(str_out)
    simmodel.close()
    x.sh("tigger-convert ${LSM:BASE}.txt -t ASCII --format \"name ra_d dec_d i\" -f ${LSM}")
    x.sh("rm -f ${LSM:BASE}.txt")

    #WRITE OUT BRIGHTEST SOURCES 
    v.CALORNOT = "cal_model"
    point_sources_sub = point_sources[0:num_cal_sources,:]
    str_out = "#format: name ra_d dec_d i\n"
    for i in range(len(point_sources_sub)):
        #print "i = ",i
        amp, l ,m = point_sources_sub[i,0], point_sources_sub[i,1], point_sources_sub[i,2]
        #m = np.absolute(m) if m < 0 else -1*m # changing the signs since meqtrees has its own coordinate system
        ra_d, dec_d = lm2radec(l,m)
        #print ra_d, dec_d
        #l_t,m_t = radec2lm(ra_d,dec_d)
        #print l_t, m_t
        name = "A"+ str(i)
        str_out += "%s %.10g %.10g %.4g\n"%(name, ra_d, dec_d,amp)
    file_name = II("${LSM:BASE}")+".txt"
    print "file_name = ",file_name
    calmodel = open(file_name,"w")
    calmodel.write(str_out)
    calmodel.close()
    x.sh("tigger-convert ${LSM:BASE}.txt -t ASCII --format \"name ra_d dec_d i\" -f ${LSM}")
    x.sh("rm -f ${LSM:BASE}.txt")
    v.CALORNOT = ''
   
#Simulate the visibilities
# If cal is True store in DATA else store in CORRECTED DATA, if cal true sim whole sky model if whole is true else calibration model.
# If cal is False sim complete sky model  
def sim_function(cal=False,whole=False):
    
    options = {}
    options['ms_sel.msname']=II("${MS}")     
    options['ms_sel.tile_size']=get_time_slots()
    if cal:
       if not whole: 
          v.CALORNOT = "cal_model"
       options['ms_sel.output_column']="DATA"
    else:
       options['ms_sel.output_column']="CORRECTED_DATA"
    options['tiggerlsm.filename']=II("${LSM}")
    
    mqt.run(script="turbo-sim.py",job="_tdl_job_1_simulate_MS",config="tdlconf.profiles",section="sim",options=options)
    v.CALORNOT = ''

#calibrate the visibilities
def cal_function(type_cal="LM"):

    options = {}
    options['ms_sel.msname']=II("${MS}")     
    options['ms_sel.tile_size']=get_time_slots()
    v.CALORNOT = "cal_model"
    #options['ms_sel.output_column']="CORRECTED_DATA"
    options['tiggerlsm.filename']=II("${LSM}")
    if type_cal=="LM":
       mqt.run(script="calico-wsrt-tens.py",job="cal_G_diag",config="tdlconf.profiles",section="selfcal",options=options)
    elif type_cal=="STEF":
       mqt.run(script="calico-stefcal.py",job="stefcal",config="tdlconf.profiles",section="G_calibration",options=options)
    #elif type_cal=="STEF_DIR": #CALLING STEFCAL directly fails at the moment
    #   stefcal.stefcal(section="G_calibration",output="CORR_DATA",options=options)
    v.CALORNOT = ''

def runall():
    #alpha_v = 5 #Power low distribution parameter
    num_sources_v = 2 #how many sources
    num_cal_sources_v = 1#how many sources in calibration model
    fov_v = 3 #degrees #the field of view in degrees

    #only do stefcal or not
    skip_LM = True
    dist = True 

    point_sources = np.array([(1,0,0),(0.2,(1*np.pi)/180,(np.pi*0)/180)]) #sets the pointsources in the sky

    #contain all the pointsources
    #point_sources = np.zeros((num_sources_v,3))

    #generate flux and positions
    #point_sources[:,0] = generate_flux(a = alpha_v,num_sources = num_sources_v,plot=False)
    #point_sources[:,1] = generate_pos(fov = fov_v,num_sources=num_sources_v)
    #point_sources[:,2] = generate_pos(fov = fov_v,num_sources=num_sources_v)
   
    ra0,dec0 = get_field_center()
    
    #generate true and calibration sky models from point_sources 
    meqskymodel(point_sources,num_cal_sources=num_cal_sources_v)
    
    #simulate complete skymodel --- store in CORRECTED_DATA  
    sim_function(cal=False)
    #simulate calibration skymodel --- store in DATA
    sim_function(cal=True)
    
    #set up imager
    image_settings()
    opt = image_advanced_settings()

    #make images of complete sky and calibrated sky model
    imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
    v.CALORNOT = "cal_model"
    imager.make_image(column="DATA",dirty=options,restore=False)
    v.CALORNOT = ''

    #determine residual CORRECTED_DATA-DATA and image
    residual()
    v.CALORNOT = "res"
    imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
    v.CALORNOT = ''

    #perform LM calibration
    if not (skip_LM):
       sim_function(cal=False)
    
       cal_function()
       v.CALORNOT = "cal_app_LM"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''
       residual()
       v.CALORNOT = "cal_res_LM"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''
    
    #perform STEFcal calibration
    if not (dist):
       sim_function(cal=False)
    
       cal_function(type_cal="STEF")
       v.CALORNOT = "cal_app_STEF"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''
       residual()
       v.CALORNOT = "cal_res_STEF"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''
    else: #perform STEFcal calibration (distilation)
       sim_function(cal=True,whole=True)
    
       cal_function(type_cal="STEF")
       v.CALORNOT = "cal_app_STEF"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''
       v.CALORNOT = "cal_whole_STEF"
       imager.make_image(column="DATA",dirty=options,restore=False)
       v.CALORNOT = ''
       residual()
       v.CALORNOT = "cal_res_dist_STEF"
       imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
       v.CALORNOT = ''

