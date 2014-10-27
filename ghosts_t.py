import numpy as np
import pylab as plt
import pickle
       
"""
This class produces the theoretical ghost patterns of a simple two source case. It can handle any array layout.
It requires that you set up the array geometry matrices. Two have been setup by default. You have a simple
3 element interferometer and WSRT to choose from (you can add any other array that you wish).
"""
class T_ghost():

    """This function initializes the theoretical ghost object
       point_sources --- contain the two source model (see main for an example) 
       antenna --- you can select a subset of the antennas, ignore some if you wish, "all" for all antennas, [1,2,3], for subset
       MS --- selects your antenna layout, "EW_EXAMPLE" or "WSRT"
    """
    def __init__(self,
                point_sources = np.array([]),
                antenna = "",
                MS=""):
        self.antenna = antenna
        self.A_1 = point_sources[0,0]
        self.A_2 = point_sources[1,0]
        self.l_0 = point_sources[1,1]
        self.m_0 = point_sources[1,2]

        # Here you can add your own antenna layout, default is WSRT and EW-EXAMPLE
        if MS == "EW_EXAMPLE":
           self.ant_names = [0,1,2]
           self.a_list = self.get_antenna(self.antenna,self.ant_names)
           #The 3 geometry matrices of the simple three element example
           self.b_m = np.zeros((3,3))          
           self.theta_m = np.zeros((3,3))          
           self.phi_m = np.array([(0,3,5),(-3,0,2),(-5,-2,0)]) 
           self.sin_delta = None
           self.wave = 3e8/1.45e9
           self.dec = np.pi/2.0       
        elif MS == "WSRT": # traditional (36,108,1332,1404) WSRT configuration 
             self.ant_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
             #The 3 geometry matrices of WSRT
             self.a_list = self.get_antenna(self.antenna,self.ant_names)
             self.b_m = np.zeros((14,14))          
             self.theta_m = np.zeros((14,14))          
             self.phi_m = 4*np.array([(0,1,2,3,4,5,6,7,8,9,9.25,9.75,18.25,18.75),
                           (-1,0,1,2,3,4,5,6,7,8,8.25,8.75,17.25,17.75),
                           (-2,-1,0,1,2,3,4,5,6,7,7.25,7.75,16.25,16.75),
                           (-3,-2,-1,0,1,2,3,4,5,6,6.25,6.75,15.25,15.75),
                           (-4,-3,-2,-1,0,1,2,3,4,5,5.25,5.75,14.25,14.75),
                           (-5,-4,-3,-2,-1,0,1,2,3,4,4.25,4.75,13.25,13.75),
                           (-6,-5,-4,-3,-2,-1,0,1,2,3,3.25,3.75,12.25,12.75),
                           (-7,-6,-5,-4,-3,-2,-1,0,1,2,2.25,2.75,11.25,11.75),
                           (-8,-7,-6,-5,-4,-3,-2,-1,0,1,1.25,1.75,10.25,10.75),
                           (-9,-8,-7,-6,-5,-4,-3,-2,-1,0,0.25,0.75,9.25,9.75),
                           (-9.25,-8.25,-7.25,-6.25,-5.25,-4.25,-3.25,-2.25,-1.25,-0.25,0,0.5,9,9.5),
                           (-9.75,-8.75,-7.75,-6.75,-5.75,-4.75,-3.75,-2.75,-1.75,-0.75,-0.5,0,8.5,9),
                           (-18.25,-17.25,-16.25,-15.25,-14.25,-13.25,-12.25,-11.25,-10.25,-9.25,-9,-8.5,0,0.5),
                           (-18.75,-17.75,-16.75,-15.75,-14.75,-13.75,-12.75,-11.75,-10.75,-9.75,-9.5,-9,-0.5,0)])
             self.sin_delta = None
             self.wave = 3e8/1.45e9 #observational wavelenght
             self.dec = np.pi/2.0 #declination       
 
    """Function processes your antenna selection string
    """
    def get_antenna(self,ant,ant_names):
       if isinstance(ant[0],int) :
          return np.array(ant)
       if ant == "all":
          return np.arange(len(ant_names))
       new_ant = np.zeros((len(ant),))
       for k in xrange(len(ant)):
           for j in xrange(len(ant_names)):
               if (ant_names[j] == ant[k]):
                 new_ant[k] = j
       return new_ant
           
    """Function calculates your delete list
    """
    def calculate_delete_list(self):
        if self.antenna == "all":
           return np.array([])
        d_list = list(xrange(self.phi_m.shape[0]))
        for k in range(len(self.a_list)):
            d_list.remove(self.a_list[k])
        return d_list
      
    """Generates your extrapolated visibilities
    baseline --- which baseline do you whish to image [0,1], selects baseline 01
    u,v --- if you do not want to create extrapolated visibilities you can give an actual uv-track here
    resolution --- the resolution of the pixels in your final image in arcseconds
    image_s --- the final extend of your image in degrees
    wave --- observational wavelength in meters
    dec --- declination of your observation
    approx --- if true we use Stefan Wijnholds approximation instead of ALS calibration
    scale --- standard deviation of your noise (very simplistic noise, please make rigorous by yourself) 
    """
    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D(self,baseline,u=None,v=None,resolution=0,image_s=0,s=0,wave=None,dec=None,approx=False,scale=None):
        if wave == None:
           wave = self.wave
        if dec == None:
           dec = self.dec
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = (self.b_m[b_list[0],b_list[1]]/wave)*np.cos(dec)
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            b_new = (b_new/wave)*np.cos(dec)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################

        #print "theta_new = ",theta_new
        #print "b_new = ",b_new
        #print "phi_new = ",phi_new 
        #print "delta_sin = ",self.sin_delta 

        #print "phi = ",phi
        #print "delta_b = ",delta_b
        #print "theta = ",theta*(180/np.pi)

        if u <> None:
           u_dim1 = len(u)
           u_dim2 = 1
           uu = u
           vv = v
           l_cor = None
           m_cor = None
        else:
           # FFT SCALING
           ######################################################
           delta_u = 1/(2*s*image_s*(np.pi/180))
           delta_v = delta_u
           delta_l = resolution*(1.0/3600.0)*(np.pi/180.0)
           delta_m = delta_l
           N = int(np.ceil(1/(delta_l*delta_u)))+1

           if (N % 2) == 0:
              N = N + 1

           delta_l_new = 1/((N-1)*delta_u)
           delta_m_new = delta_l_new  
           u = np.linspace(-(N-1)/2*delta_u,(N-1)/2*delta_u,N)
           v = np.linspace(-(N-1)/2*delta_v,(N-1)/2*delta_v,N)
           l_cor = np.linspace(-1/(2*delta_u),1/(2*delta_u),N)
           m_cor = np.linspace(-1/(2*delta_v),1/(2*delta_v),N)
           uu,vv = np.meshgrid(u,v)
           u_dim1 = uu.shape[0]
           u_dim2 = uu.shape[1] 
           #######################################################
        
        #DO CALIBRATION
        ####################################################### 

        V_R_pq = np.zeros(uu.shape,dtype=complex)
        V_G_pq = np.zeros(uu.shape,dtype=complex)
        temp =np.ones(phi_new.shape ,dtype=complex)

        for i in xrange(u_dim1):
            for j in xrange(u_dim2):
                if u_dim2 <> 1:
                   u_t = uu[i,j]
                   v_t = vv[i,j]
                else:
                   u_t = uu[i]
                   v_t = vv[i]
                #BASELINE CORRECTION (Single operation)
                #####################################################
                #ADDITION
                v_t = v_t - delta_b
                #SCALING
                u_t = u_t/phi
                v_t = v_t/(np.absolute(np.sin(dec))*phi)
                #ROTATION (Clockwise)
                u_t_r = u_t*np.cos(theta) + v_t*np.sin(theta)
                v_t_r = -1*u_t*np.sin(theta) + v_t*np.cos(theta)
                #u_t_r = u_t
                #v_t_r = v_t
                #NON BASELINE TRANSFORMATION (NxN) operations
                #####################################################
                #ROTATION (Anti-clockwise)
                u_t_m = u_t_r*np.cos(theta_new) - v_t_r*np.sin(theta_new)
                v_t_m = u_t_r*np.sin(theta_new) + v_t_r*np.cos(theta_new)
                #u_t_m = u_t_r
                #v_t_m = v_t_r
                #SCALING
                u_t_m = phi_new*u_t_m
                v_t_m = phi_new*np.absolute(np.sin(dec))*v_t_m
                #ADDITION
                v_t_m = v_t_m + b_new
           
                #print "u_t_m = ",u_t_m
                #print "v_t_m = ",v_t_m                
                
                #NB --- THIS IS WHERE YOU ADD THE NOISE
                if scale == None:
                   R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0))
                else:
                   R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0)) + np.random.normal(size=u_t_m.shape,scale=scale)

                if not approx:
                   d,Q = np.linalg.eigh(R)
                   D = np.diag(d)
                   Q_H = Q.conj().transpose()
                   abs_d=np.absolute(d)
                   index=abs_d.argmax()
                   if (d[index] >= 0):
                      g=Q[:,index]*np.sqrt(d[index])
                   else:
                      g=Q[:,index]*np.sqrt(np.absolute(d[index]))*1j
                   G = np.dot(np.diag(g),temp)
                   G = np.dot(G,np.diag(g.conj()))
                   if self.antenna == "all":
                      if u_dim2 <> 1:
                         V_R_pq[i,j] = R[b_list[0],b_list[1]]
                         V_G_pq[i,j] = G[b_list[0],b_list[1]]
                      else:
                         V_R_pq[i] = R[b_list[0],b_list[1]]
                         V_G_pq[i] = G[b_list[0],b_list[1]]
                   else:
                       for k in xrange(p_new.shape[0]):
                           for l in xrange(p_new.shape[1]):
                               if (p_new[k,l] == b_list[0]) and (q_new[k,l] == b_list[1]):
                                  if u_dim2 <> 1:
                                     V_R_pq[i,j] = R[k,l]
                                     V_G_pq[i,j] = G[k,l]
                                  else:
                                     V_R_pq[i] = R[k,l]
                                     V_G_pq[i] = G[k,l]
                else:
                    R1 = (R - self.A_1)/self.A_2
                    P = R1.shape[0]
                    if self.antenna == "all":
                       G = self.A_1 + ((0.5*self.A_2)/P)*(np.sum(R1[b_list[0],:])+np.sum(R1[:,b_list[1]]))
                       G = (G + ((0.5*self.A_2)/P)**2*R1[b_list[0],b_list[1]]*np.sum(R1))
                       if u_dim2 <> 1:
                          V_R_pq[i,j] = R[b_list[0],b_list[1]]
                          V_G_pq[i,j] = G
                       else:
                          V_R_pq[i] = R[b_list[0],b_list[1]]
                          V_G_pq[i] = G
                    else:
                        for k in xrange(p_new.shape[0]):
                            for l in xrange(p_new.shape[1]):
                                if (p_new[k,l] == b_list[0]) and (q_new[k,l] == b_list[1]):
                                   G = self.A1 + ((0.5*self.A2)/P)*(np.sum(R1[k,:])+np.sum(R1[:,l]))
                                   G = (G + ((0.5*self.A2)/P)**2*R1[k,l]*np.sum(R1))
                                   if u_dim2 <> 1:
                                      V_R_pq[i,j] = R[k,l]
                                      V_G_pq[i,j] = G
                                   else:
                                      V_R_pq[i] = R[k,l]
                                      V_G_pq[i] = G
        return u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor
    
    def vis_function(self,type_w,avg_v,V_G_pq,V_G_qp,V_R_pq):
        if type_w == "R":
           vis = V_R_pq
        elif type_w == "RT":
           vis = V_R_pq**(-1)
        elif type_w == "R-1":
           vis = V_R_pq - 1
        elif type_w == "RT-1":
           vis = V_R_pq**(-1)-1
        elif type_w == "G":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2
           else:
              vis = V_G_pq
        elif type_w == "G-1":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2-1
           else:
              vis = V_G_pq-1
        elif type_w == "GT":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2
           else:
              vis = V_G_pq**(-1)
        elif type_w == "GT-1":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2-1
           else:
              vis = V_G_pq**(-1)-1
        elif type_w == "GTR-R":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq - V_R_pq
        elif type_w == "GTR":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq
        elif type_w == "GTR-1":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-1
           else:
              vis = V_G_pq**(-1)*V_R_pq-1
        return vis
    
    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_2D(self,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False,wave=None,dec=None,approx=False):
        if wave  == None:
           wave = self.wave           
        if dec == None:
           dec = self.dec
        ant_len = len(self.a_list)
        counter = 0
        baseline = [0,0]

        for k in xrange(ant_len):
            for j in xrange(k+1,ant_len):
                baseline[0] = self.a_list[k]
                baseline[1] = self.a_list[j]
                counter = counter + 1                 
                print "counter = ",counter
                print "baseline = ",baseline
                if avg_v:
                   baseline_new = [0,0]
                   baseline_new[0] = baseline[1]
                   baseline_new[1] = baseline[0]
                   u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)
                else:
                   V_G_qp = 0

                u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)

                if (k==0) and (j==1):
                   vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
                else:
                   vis = vis + self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
        
        vis = vis/counter           

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)
        
        N = l_cor.shape[0]

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:
 
           image = np.fft.fft2(vis)/N**2

 
        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/1)*100

        if plot:

           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure() 
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True,dec=dec)
           
           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()
        
           fig = plt.figure() 
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True)
           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()

        return image,l_old,m_old
    
    """Generates your ghost map
    baseline --- which baseline do you whish to image [0,1], selects baseline 01
    resolution --- the resolution of the pixels in your final image in arcseconds
    image_s --- the final extend of your image in degrees
    s --- oversampling rate
    sigma --- size of kernal used to fatten the ghosts otherwise they would be delta functions
    type --- which matrix do you wish to image
    avg --- average between baseline pq and qp
    plot --- plot the image
    mask --- plot the theoretical derived positions with crosses
    label_v --- labels them --- NOT SUPPROTED
    wave --- observational wavelength in meters
    dec --- declination of your observation
    save_fig --- saves figure
    approx --- if true we use Stefan Wijnholds approximation instead of ALS calibration
    difference --- between stefans approach and standard
    scale --- standard deviation of your noise (very simplistic noise, please make rigorous by yourself) 
    """
    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_pq_2D(self,baseline,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False,wave=None,dec=None,label_v=False,save_fig=False,approxi=False,difference=False,scale=None):
        if wave == None:
           wave = self.wave
        if dec == None:
           dec = self.dec
        
        if avg_v:
           baseline_new = [0,0]
           baseline_new[0] = baseline[1]
           baseline_new[1] = baseline[0]
           u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approxi,scale=scale)
        else:
           V_G_qp = 0

        u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approxi,scale=scale)


        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)
        
        N = l_cor.shape[0]

        vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
        if difference and approx:

           if avg_v:
              baseline_new = [0,0]
              baseline_new[0] = baseline[1]
              baseline_new[1] = baseline[0]
              u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=False)
           else:
              V_G_qp = 0

           u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=False)
           vis2 = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
           vis = vis2 - vis

        #vis = V_G_pq-1

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:
 
           image = np.fft.fft2(vis)/N**2

 
        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/0.1)*100

        if plot:

           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure() 
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           if label_v:
              self.plot_source_labels_pq(baseline,im=image_s,plot_x = False)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
             p = self.create_mask(baseline,plot_v = True,dec=dec)

           #for k in xrange(len(p)):
           #    plt.plot(p[k,1]*(180/np.pi),p[k,2]*(180/np.pi),"kv")

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Real")
           
           if save_fig:     
              plt.savefig("R_pq"+str(baseline[0])+str(baseline[1])+".pdf",bbox_inches="tight") 
              plt.clf()
           else:
              plt.show()
        
           fig = plt.figure() 
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           if label_v:
              self.plot_source_labels_pq(baseline,im=image_s,plot_x = False)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])
           
           if mask:
              self.create_mask(baseline,plot_v = True,dec=dec)

           plt.xlabel("$l$ [degrees]")
           plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Imag")
           plt.ylabel("$m$ [degrees]")
           if save_fig:     
              plt.savefig("I_pq"+str(baseline[0])+str(baseline[1])+".pdf",bbox_inches="tight") 
              plt.clf()
           else:
              plt.show()

        return image,l_old,m_old

    def plt_circle_grid(self,grid_m):
        plt.hold('on')
        rad = np.arange(1,1+grid_m,1)
        x = np.linspace(0,1,500)
        y = np.linspace(0,1,500)

        x_c = np.cos(2*np.pi*x)
        y_c = np.sin(2*np.pi*y)
        for k in range(len(rad)):
            plt.plot(rad[k]*x_c,rad[k]*y_c,"k",ls=":",lw=0.5)
    
    def create_mask_all(self,plot_v = False,dec=None):
        if dec == None:
           dec = self.dec
        sin_delta = np.absolute(np.sin(dec))
        
        point_sources = np.array([(1,0,0)])
        point_sources = np.append(point_sources,[(1,self.l_0,-1*self.m_0)],axis=0) 
        point_sources = np.append(point_sources,[(1,-1*self.l_0,1*self.m_0)],axis=0) 
        
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        d_list = self.calculate_delete_list()

        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################
        if plot_v == True:
           plt.plot(0,0,"rx")
           plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),"rx")
           plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),"rx")

        len_a = len(self.a_list)
        b_list = [0,0]

        first = True

        for h in xrange(len_a):
            for i in xrange(h+1,len_a):
                b_list[0] = self.a_list[h]
                b_list[1] = self.a_list[i]
                phi = self.phi_m[b_list[0],b_list[1]]
                delta_b = self.b_m[b_list[0],b_list[1]]
                theta = self.theta_m[b_list[0],b_list[1]]
                for j in xrange(theta_new.shape[0]):
                    for k in xrange(j+1,theta_new.shape[0]):
                        if not np.allclose(phi_new[j,k],phi):
                           l_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.l_0+sin_delta*np.sin(theta_new[j,k]-theta)*self.m_0)                
                           m_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.m_0-sin_delta**(-1)*np.sin(theta_new[j,k]-theta)*self.l_0)                
                           if plot_v == True:
                              plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),"rx")  
                              plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),"rx")  
                           point_sources = np.append(point_sources,[(1,l_cordinate,-1*m_cordinate)],axis=0) 
                           point_sources = np.append(point_sources,[(1,-1*l_cordinate,1*m_cordinate)],axis=0) 
        
        return point_sources

    def create_mask(self,baseline,plot_v = False,dec = None,plot_markers = False):
        if dec == None:
           dec = self.dec
        sin_delta = np.absolute(np.sin(dec))
        point_sources = np.array([(1,0,0)])
        point_sources_labels = np.array([(0,0,0,0)])
        point_sources = np.append(point_sources,[(1,self.l_0,-1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(baseline[0],baseline[1],baseline[0],baseline[1])],axis=0)
        point_sources = np.append(point_sources,[(1,-1*self.l_0,1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(baseline[1],baseline[0],baseline[0],baseline[1])],axis=0)
        
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = self.b_m[b_list[0],b_list[1]]
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################
        if plot_v == True:
           if plot_markers:
              mk_string = self.return_color_marker([0,0])
              plt.plot(0,0,self.return_color_marker([0,0]),label="(0,0)",mfc='none',ms=5)
              plt.hold('on')
              mk_string = self.return_color_marker(baseline)
              plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),self.return_color_marker(baseline),label="("+str(baseline[0])+","+str(baseline[1])+")",mfc='none',mec=mk_string[0],ms=5)
              mk_string = self.return_color_marker([baseline[1],baseline[0]])
              plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),self.return_color_marker([baseline[1],baseline[0]]),label="("+str(baseline[1])+","+str(baseline[0])+")",mfc='none',mec=mk_string[0],ms=5)
           else:             
              plt.plot(0,0,"rx")
              plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),"rx")
              plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),"gx")
        for j in xrange(theta_new.shape[0]):
            for k in xrange(j+1,theta_new.shape[0]):
                #print "Hallo:",j," ",k
                if not np.allclose(phi_new[j,k],phi):
                   #print "phi = ",phi_new[j,k]/phi
                   l_cordinate = (phi_new[j,k]*1.0)/(1.0*phi)*(np.cos(theta_new[j,k]-theta)*self.l_0+sin_delta*np.sin(theta_new[j,k]-theta)*self.m_0) 
                   #print "l_cordinate = ",l_cordinate*(180/np.pi)               
                   m_cordinate = (phi_new[j,k]*1.0)/(phi*1.0)*(np.cos(theta_new[j,k]-theta)*self.m_0-sin_delta**(-1)*np.sin(theta_new[j,k]-theta)*self.l_0)                
                   #print "m_cordinate = ",m_cordinate*(180/np.pi)               
                   if plot_v == True:
                      if plot_markers:
                         mk_string = self.return_color_marker([j,k])
                         plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),self.return_color_marker([j,k]),label="("+str(j)+","+str(k)+")",mfc='none',mec=mk_string[0],ms=5)  
                         mk_string = self.return_color_marker([k,j])
                         plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),self.return_color_marker([k,j]),label="("+str(k)+","+str(j)+")",mfc='none',mec=mk_string[0],ms=5) 
                         plt.legend(loc=8,ncol=9,numpoints=1,prop={"size":7},columnspacing=0.1) 
                      else:
                         plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),"rx")  
                         plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),"gx")  
                   point_sources = np.append(point_sources,[(1,l_cordinate,-1*m_cordinate)],axis=0) 
                   point_sources_labels = np.append(point_sources_labels,[(j,k,baseline[0],baseline[1])],axis=0)
                   point_sources = np.append(point_sources,[(1,-1*l_cordinate,1*m_cordinate)],axis=0) 
                   point_sources_labels = np.append(point_sources_labels,[(k,j,baseline[0],baseline[1])],axis=0)
        
        return point_sources,point_sources_labels

    #window is in degrees, l,m,point_sources in radians, point_sources[k,:] ---> kth point source 
    def extract_flux(self,image,l,m,window,point_sources,plot):
        window = window*(np.pi/180)
        point_sources_real = np.copy(point_sources)
        point_sources_imag = np.copy(point_sources)
        for k in range(len(point_sources)):
            l_0 = point_sources[k,1]
            m_0 = point_sources[k,2]*(-1)
            
            l_max = l_0 + window/2.0
            l_min = l_0 - window/2.0
            m_max = m_0 + window/2.0
            m_min = m_0 - window/2.0

            m_rev = m[::-1]

            #ll,mm = np.meshgrid(l,m)
            
            image_sub = image[:,(l<l_max)&(l>l_min)]
            #ll_sub = ll[:,(l<l_max)&(l>l_min)]
            #mm_sub = mm[:,(l<l_max)&(l>l_min)]
            
            if image_sub.size <> 0:  
               image_sub = image_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #ll_sub = ll_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #mm_sub = mm_sub[(m_rev<m_max)&(m_rev>m_min),:]
            
            #PLOTTING SUBSET IMAGE
            if plot:
               l_new = l[(l<l_max)&(l>l_min)]
               if l_new.size <> 0:
                  m_new = m[(m<m_max)&(m>m_min)]
                  if m_new.size <> 0:
                     l_cor = l_new*(180/np.pi)
                     m_cor = m_new*(180/np.pi)

                     # plt.contourf(ll_sub*(180/np.pi),mm_sub*(180/np.pi),image_sub.real)
                     # plt.show()

                     #fig = plt.figure() 
                     #cs = plt.imshow(mm*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()
                     #fig = plt.figure() 
                     #cs = plt.imshow(ll*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()
                   
                     fig = plt.figure() 
                     cs = plt.imshow(image_sub.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
                     fig.colorbar(cs)
                     plt.title("REAL")
                     plt.show()
                     fig = plt.figure() 
                     cs = plt.imshow(image_sub.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     fig.colorbar(cs)
                     plt.title("IMAG")
                     plt.show()
            #print "image_sub = ",image_sub
            if image_sub.size <> 0:
               max_v_r = np.amax(image_sub.real)
               max_v_i = np.amax(image_sub.imag)
               min_v_r = np.amin(image_sub.real)
               min_v_i = np.amin(image_sub.imag)
               if np.absolute(max_v_r) > np.absolute(min_v_r):
                  point_sources_real[k,0] = max_v_r
               else:
                  point_sources_real[k,0] = min_v_r
               if np.absolute(max_v_i) > np.absolute(min_v_i):
                  point_sources_imag[k,0] = max_v_i
               else:
                  point_sources_imag[k,0] = min_v_i
           
            else:
              point_sources_real[k,0] = 0
              point_sources_imag[k,0] = 0
       
        return point_sources_real,point_sources_imag          
    
if __name__ == "__main__":
        #two source model, can only support two point sources (at center and 1 degree right of origin) [flux,l coordinate,m coordinate]
        point_sources = np.array([(1,0,0),(0.2,(1*np.pi)/180,(0*np.pi)/180)])
        print "EXECUTING...."     
        #initializes ghost object
        t = T_ghost(point_sources,"all","WSRT")
        #plots ghost map for baseline 01 (resolution 150 arcseconds, extend 3 degrees)
        image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="G-1",plot=True,mask=True)
        #plots ghost map for baseline 12 (resolution 150 arcseconds, extend 3 degrees)
        image,l_v,m_v = t.sky_pq_2D([1,2],150,3,2,sigma = 0.05,type_w="G-1",plot=True,mask=True)

        #NB naive noise implementation verify if correct yourself....
        
