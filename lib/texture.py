"""
DanMAX Texture

Collection of functions related to texture analysis from diffraction data collected at DanMAX.

## DanMAX coordinate definition ##
The coordinates are here defined from eulerian angles: ω χ, and φ, using YXY rotations. The laboratory coordinates X<sub>L</sub>, Y<sub>L</sub>, and Z<sub>L</sub> are defined such that Z<sub>L</sub> is along the beam,  Y<sub>L</sub> is vertical with the positive direction upwards and  X<sub>L</sub> is horizontal, pointing away from the ring. Consequently, ω is a rotation in the horizontal plane around Y<sub>L</sub>, χ is a rotation around the new X<sub>L</sub>', and φ is a rotation around Y<sub>L</sub>''.  
Example 1:  
A sample normal along the beam has the (x,y,z) coordinates (0,0,1). Rotating ω 90° would result in the new coordinates (x',y',z') = (1,0,0).  
Example 2:  
A sample normal along the beam rotated along all three angles 90° results in the following new corrdinates: (0,0,1) -> (1,0,0) -> (0,-1,0) -> (0,0,-1)  
  
NB: while the sample normal in example 2 ends up anti-parallel with the beam, it is *not* the same as a 180° ω-rotation as the sample will also have rotated around the normal. 

## Absorption correction ##
The transmission (absorption) calculation used here is a home-cooked variation of the equations presented in He, Bob B. Two-dimensional X-ray Diffraction. John Wiley & Sons, 2018 p.203-207.  
The beam path length is the sum of the incident and diffracted beam path length:  
t = z/cos(eta) + (t0-z)/cos(zeta)  
where t0 is the thickness of the (flat) sample, z is the distance from the surface, cos(eta) is the dot product of incident beam and the sample normal, and cos(zeta) is the dot product of the diffracted beam and the sample normal. (All vectors are unit vectors)  
The transmitted is then the integral of exp(-mu*t) over the sample thickness t0, normalized with the incident path length (https://shorturl.at/frvLX)
"""


from numpy import sin, cos, pi
import numpy as np
from pyFAI import geometry
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def R_yxy(omega=0,chi=0,phi=0):
     """
     YXY rotation matrix - Angles in degrees
     Relates the sample coordinates A and laboratory coordinates A_L, such that RA_L = A
     The transpose of R is equal to the inverse of R: R_T = R^-1 and RR_T=E, where E is the identity matrix.
     """
     o = omega*pi/180
     c = chi*pi/180
     p = phi*pi/180
     
     R = np.array([[ cos(o)*cos(p)-cos(c)*sin(o)*sin(p) , sin(c)*sin(p) , -cos(p)*sin(o)-cos(o)*cos(c)*sin(p)],
                   [ sin(o)*sin(c)                      , cos(c)        , cos(o)*sin(c)                      ],
                   [ cos(o)*sin(p)+cos(c)*cos(p)*sin(o)  , -cos(p)*sin(c), cos(o)*cos(c)*cos(p)-sin(o)*sin(p) ]])
     return(R)

    
def poleAngles(Q_s):
    """Return Pole figure radial and azimuthal coordinates (°) from scattering vector in sample coordinates"""
    # radial pole coordinate
    rad = np.arccos(abs(Q_s[2]))*180/pi
    #r_p = np.arcsin(abs(Q_s[2]))*180/pi
    # azimuthal pole coordinate
    azi = (360+np.arccos(-Q_s[0]/(np.sqrt(Q_s[0]**2+Q_s[1]**2)))*np.sign(-Q_s[1])*180/pi)%360
    return rad, azi


def Q_unit(eta=0,tth=0):
    """Return Q as unit vector - Angles in degrees"""
    e = eta*pi/180
    t = tth*pi/360
    # scattering vector in lab coord
    Q = np.stack([-cos(e)*cos(t),
                  -sin(e)*cos(t),
                  np.repeat(-sin(t),len(e))])
    return Q


def getS0(S0=[0,0,1]):
    """Return the incident beam unit vector in lab coordinates. Default along laboratory z-axis"""
    return np.array(S0)


def getS(tth,eta):
    """
    Return the scattered beam unit vector in lab coordinates. Not to be confused with the scattering vector Q.
    Parameters: 2theta and azimuthal angle (eta) in degrees.
    If both parameters are scalars, the returned array has the shape (3,).
    If one parameter is a scalar and the other an array with shape (n,), the returned array has the shape (3,n).
    If both parameters are arrays with shapes (m,) and (n,), the returned array has the shape (3,n,m).
    """
    t,e = np.array(tth)*pi/180, np.array(eta)*pi/180
    if len(t.shape)==1 and len(e.shape)==1:
        t,e = np.meshgrid(t,e)
    
    S = np.array([-sin(t)*cos(e),
                  -sin(t)*sin(e),
                  cos(t)*np.ones(e.shape)]) # ensure same shape for all entries
    return S


def getN(omega,chi,phi,N0=[0,0,1]):
    """Return the sample normal unit vector in lab coordinates. Assumes the normal is along the beam (z-axis) at omega, chi, phi = 0"""
    N0 = np.array(N0)
    return np.matmul(R_yxy(omega, chi, phi).T,N0).T


def getPixelCoords(pname,danmax_convention=True,corners=False):
    """
    Return the pixel coordinates in meter for a given PONI configuration
    
    Parameters:
        pname             - PONI file path
        danmax_convention - (default=True) return the coordinates in the danmax laboratory convention,
                            otherwise use the default PyFAI convention
        corners           - (default=False) If True, return the coordinates of the pixel corners rather then the centers
    Return:
        xyz - (danmax) numpy array of shape [(x,y,z),h,w] - Index [:,0,0] corresponds to the top left corner of the detector image
        OR
        xyz - (danmax,corners) numpy array of shape [(x,y,z),h+1,w+1] - Index [:,0,0] corresponds to the top left corner of the detector image
        OR
        xyz - (PyFAI) numpy array of shape [h,w,(z,y,x)] *see help(pyFAI.geometry.core.Geometry.position_array)
    """
    # read PONI and get xyz positions (in m)
    poni = geometry.core.ponifile.PoniFile(pname).as_dict()
    geo = geometry.core.Geometry(**{key:poni[key] for key in ['dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3','detector','wavelength']}) 
    xyz = geo.position_array(corners=corners)
    if danmax_convention:
        if not corners:
            # convert to danmax convention NOTE: Might need to be flipped/rotated
            # [h,w,(z,y,x)] --> [(x,y,z),h,w] 
            xyz = np.transpose(xyz,axes=(2,0,1))[[2,1,0],:,:]
            xyz[[0,1]] *= -1
        else:
            # [h,w,4,(z,y,x)] --> [(x,y,z),(ul,ll,ur,lr),h,w] 
            xyz_c = np.transpose(xyz,axes=(3,2,0,1))[[2,1,0],:,:,:]
            xyz_c[[0,1]] *= -1

            #[(x,y,z),h+1,w+1] 
            xyz = np.full((3,xyz_c.shape[2]+1,xyz_c.shape[3]+1)
                        ,0.)
            # upper left
            xyz[:,:-1,:-1] = xyz_c[:,0,:,:]
            # lower left
            xyz[:,-1,:-1] = xyz_c[:,1,-1,:]
            # upper rigth
            xyz[:,:-1,-1] = xyz_c[:,2,:,-1]
            # lower right
            xyz[:,-1,-1] = xyz_c[:,3,-1,-1]
    return xyz
    

def absorptionCorrection2D(pname,t0,mu,omega=0,chi=0,phi=0,normalize=True):
    """
    2D absorption correction for flat samples
    Return the normalized absorption correction A, such that im_corrected = im*A
    Index [0,0] of A corresponds to the top left corner of the detector image
    
    Parameters:
        pname    - PONI file path
        t0       - sample thickness in mm
        mu       - Absorption coefficient in cm-1
        omega=0  - sample rotation in degrees around y
        chi=0    - sample rotation in degrees around x'
        phi=0    - sample rotation in degrees around y''
    Return
        A        - 2D absorption correction (numpy array) 
    """
    mu = mu*10**-1 # mm-1
    # read PONI and get xyz positions (in m)
    xyz = getPixelCoords(pname) # [(x,y,z),h,w] - Index [:,0,0] corresponds to the top left corner of the detector image

    # define unit vectors
    S0 = getS0()
    S = (xyz / np.sqrt(np.sum(xyz**2,axis=0)))
    N = getN(omega,chi,phi)
    # reciprocal dot product of incident beam and sample normal vectors (1/cos(eta))
    sec_eta = 1/np.dot(S0,N)
    # reciprocal dot product of diffracted beam and sample normal vectors (1/cos(zeta))
    sec_zeta = 1/np.sum((S.T*N),axis=2)
    T = 1/(sec_eta*t0) * (np.exp(-mu*t0*sec_eta) - np.exp(-mu*t0*sec_zeta.T)) / (mu*(sec_zeta.T-sec_eta))
    # exception for the limiting case where sec_eta ~=  sec_zeta
    T[np.abs(sec_eta-sec_zeta.T)<10**-5] = (1/(sec_zeta.T*t0) *t0 *np.exp(-mu*t0*sec_zeta.T))[np.abs(sec_eta-sec_zeta.T)<10**-5]
    A = 1/T
    if normalize:
        A /= A.min()
    return A


def absorptionCorrection1D(tth,t0,mu,normalize=True):
    """
    1D absorption correction for flat samples with the normal parallel to the incident beam
    Return the normalized absorption correction A, such that I_corrected = I*A
    
    Parameters:
        tth      - 2theta values (numpy array)
        t0       - sample thickness in mm
        mu       - Absorption coefficient in cm-1
    Return
        A        - 1D absorption correction (numpy array) 
    """
    mu = mu*10**-1 # mm-1
    
    # define unit vectors
    S0 = getS0()
    S = getS(tth,270) # the azimuthal angle is arbitrarily chosen to correspond to the vertical axis, eta = 270 deg
    N = S0

    sec_zeta = 1/np.dot(S.T,N)
    T = 1/(t0) * (np.exp(-mu*t0) - np.exp(-mu*t0*sec_zeta.T)) / (mu*(sec_zeta.T-1))
    # exception for the limiting case where sec_zeta = 1 (tth=0)
    T[np.abs(1-sec_zeta.T)<10**-5] = np.exp(-mu*t0)
    A = 1/T
    if normalize:
        A /= A.min()
    return A

def colorCylinderAnnotation(ax=None,zoom=0.1,xy=(1.02,0.0)):
    """
    Add a color cylinder annotation to an axis
    Parameters
        ax   - matplotlib axes. Uses plt.gca() if ax=None (default=None)
        zoom - overlay image zoom factor (default=0.1)
        xy   - Annotation coordinates in fractions of the axis (default=(1.02,0.0)).
               Overlay image anchor in lower left corner 
    Return 
        ab   - matplotlib.offsetbox.AnnotationBbox
    """
    cyl = mpimg.imread('cylinder.png')
    ab = AnnotationBbox(OffsetImage(cyl, zoom=zoom), # offset image
                        xy,                  # xy coordinates
                        xycoords='axes fraction',  # coordinate system
                        frameon=False,             # frame
                        box_alignment=(0,0))       # anchor point
    if ax is None:
        ax = plt.gca()
    ax.add_artist(ab)
    return ab