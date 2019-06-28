# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:44:42 2018

@author: Ihor Sahalianov
"""
import numpy as np
import math
import timeit

def get_value_from_input_line(line):
    """
    Method splits an input line 
    with ": " separator and returns value in string format
    """
    line = line.split(': ')
    return (str)(line[1])

def get_input_data():
    """
    Get all input data from the "capacitance__input.txt" and transform it into a.u.
    """
    with open('./capacitance_input.txt', 'r') as f:  
        lx       = float(get_value_from_input_line (f.readline()))
        ly       = float(get_value_from_input_line (f.readline()))
        lz       = float(get_value_from_input_line (f.readline()))
        box_plus = float(get_value_from_input_line (f.readline()))
        charge   = int  (get_value_from_input_line (f.readline()))
        n_atoms  = int  (get_value_from_input_line (f.readline()))
        cutoff   = float(get_value_from_input_line (f.readline()))

    #All distances are transformed into Bohr
    lx *= 1.8897
    ly *= 1.8897
    lz *= 1.8897
    
    return lx, ly, lz, box_plus, charge, n_atoms, cutoff

def get_atoms_data(n_atoms, cutoff):
    """
    Get all atoms data from the cube file: 
        x, y, z and VdW radius of each atom 
    All data is converted into a.u.
    All VdW values are multiplied by the input scaled factor "cutoff"
    """
    ax   = np.zeros(n_atoms).astype(np.float64)
    ay   = np.zeros(n_atoms).astype(np.float64)
    az   = np.zeros(n_atoms).astype(np.float64)
    ac   = np.zeros(n_atoms).astype(np.float64)
    
    vdw_h  = 1.2      #VdW radius of H
    vdw_c  = 1.7      #VdW radius of C
    vdw_o  = 1.52     #VdW radius of O
    vdw_s  = 1.8      #VdW radius of S
    vdw_cl = 1.75     #VdW radius of Cl

    vdw_h  *= 1.8897
    vdw_c  *= 1.8897
    vdw_o  *= 1.8897
    vdw_s  *= 1.8897
    vdw_cl *= 1.8897

    with open('./cube_potential.cube', 'r') as f:  
        line = f.readline()
        line = f.readline()
        line = f.readline()
        data = line.split()
        sx = float(data[1])
        sy = float(data[2])
        sz = float(data[3])

        line = f.readline()
        data = line.split()
        nx = int(data[0])   
        dx = float(data[1])    
        line = f.readline()
        data = line.split()
        ny = int(data[0])   
        dy = float(data[2])    
        line = f.readline()
        data = line.split()
        nz = int(data[0])   
        dz = float(data[3])    

        for i in range(n_atoms):
            line = f.readline()
            data = line.split()
            
            ac[i] = float(data[0])
            ax[i] = float(data[2])
            ay[i] = float(data[3])
            az[i] = float(data[4])
            
            if(ac[i] == 1):
                ac[i] = vdw_h  * cutoff
            if(ac[i] == 6):
                ac[i] = vdw_c  * cutoff
            if(ac[i] == 8):
                ac[i] = vdw_o  * cutoff
            if(ac[i] == 16):
                ac[i] = vdw_s  * cutoff
            if(ac[i] == 17):
                ac[i] = vdw_cl * cutoff
     
    n = nx*ny*nz    

    return n, nx, ny, nz, ax, ay, az, ac, dx, dy, dz, sx, sy, sz

def get_potential_data(n):
    """
    Get potantial data: x, y, z and cutoff radius of each atom 
    All data is in a.u.
    """
    p  = []

    with open('./cube_potential.cube', 'r') as f: 
        for i in range(6 + n_atoms):
            next(f)
        for line in f:
            data = line.split()
            for j in range(len(data)):
                p.append(float(data[j]))
    
    return p

def print_work_time(time_start):
    """
    Calculating of the elapsed time
    """
    def get_work_time(time_start):
        work_time = timeit.default_timer() - time_start
        hours     = int(work_time / 3600)
        minutes   = int((work_time - hours * 3600) / 60)
        seconds   = int(work_time - hours * 3600 - minutes * 60) 
        return hours, minutes, seconds
    hours, minutes, seconds = get_work_time(time_start)
    return hours, minutes, seconds
 
def get_ex_ey_ez(p, n1, n2x, n2y, n2z, dx, dy, dz):
    """
    differentiation of the electrostatic potential (ESP)
    ex, ey, ez - componets of electric field (gradient of the ESP)
    """
    ex = (p[n2x] - p[n1]) / dx
    ey = (p[n2y] - p[n1]) / dy
    ez = (p[n2z] - p[n1]) / dz
    return ex, ey, ez  

def check_dist_between_two_points(x1, y1, z1, 
                                  x2, y2, z2, 
                                  r_cutoff):
    """
    If return 1, then distance between two points is smaller than r_cutoff
    If return 0, then distance between two points is bigger  than r_cutoff
    Square root is too long to calculate for many points.
    Hence, a little bit more complex algorithm is followed below.
    """
    delta_x = x2 - x1
    if(delta_x < 0.0): 
        delta_x *= -1
    if(delta_x > r_cutoff):
        return 0
    delta_y = y2 - y1
    if(delta_y < 0.0): 
        delta_y *= -1
    if(delta_y > r_cutoff):
        return 0 
    delta_z = z2 - z1
    if(delta_z < 0.0): 
        delta_z *= -1
    if(delta_z > r_cutoff):
        return 0
    delta_r = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
    if(delta_r > r_cutoff):
        return 0
    return 1

def check_point_and_all_atoms(x1, y1, z1, 
                              n_atoms, 
                              ax, ay, az, ac): 
    """
    If return 1, then point is not ok with cutoff radii
    If return 0, then point is     ok with cutoff radii   
    """
    for i in range (n_atoms):
        coin = check_dist_between_two_points(x1, y1, z1, 
                                             ax[i], ay[i], az[i], 
                                             ac[i])
        if(coin == 1):
            return 1
    return 0

def check_all_point_and_all_atoms(n, nx, ny, nz, 
                                  dx, dy, dz, 
                                  n_atoms, 
                                  ax, ay, az, ac, 
                                  sx, sy, sz): 
    """
    If return 1, then point is not ok with cutoff radiuses (intersections)
    If return 0, then point is     ok with cutoff radiuses   
    """
    p_check = np.zeros(n).astype(np.int)    
    
    for i in range (nx):
        for j in range (ny):
            for k in range (nz):
                px = sx + i * dx
                py = sy + j * dy
                pz = sz + k * dz
                p_check[i*ny*nz + j*nz + k] = check_point_and_all_atoms(px, py, pz, 
                                                                        n_atoms, 
                                                                        ax, ay, az, ac)
    return p_check

def get_w(nx,ny,nz,dx,dy,dz,p,p_check):
    """
    Calculate total electrostatic energy in Si units
    Volume, occupied by atoms, is omitted
    """
    eps0 = 8.85*1e-12
    dv = dx*dy*dz
    w = 0.0
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                n1  =   i   *ny*nz +   j  *nz +   k
                n2x = (i+1) *ny*nz +   j  *nz +   k
                n2y =   i   *ny*nz + (j+1)*nz +   k
                n2z =   i   *ny*nz +   j  *nz + (k+1)
                if((p_check[n1]  == 0) and 
                   (p_check[n2x] == 0) and 
                   (p_check[n2y] == 0) and 
                   (p_check[n2z] == 0)):
                    ex, ey, ez = get_ex_ey_ez(p, n1, n2x, n2y, n2z, dx, dy, dz)
                    e2 = ex*ex + ey*ey + ez*ez
                    w += e2
                    
    w *= 5.14220652e11 * 5.14220652e11
    w *= dv * 5.2918e-11 * 5.2918e-11 * 5.2918e-11
    w *= eps0 / 2.0
    return w

def get_c(box_plus, lx,ly,lz,charge,w):
    """
    Calculate volumetric capacitance as C = Q^2 / W in F/cm^3
    The size of the simulation box is extracted from the input and enlarged by box_plus
    Capacitance is converted into Si units
    """
    lx += box_plus * 2.0 * 1.8897
    ly += box_plus * 2.0 * 1.8897
    lz += box_plus * 2.0 * 1.8897
    v_sample = lx*ly*lz / (1.8897 * 1.8897 * 1.8897) / (1e8 * 1e8 * 1e8)
    c = charge * charge / (2.0 * w) * (1.6 * 1e-19 * 1.6 * 1e-19) / v_sample
    return c
            


time_start = timeit.default_timer()

lx, ly, lz, box_plus, charge, n_atoms, cutoff = get_input_data()
n, nx, ny, nz, ax, ay, az, ac, dx, dy, dz, sx, sy, sz = get_atoms_data(n_atoms, cutoff)
p = get_potential_data(n)

p_check = check_all_point_and_all_atoms(n, nx, ny, nz, 
                                        dx, dy, dz, 
                                        n_atoms, 
                                        ax, ay, az, ac, 
                                        sx, sy, sz)

w = get_w(nx,ny,nz,dx,dy,dz,p,p_check)
c = get_c(box_plus, lx,ly,lz,charge,w)

hours, minutes, seconds = print_work_time(time_start)
with open('./capacitance_results_output.txt', 'a', encoding='utf-8') as fl:
    print('Vol. capacitance:   ',
          c , '    Energy W:    ', w, 
          file = fl, flush=True)
    print('Total work time:', 
          hours, 'hours', minutes, 'min.', seconds, 'sec.',
          file = fl, flush=True)


