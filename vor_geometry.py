#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:01:39 2019

@author: zhejun
"""
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d 
import matplotlib.pyplot as plt
from collections import defaultdict


def poly_area(corners):
    """calculate area of polygon"""
    area = 0.0
    n = len(corners)
    for i in xrange(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def calculate_area(vor):
    regions = (vor.regions[regi] for regi in vor.point_region)
    return np.array([0 if -1 in reg else poly_area(vor.vertices[reg])
                     for reg in regions])
    
def build_voronoi_ridge_index(vor, liquid_id):    
    index = vor.ridge_points
    index = np.vstack((index, index[:,::-1]))
    index_dict = dict()
    for lid in liquid_id:
        index_dict[lid] = index[index[:,0] == lid][:,1]
    return index_dict

def build_voronoi_neighbor_dict(vor):
    index = vor.ridge_points
    index = np.vstack((index, index[:,::-1]))
    neighbor_dict = defaultdict(list)
    for row in index:
        neighbor_dict[row[0]].append(row[1])
    return dict(neighbor_dict)

def voronoi_neighbor(liquid, index, area, ratio = 0.75):
    """
    liquid: index of particle identified as liquid, np.array
    index: a 2D array 
    """
    vor_liquid = list()
    interface = list()
    for liq in liquid:
        neighbors = index[liq]
        qualified_neighbor = np.isin(neighbors, liquid)
        threshold = np.sum(qualified_neighbor)/float(len(neighbors))
        if area[liq] > 200:
            if threshold >= ratio:
                vor_liquid.append(liq)
            else:
                interface.append(liq)
    return np.array(vor_liquid), np.array(interface)

def solid_interface(solid_id, liquid_id, interface, index_dict):
    interface = list(interface)
    qual_solid = list()
    for sol in solid_id:
        neighbors = index_dict[sol]
        mask = np.isin(neighbors, liquid_id)
        if np.sum(mask)>=2:
            interface.append(sol)
        else:
            qual_solid.append(sol)
    return np.array(qual_solid), np.array(interface)

def density_calculation(area, sidelength):
    total_area = np.sum(area)
    particle_area = len(area) * sidelength ** 2
    return particle_area/float(total_area)

    
def vornoi_liquid(xys, ids, sidelength, ratio,  hist = False, plot_area = True, radial_mask = []):
    vor = Voronoi(xys)
    temp = calculate_area(vor)
    t1 = temp <= 1200
    t2 = temp > 200
    #reverse_mask = 1 - radial_mask
    # this threshold can exclude boundary particles
    threshold = t1 & t2
    # id of liquid
    liquid_id = list(np.where(np.array(np.array(ids) + radial_mask) == 0)[0])
    # dictionary: key, pt id; value, neighbors
    index_dict = build_voronoi_neighbor_dict(vor)
    vor_liquid, interface = voronoi_neighbor(liquid_id, index_dict, temp,ratio)
    qualified_solid = ids & threshold
    solid_id = np.where(qualified_solid == 1)[0]
    qualified_solid, interface = solid_interface(solid_id, liquid_id, interface, index_dict)
    liquid_area = temp[vor_liquid]
    liquid_density = density_calculation(liquid_area, sidelength)
    if np.sum(qualified_solid) == 0:
        solid_density = 0
    else:
        solid_area = temp[qualified_solid]    
        solid_density = density_calculation(solid_area, sidelength)
    if hist:
        ids = np.asarray(ids)
        fig, ax = plt.subplots()
        ax.hist(temp[ids], 50, range=[500, 1500], log=True)
        ax.hist(temp[~ids], 50, range=[500, 1500], log=True)
        plt.show()
    if plot_area:
        xys = np.array(xys)
        fig_v, ax_v = plt.subplots(3,1,figsize = (15,45))
        voronoi_plot_2d(vor, ax = ax_v[0], show_points = False, show_vertices = False)
        ax_v[0].scatter(xys[:,0], xys[:,1], c = np.array(ids), cmap = 'Paired')
        voronoi_plot_2d(vor, ax = ax_v[1], show_points = False, show_vertices = False)
        final_liquid = np.zeros_like(ids)
        final_liquid[vor_liquid] = 1
        final_category = np.zeros(len(ids))
        if np.any(interface):
            final_category[interface] = 0.9
        if np.sum(qualified_solid):
            final_category[qualified_solid] = 1
        final_category[vor_liquid] = 0.3
        ax_v[1].scatter(xys[:,0], xys[:,1], c = final_category, cmap = 'Paired')
        voronoi_plot_2d(vor, ax = ax_v[2], show_points = False, show_vertices = False)
        thre = temp < 5000
        im1 = ax_v[2].scatter(xys[thre][:,0], xys[thre][:,1], c = temp[thre], cmap = 'Paired')
        #cax0 = fig_v.add_axes([0.1,0.9,0.7,0.025])
        cax1 = fig_v.add_axes([0.1,0.35,0.7,0.02])
        #fig_v.colorbar(im0, cax = cax0, orientation = 'horizontal')
        fig_v.colorbar(im1, cax = cax1, orientation = 'horizontal')
        plt.show()    
    return liquid_density, solid_density, temp
    
