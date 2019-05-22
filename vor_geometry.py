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
from shapely.geometry import Polygon, Point


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



def voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def finite_voronoi(vor, boundary_shape):
    areas = list()
    for p in voronoi_polygons(vor, 200):
        areas.append(p.intersection(boundary_shape).area)
    return np.array(areas)

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
    input:
        liquid: index of particle identified as liquid, np.array
        index: a pair of 2D array , np.array
        ratio: threshold #liquid_neighbor/#neighbor, float
    output:
        vor_liquid: liquid particle id, np.array
        liquid: interfacial particles, np.array
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


def bond_orient(xys, neighbor_dict, fold = 6):
    sort_key = np.sort(neighbor_dict.keys())
    ret = list()
    for key in sort_key:
        neighbor = xys[neighbor_dict[key]]
        disp = neighbor - xys[key]
        angle = np.arctan2(*disp.T)
        ret.append(np.abs(np.exp(1j*fold*angle)))
    return ret

def orient_param(ors, neighbor_dict, fold = 4):
    skey = np.sort(neighbor_dict.keys())
    ret = list()
    for key in skey:
        value = neighbor_dict[key]
        temp = [np.exp(fold*1j*ors[ids]) for ids in value]
        temp.append(np.exp(fold*1j*ors[key]))
        ret.append(np.abs(np.mean(temp)))
    ret = np.asarray(ret)
    return ret
    
def global_orient(ors, fold = 4):
    return np.abs(np.mean(np.exp(fold * 1j * ors)))
    
   
def voronoi_liquid(xys, ors, disp, ids, sidelength, ratio,  boundary_shape, hist = False, plot_area = 1, radial_mask = []):
    """
    Input:
        xys, position of particles: N*2 array
        ors, orientation of particles: N array
        ids, an array of True or False mask, if True we think it's solid, False otherwise
        ratio: threshold #liquid_neighbor/#neighbor, float
    Output:
        densities: a float number 
        temp: voronoi area of each particle
        order: a float number means molecular order
    """
    vor = Voronoi(xys)
    #temp = calculate_area(vor)
    temp = finite_voronoi(vor, boundary_shape)
    t1 = temp <= 1200
    t2 = temp > 200
    #reverse_mask = 1 - radial_mask
    # this threshold can exclude boundary particles
    threshold = t1&t2
    # id of liquid
    liquid_id = list(np.where(np.array(np.array(ids) + radial_mask) == 0)[0])
    # dictionary: key, pt_id; value, neighbors
    index_dict = build_voronoi_neighbor_dict(vor)
    # vor_liquid is the id for liquids
    vor_liquid, interface = voronoi_neighbor(liquid_id, index_dict, temp,ratio)
    #print(threshold)
    qualified_solid = ids & threshold
    solid_id = np.where(qualified_solid == 1)[0]
    # qualified_solid is the id for solids, interface is id for interfacial particles
    qualified_solid, interface = solid_interface(solid_id, liquid_id, interface, index_dict)
    liquid_area = temp[vor_liquid]
    liquid_density = density_calculation(liquid_area, sidelength)
    # get solid liquid fraction
    ncount = len(qualified_solid) + len(vor_liquid) + len(interface)
    liquid_fraction = len(vor_liquid)/float(ncount)
    solid_fraction = len(qualified_solid)/float(ncount)
    ##########################################################
    # get order parameter for two phases
    ##########################################################
    #orders = orient_param(ors, index_dict)
    #if len(qualified_solid) != 0:
    #    solid_order = np.mean(orders[qualified_solid])
    #else:
    #    solid_order = 0
    #liquid_order = np.mean(orders[vor_liquid])
    if len(qualified_solid) != 0:
        solid_ors = ors[qualified_solid]
        solid_xys = disp[qualified_solid]
        angles = np.arctan2(*[solid_xys[:,1], solid_xys[:,0]])
        cors = angles % (2*np.pi)
        g_solid_ors = solid_ors - cors
        g_solid_order = global_orient(g_solid_ors)
        # solid density cal
        solid_area = temp[qualified_solid]    
        solid_density = density_calculation(solid_area, sidelength)
    else:
        g_solid_order = np.nan
        solid_density = np.nan
    liquid_ors = ors[vor_liquid]
    g_liquid_order = global_orient(liquid_ors)

    if hist:
        ids = np.asarray(ids)
        fig, ax = plt.subplots()
        ax.hist(temp[ids], 50, range=[500, 1500], log=True)
        ax.hist(temp[~ids], 50, range=[500, 1500], log=True)
        plt.show()
    if plot_area:
        xys = np.array(xys)
        fig_v, ax_v = plt.subplots(2,1,figsize = (15,15*2))
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
        #voronoi_plot_2d(vor, ax = ax_v[2], show_points = False, show_vertices = False)
        #thre = temp < 5000
        #im1 = ax_v[2].scatter(xys[thre][:,0], xys[thre][:,1], c = temp[thre], cmap = 'Paired')
        #cax0 = fig_v.add_axes([0.1,0.9,0.7,0.025])
        #cax1 = fig_v.add_axes([0.1,0.35,0.7,0.02])
        #fig_v.colorbar(im0, cax = cax0, orientation = 'horizontal')
        #fig_v.colorbar(im1, cax = cax1, orientation = 'horizontal')
        plt.show()   
    return liquid_density, solid_density, temp, g_liquid_order, g_solid_order, solid_fraction, liquid_fraction
    
