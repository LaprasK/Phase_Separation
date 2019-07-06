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



def voronoi_polygons(voronoi, diameter, boundary):
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
    
    ret_areas = list()
    inter_polygon = list()
    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            single_poly = Polygon(voronoi.vertices[region])
            inter_polygon.append(single_poly.intersection(boundary))
            ret_areas.append(single_poly.intersection(boundary).area)
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
        single_poly = Polygon(np.concatenate((finite_part, extra_edge)))
        inter_polygon.append(single_poly.intersection(boundary))
        ret_areas.append(single_poly.intersection(boundary).area)
    return inter_polygon, np.array(ret_areas)
    

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

def liquid_interface(liquid, index, area, ratio = 0.75):
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


def local_bond_orient(xys, neighbor_dict, fold = 6):
    '''
    return local bond orientation order parameter
    '''
    sort_key = np.sort(neighbor_dict.keys())
    ret = list()
    for key in sort_key:
        neighbor = xys[neighbor_dict[key]]
        disp = neighbor - xys[key]
        angle = np.arctan2(*disp.T)
        ret.append(np.abs(np.exp(1j*fold*angle)))
    return ret

def orient_param(ors, neighbor_dict, fold = 4):
    '''
    return local orientation order parameter
    '''
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
    ret_dict = dict()
    vor = Voronoi(xys)
    #temp = calculate_area(vor)
    #temp = finite_voronoi(vor, boundary_shape)
    polygons, temp = voronoi_polygons(vor, 300, boundary_shape)
    ret_dict['area'] = temp
    #t1 = temp <= 1200
    t2 = temp > 200
    #reverse_mask = 1 - radial_mask
    # this threshold can exclude boundary particles
    threshold = t2
    # id of liquid
    #liquid_id = list(np.where(np.array(np.array(ids) + radial_mask) == 0)[0])
    liquid_id = list(np.where(np.array(np.array(ids)) == 0)[0])
    # dictionary: key, pt_id; value, neighbors
    index_dict = build_voronoi_neighbor_dict(vor)
    # vor_liquid is the id for liquids
    vor_liquid, interface = liquid_interface(liquid_id, index_dict, temp,ratio)
    #print(threshold)
    qualified_solid = ids & threshold
    solid_id = np.where(qualified_solid == 1)[0]
    # qualified_solid is the id for solids, interface is id for interfacial particles
    qualified_solid, interface = solid_interface(solid_id, liquid_id, interface, index_dict)
    local_bond_6 = local_bond_orient(xys, index_dict, fold = 6)
    local_bond_4 = local_bond_orient(xys, index_dict, fold = 4)
    local_orient = orient_param(ors, index_dict, 4)
    if len(vor_liquid) != 0:
        liquid_area = temp[vor_liquid]
        liquid_density = density_calculation(liquid_area, sidelength)
        liquid_ors = ors[vor_liquid]
        g_liquid_order = global_orient(liquid_ors)
    else:
        liquid_density = np.nan
        g_liquid_order = np.nan
    ret_dict['liquid_density'] = liquid_density
    ret_dict['liquid_order'] = g_liquid_order
    # get solid liquid fraction
    ncount = len(qualified_solid) + len(vor_liquid) + len(interface)
    liquid_fraction = len(vor_liquid)/float(ncount)
    solid_fraction = len(qualified_solid)/float(ncount)
    ret_dict['ids'] = vor_liquid, qualified_solid, interface
    ret_dict['liquid_fraction'] = liquid_fraction
    ret_dict['solid_fraction'] = solid_fraction
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
        # solid density calculation
        solid_area = temp[qualified_solid]    
        solid_density = density_calculation(solid_area, sidelength)
        # solid local bond
        #local_bond_orient(solid_xys, , fold = 6)
    else:
        g_solid_order = np.nan
        solid_density = np.nan
    ret_dict['solid_density'] = solid_density
    ret_dict['solid_order'] = g_solid_order
    


    if hist:
        ids = np.asarray(ids)
        fig, ax = plt.subplots()
        ax.hist(temp[ids], 50,  log=True)
        ax.hist(temp[~ids], 50, log=True)
        plt.show()
    if plot_area:
        xys = np.array(xys)
        fig_v, ax_v = plt.subplots(2,1,figsize = (15,15*2))
        #voronoi_plot_2d(vor, ax = ax_v[0], show_points = False, show_vertices = False)
        for p in polygons:
            x, y = zip(*p.exterior.coords)
            ax_v[0].plot(x,y,'k-')
            ax_v[1].plot(x,y,'k-')
        ax_v[0].scatter(xys[:,0], xys[:,1], c = np.array(ids), cmap = 'Paired')
        #voronoi_plot_2d(vor, ax = ax_v[1], show_points = False, show_vertices = False)
        
        
        
        if np.any(interface):
            ax_v[1].scatter(xys[:,0][interface], xys[:,1][interface], c = 'yellow')
        if np.sum(qualified_solid):
            ax_v[1].scatter(xys[:,0][qualified_solid], xys[:,1][qualified_solid], c = '#D2691E')
        if len(vor_liquid)!= 0:
            ax_v[1].scatter(xys[:,0][vor_liquid], xys[:,1][vor_liquid], c = 'green')
    
        
        #voronoi_plot_2d(vor, ax = ax_v[2], show_points = False, show_vertices = False)
        #thre = temp < 5000
        #im1 = ax_v[2].scatter(xys[thre][:,0], xys[thre][:,1], c = temp[thre], cmap = 'Paired')
        #cax0 = fig_v.add_axes([0.1,0.9,0.7,0.025])
        #cax1 = fig_v.add_axes([0.1,0.35,0.7,0.02])
        #fig_v.colorbar(im0, cax = cax0, orientation = 'horizontal')
        #fig_v.colorbar(im1, cax = cax1, orientation = 'horizontal')
        plt.show()
    return ret_dict
    #return liquid_density, solid_density, temp, g_liquid_order, g_solid_order, solid_fraction, liquid_fraction


class vor_particle:
    def __init__(self, xys, ors, disp, dynamic_solid, sidelength, ratio, boundary_shape, \
                 radial_mask = []):
        self.xys = xys
        self.ors = ors
        self.disp = disp
        self.dynamic_solid = dynamic_solid
        self.sidelength = sidelength
        self.ratio = ratio
        self.boundary_shape = boundary_shape
        self.radial_mask = radial_mask
        self.solid_liquid_interface()
        self.order_paras()
        self.solid_local_bond6, self.solid_local_bond4, self.solid_local_mole6, self.solid_local_mole4 = self.local_orders(self.solid_id)
        self.liquid_local_bond6, self.liquid_local_bond4, self.liquid_local_mole6, self.liquid_local_mole4 = self.local_orders(self.vor_liquid)
        self.inter_local_bond6, self.inter_local_bond4, self.inter_local_mole6, self.inter_local_mole4 = self.local_orders(self.interface)
        
    def solid_liquid_interface(self):
        #self.ret_dict = dict()
        self.vor = Voronoi(self.xys)
        self.polygons, self.areas = voronoi_polygons(self.vor, 300, self.boundary_shape)
        area_criteria = self.areas > 200
        liquid_id = list(np.where(np.array(np.array(self.dynamic_solid)) == 0)[0])
        self.neighbor_dict = build_voronoi_neighbor_dict(self.vor)
        self.vor_liquid, interface = liquid_interface(liquid_id, self.neighbor_dict, area_criteria, self.ratio)
        qualified_solid = self.dynamic_solid & area_criteria
        solid_id = np.where(qualified_solid == 1)[0]
        self.solid_id, self.interface = solid_interface(solid_id, liquid_id, interface, self.neighbor_dict)
        
    def order_paras(self):
        # liquid orders    
        if len(self.vor_liquid) != 0:
            self.liquid_area = self.areas[self.vor_liquid]
            self.liquid_density = density_calculation(self.liquid_area, self.sidelength)
            self.global_liquid_order = global_orient(self.ors[self.vor_liquid])
        else:
            self.liquid_density = np.nan
            self.global_liquid_order = np.nan
        # solid orders
        
        if len(self.solid_id) != 0:
            solid_ors = self.ors[self.solid_id]
            solid_disp = self.disp[self.solid_id]
            angles = np.arctan2(*[solid_disp[:,1], solid_disp[:,0]])
            cors = angles % (2*np.pi)
            global_solid_ors = solid_ors - cors
            self.global_solid_order = global_orient(global_solid_ors)
            self.solid_area = self.areas[self.solid_id]
            self.solid_density = density_calculation(self.solid_area, self.sidelength)
        else:
            self.solid_density = np.nan
            self.global_solid_order = np.nan              
            
        self.ncount = len(self.solid_id) + len(self.interface) + len(self.vor_liquid)
        self.liquid_fraction = len(self.vor_liquid)/self.ncount
        self.solid_fraction = len(self.solid_id)/self.ncount

        
    def local_orders(self, phase_id):
        sort_key = np.sort(phase_id)
        local_bond6 = list()
        local_bond4 = list()
        local_mole6 = list()
        local_mole4 = list()
        for key in sort_key:
            neighbors = np.array(self.neighbor_dict[key])
            mask = np.isin(neighbors,phase_id)
            #bond order
            disp = self.xys[neighbors][mask] - self.xys[key]
            angle = np.arctan2(*disp.T)
            local_bond6.append(np.abs(np.mean(np.exp(1j*6*angle))))
            local_bond4.append(np.abs(np.mean(np.exp(1j*4*angle))))
            #molecular order
            neighbor_ors6 = [np.exp(1j*6*self.ors[ids]) for ids in neighbors[mask]]
            neighbor_ors6.append(np.exp(6*1j*self.ors[key]))
            neighbor_ors4 = [np.exp(1j*4*self.ors[ids]) for ids in neighbors[mask]]
            neighbor_ors4.append(np.exp(4*1j*self.ors[key]))
            local_mole6.append(np.abs(np.mean(neighbor_ors6)))
            local_mole4.append(np.abs(np.mean(neighbor_ors4)))
        return np.array(local_bond6), np.array(local_bond4), np.array(local_mole6), np.array(local_mole4)
    
    

        '''
        # solid local orders
        solid_vor = Voronoi(self.xys[self.solid_id])
        solid_neighbor_dict = build_voronoi_neighbor_dict(solid_vor)
        self.solid_6bond_orders = local_bond_orient(self.xys[self.solid_id], solid_neighbor_dict, fold= 6)
        self.local_solid_molecular = orient_param(self.ors[self.solid_id], solid_neighbor_dict, fold = 4)
        
        # liquid local orders
        liquid_vor = Voronoi(self.xys[self.vor_liquid])
        liquid_neighbor_dict = build_voronoi_neighbor_dict(liquid_vor)
        self.liquid_6bond_orders = local_bond_orient(self.xys[self.vor_liquid], liquid_neighbor_dict, fold = 6)
        self.local_liquid_molecular = orient_param(self.ors[self.vor_liq uid], liquid_neighbor_dict, fold = 4)
        '''
        
    def plot_order(self, case):
        xys = np.array(self.xys)
        fig_v, ax_v = plt.subplots(1,1,figsize = (15,15))
        if case == 'bond6':
            colors = [self.inter_local_bond6, self.solid_local_bond6, self.liquid_local_bond6]
        elif case == 'bond4':
            colors = [self.inter_local_bond4, self.solid_local_bond4, self.liquid_local_bond4]
        elif case == 'mole6':
            colors = [self.inter_local_mole6, self.solid_local_mole6, self.liquid_local_mole6]
        elif case == 'mole4':
            colors = [self.inter_local_mole4, self.solid_local_mole4, self.liquid_local_mole4]
        for p in self.polygons:
            x, y = zip(*p.exterior.coords)
            ax_v.plot(x,y,'k-')
            #ax_v[1].plot(x,y,'k-')
        # plot defined solid, liquid and interface
        if np.any(self.interface):
            ax_v.scatter(xys[:,0][self.interface], xys[:,1][self.interface], c = colors[0])
        if np.sum(self.solid_id):
            ax_v.scatter(xys[:,0][self.solid_id], xys[:,1][self.solid_id], c = colors[1])
        if len(self.vor_liquid)!= 0:
            ax_v.scatter(xys[:,0][self.vor_liquid], xys[:,1][self.vor_liquid], c = colors[2])
        #ax_v[1].scatter(xys[:,0], xys[:,1], c = np.array(self.dynamic_solid), cmap = 'Paired')
        plt.show()
        return
        
    def plot_voronoi(self):
        xys = np.array(self.xys)
        fig_v, ax_v = plt.subplots(1,1,figsize = (15,15))
        for p in self.polygons:
            x, y = zip(*p.exterior.coords)
            ax_v.plot(x,y,'k-')
            #ax_v[1].plot(x,y,'k-')
        # plot defined solid, liquid and interface
        if np.any(self.interface):
            ax_v.scatter(xys[:,0][self.interface], xys[:,1][self.interface], c = 'yellow')
        if np.sum(self.solid_id):
            ax_v.scatter(xys[:,0][self.solid_id], xys[:,1][self.solid_id], c = '#D2691E')
        if len(self.vor_liquid)!= 0:
            ax_v.scatter(xys[:,0][self.vor_liquid], xys[:,1][self.vor_liquid], c = 'green')
        #ax_v[1].scatter(xys[:,0], xys[:,1], c = np.array(self.dynamic_solid), cmap = 'Paired')
        plt.show()
        return 