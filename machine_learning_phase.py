#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:16:20 2019

@author: zhejun
"""

import helpy
import ring_motion
import numpy as np
import matplotlib.pyplot as plt
import tracks
from scipy.spatial import cKDTree as KDTree
from vor_geometry import *
import velocity
from collections import defaultdict
import os
from shapely.geometry import Point


class phase_coex(object):
    def __init__(self, prefix, number_config = 200, config_len = 50, real_particle = 0.2, fps = 2.5, \
                 nearest_neighbor_number = 5,  plot_check = False, test_layer = 2, vomega = 0.12):
        self.prefix = prefix
        self.number_config = number_config
        self.config_len = config_len
        self.real_particle = real_particle
        self.fps = fps
        self.nnn = nearest_neighbor_number
        self.system_area = np.pi * 4 **2
        self.result = defaultdict(list)
        self.load_and_process()
        parent_direct = os.path.abspath(os.path.join(self.prefix, os.pardir))
        self.plot_check = plot_check
        self.vomega_criteria = vomega
        self.test_layer = test_layer
        self.save_name = parent_direct + '/config_vdata.npy'
        self.result_name = parent_direct + '/phase_data.npy'
        vdata_file = self.save_name
        if os.path.isfile(vdata_file):
            self.config_vdata = np.load(vdata_file).item()
            if len(self.config_vdata.keys()) != number_config:
                self.build_config_vdata()
        else:
            self.config_vdata = dict()
            self.build_config_vdata()
        
    def load_and_process(self):
        self.pdata = helpy.load_data(self.prefix, 'p')
        self.x0, self.y0, self.R = ring_motion.boundary(self.prefix)
        self.side_len = self.R * self.real_particle /4.0  ##4inch in experiment 
        self.max_dist = self.side_len/1.25        
        self.frames = self.pdata['f']
        self.boundary_shape = Point(self.y0, 1024 - self.x0).buffer(self.R)
        
        
    def track_config(self, pdata, startframe):
        pfsets = helpy.load_framesets(pdata)
        pftrees = {f: KDTree(helpy.consecutive_fields_view(pfset, 'xy'),
                             leafsize=32) for f, pfset in pfsets.iteritems()}
        trackids = tracks.find_tracks(pdata, maxdist= self.max_dist, giveup = 10, \
                                      n = 0, stub = 20, pftrees = pftrees, \
                                      pfsets = pfsets, startframe = startframe)
        trackids = tracks.remove_duplicates(trackids, data = pdata)
        return_data = helpy.initialize_tdata(pdata, trackids, pdata['o'])
        return_data = helpy.add_self_view(return_data, ('x','y'),'xy')
        return return_data

    
    def build_config_vdata(self):
        for i in range(0, self.number_config):
            startframe = i * self.config_len
            mask = (self.frames >= startframe) & (self.frames < (i+1)*self.config_len)
            config_pdata = self.pdata[mask]
            config_tdata = self.track_config(config_pdata, startframe)
            tracksets = helpy.load_tracksets(config_tdata, run_track_orient = False, min_length = self.config_len//2, \
                                             run_repair = 'interp')
            #
            for pid, par_data in tracksets.items():
                par_data['o'] = self.fix_orient(tracksets, pid)
            track_prefix = {self.prefix: tracksets}
            v_data = velocity.compile_noise(track_prefix, width=(0.525,), cat = False, side = self.side_len, \
                                        fps = self.fps, ring = True, x0 = self.x0, y0 = self.y0, skip = 1, \
                                        grad = False, start = 0)
            v_data = v_data[self.prefix]
            self.config_vdata[startframe] = v_data
        np.save(self.save_name, self.config_vdata)
        return

    def smooth_orient(self, ret):
        for i in range(len(ret)-1):
            if np.isnan(ret[i+1]):
                ret[i+1] = ret[i]
            vorient = ret[i+1] - ret[i]
            #print(vorient)
            if np.abs(vorient) < np.pi/6:
                continue
            elif np.abs(vorient) > np.pi/3:
                count = vorient // (np.pi/2)
                if count == 0:
                    ret[i+1] = ret[i+1] - np.sign(vorient)*np.pi/2
                else:
                    ret[i+1] = ret[i+1] - count * np.pi/2
                vo2 = ret[i+1] - ret[i]
                if np.abs(vo2) < np.pi/6:
                    continue
                elif np.abs(vo2) > np.pi/3:
                    ret[i+1] -= np.pi/2*np.sign(vo2)
                else:
                    ret[i+1] -= np.pi/4*np.sign(vo2)
        return ret
    
    def fix_orient(self, orient, key):
        orient = orient[key]['o']
        ret = orient.copy()
        # if the first detection is correct
        ret = self.smooth_orient(ret)
        match_rate = float(np.sum(ret == orient))/len(orient)
        v0 = np.abs(orient[1] - orient[0])
        bad = (v0 > np.pi/6) & (v0 < np.deg2rad(320))
        if (match_rate < 0.5) and bad:
            ret = orient.copy()[::-1]
            ret = self.smooth_orient(ret)[::-1]
        match_rate = float(np.sum(ret == orient))/len(orient)
        good = (ret < np.deg2rad(350)) & (ret > np.deg2rad(10))
        good = np.sum(good) > 35
        match_rate = float(np.sum(ret == orient))/len(orient)
        if good & (match_rate < 0.5):
            ret = orient.copy()[::-1]
            ret = self.smooth_orient(ret)[::-1]        
        return ret
    
    def single_config_detect(self, v_data):
        '''
        output:
        qualify_id: if order_parameter is np.nan, exclude that particle, length equals detected particles in\
                    the continuous video
        '''
        qualify_id, order_para_mean  = list(), list()
        vr_mean, vomega_list = list(), list()
        for pid, pid_data in v_data.items():
            center_orient = pid_data['corient']
            particle_orient = pid_data['o']
            cen_or_vec =  np.asarray([np.cos(center_orient), np.sin(center_orient)]).T
            particle_or_vec = np.asarray([np.cos(particle_orient), np.sin(particle_orient)]).T
            product = np.abs([np.dot(cen_or_vec[j], particle_or_vec[j]) for j in range(len(cen_or_vec))])
            # exclude all order parameter is np.nan
            if np.sum(np.isnan(product)) == len(cen_or_vec):
                continue
            qualify_id.append(pid)
            # order parameter
            order_para_mean.append(np.nanmean(product))
            vr_mean.append(np.nanmean(pid_data['vradi']))
            vomega_list.append(np.nanmean(np.abs(pid_data['vomega'])))
        qualify_id = np.asarray(qualify_id)
        order_para_mean = np.asarray(order_para_mean)
        vr_mean = np.asarray(vr_mean)
        vomega_list = np.asarray(vomega_list)
        return qualify_id, order_para_mean, vr_mean, vomega_list


    def solid_criteria(self, order, vr, vomega):
        # order parameter criteria
        order_mask = (order >= 1/np.sqrt(2) - 0.12) & (order <= 1/np.sqrt(2) + 0.12)
        # vr criteria
        solid_vr_mean = np.mean(vr)
        solid_vr_std = np.std(vr)
        if solid_vr_std < 0.006:
            solid_vr_std = 0.006  
        vr_mask = (vr >= solid_vr_mean - 1.5 * solid_vr_std) & (vr <= solid_vr_mean + 1.5 * solid_vr_std)
        #vomega criteria
        vomega_mask = vomega < self.vomega_criteria
        return order_mask, vr_mask, vomega_mask
        
    
    def phase_detection(self):
        self.solids = []
        self.detect = []
        self.liquid_density = list()
        self.solid_density = list()
        self.solid_fraction = list()
        self.liquid_fraction = list()
        self.solid_molecular_order = list()
        self.liquid_molecular_order = list()
        self.xys = dict()
        self.final_id = dict()
        self.solid_vor_area = np.empty(0)
        self.liquid_vor_area = np.empty(0)
        plot_number = 0
        sorted_keys = sorted(self.config_vdata.keys())
        for startframe in sorted_keys:                
            #pids = v_data.keys()
            v_data = self.config_vdata[startframe]
            qualify_id, order_para_mean, vr_mean, vomega_list = self.single_config_detect(v_data)
            self.detect.append(len(qualify_id))
            qualify_id_set = set(qualify_id)            
            order_mask, vr_mask, vomega_mask = self.solid_criteria(order_para_mean, vr_mean, vomega_list)
            
            fdata = helpy.load_framesets(v_data)              
                
            
            # find the frame where contains all the qualified particles
            count = 0
            while (count < 49) & (not set(fdata[startframe+count]['t']).issuperset(qualify_id_set)):
                count+=1
                if count == 49:
                    break
            startframe += count
            # for the idtentified frame, make TRUE if t in qualified_id
            fdata_track = fdata[startframe]['t']
            track_mask = list()
            for t in fdata_track:
                track_mask.append(t in qualify_id)
            track_mask = np.asarray(track_mask)            
            #build KDTree to query the nearest neighbor
            xys = helpy.consecutive_fields_view(fdata[startframe][track_mask], 'xy')
            ors = helpy.consecutive_fields_view(fdata[startframe][track_mask], 'o')
            disp = xys - [self.x0, self.y0] # displacement to the center
            radial = np.hypot(*disp.T)
            criteria = self.R - 1.4*self.side_len
            radial_mask = radial >= criteria
            #switch x, y coordinate into the regular orientation
            xys = xys[:,::-1]
            xys[:,1] = 1024 - xys[:,1]
            self.xys[startframe] = xys
            
            ftree = KDTree(xys, leafsize = 16)
            #####################################################################
            # 1st iteration, 
            # find at least two particles within 1.5 particle size radius
            # 3 of your neighbor must satisfy vr_criteria.
            # need to meet vomega criteria 
            #####################################################################           
            final_mask = []       

            for pt_id in range(len(xys)):
                if not vr_mask[pt_id]:
                    final_mask.append(False)
                    continue
                dists, ids = ftree.query(xys[pt_id], self.nnn)
                #if np.all(dists < self.side_len * 2.0):
                if np.sum(dists < self.side_len*1.5) > 2:
                    final_mask.append(np.sum(vr_mask[ids]) > 3)
                else:
                    final_mask.append(False)
            temp_mask = np.array(final_mask) & np.array(vomega_mask)
            
                        
            ##############################################################################
            # if you neighbors qualified then you will be solid ,exclude detection error
            # 
            # qualified_id is a True or False mask
            ##############################################################################
            qualified_solid = list()
            for pt_id in range(len(xys)):
                dists, ids = ftree.query(xys[pt_id], self.nnn)
                qualified_solid.append(temp_mask[pt_id] or np.sum(temp_mask[ids[1:]]) >= 3)
            
            self.final_id[startframe] = qualified_solid    
            solid_number = np.sum(qualified_solid)
            self.solids.append(solid_number) 
            
            
                       
            plot_vor = startframe < 550
            rho_liquid, rho_solid = self.density_calculation(solid_number, len(qualify_id), xys, ors, disp,\
                                                             qualified_solid, plot_vor, radial_mask)
            #print(qualified_solid)
            self.liquid_density.append(rho_liquid)
            self.solid_density.append(rho_solid)
            
            if plot_number < self.plot_check:
                xs = helpy.consecutive_fields_view(fdata[startframe][track_mask],'x')
                ys = helpy.consecutive_fields_view(fdata[startframe][track_mask],'y')
                self.plot_check_solid(xs, ys, vr_mean, vr_mask, order_para_mean, \
                                      order_mask,vomega_list, vomega_mask, final_mask,\
                                      qualified_solid)
                plot_number += 1
        self.save_phase()
        return len(qualify_id)
        
    
    def density_calculation(self, number_of_solids, total_number, xys, ors, disp, ids, plot_vor, radial_mask):
        solids_area =  self.real_particle ** 2 * number_of_solids / 0.95347
        fraction = solids_area/self.system_area
        #if fraction < 0.1:
        if False:
            liquid_area = self.system_area - solids_area
            number_liquid = total_number - number_of_solids
            rho_liquid = number_liquid * self.real_particle ** 2/ float(liquid_area)
            rho_solid = np.nan
            inside_total = np.sum(radial_mask)
            if np.sum(radial_mask & ids) >= 2:
                xy_inside = disp[radial_mask & ids]
                or_inside = ors[radial_mask & ids]
                angles = np.arctan2(*[xy_inside[:,1], xy_inside[:,0]])
                cors = angles % (2*np.pi)
                solid_ors = or_inside - cors
                sorder = global_orient(solid_ors)
            else:
                sorder = np.nan
            inside_mask = np.invert(ids) & radial_mask
            inside_liquid = ors[inside_mask]
            self.liquid_molecular_order.append(global_orient(inside_liquid))
            self.liquid_fraction.append(len(inside_liquid)/float(inside_total))
            self.solid_fraction.append(1-len(inside_liquid)/float(inside_total))
            self.solid_molecular_order.append(sorder)
        else:
            hist = False
            ratio = 0.5 if fraction > 0.85 else 0.75
            rho_liquid, rho_solid, varea, liquid_order, solid_order, solid_fraction, liquid_fraction = \
                voronoi_liquid(xys, ors, disp, ids, self.side_len, ratio, self.boundary_shape, hist, plot_vor, radial_mask)
            ids = np.asarray(ids)
            self.solid_molecular_order.append(solid_order)
            self.liquid_molecular_order.append(liquid_order)
            self.solid_vor_area = np.append(self.solid_vor_area, varea[ids])
            self.liquid_vor_area = np.append(self.liquid_vor_area, varea[~ids])
            self.solid_fraction.append(solid_fraction)
            self.liquid_fraction.append(liquid_fraction)
        return rho_liquid, rho_solid
        
    
    def plot_check_solid(self,xs,ys, vr_mean, vr_mask, dot_mean_list, order_mask, \
                         vomega_list, vomega_mask, final_mask, qualified_solid):
        #fig_omega, ax_omga = plt.subplots(figsize = (5,5))
        #omga_img = ax_omga.scatter(ys,1024 - xs, c = vomega_list, cmap = 'Paired')
        #fig_omega.colorbar(omga_img)
        solid_fig, solid_ax = plt.subplots(figsize = (10,10))
        solid_ax.scatter(ys, 1024-xs, c = qualified_solid, cmap = 'Paired')
        solid_ax.set_xticks([])
        solid_ax.set_yticks([])
        solid_fig.savefig(self.prefix + "_solid"+str(len(qualified_solid))+".pdf",\
                          dpi = 400, bbox_inches = 'tight')
        plt.show()
        return

    
    def get_solid_density(self, fdata):
        inner = self.R - self.test_layer * self.side_len
        total_area = np.pi*(self.R**2 - inner ** 2)
        solid_density = list()
        for frame, framedata in fdata.items():
            qualify_mask = (framedata['r'] > inner)
            qualify_mask = qualify_mask & (framedata['r'] < self.R)
            mask_sum = np.sum(qualify_mask)
            solid_density.append(mask_sum * self.side_len ** 2/ total_area)
        return np.mean(solid_density)
            
    
    def save_phase(self):
        result = dict()
        result['liquid_density'] = self.liquid_density
        result['liquid_order'] = self.liquid_molecular_order
        result['solid_density'] = self.solid_density
        result['solid_order'] = self.solid_molecular_order
        result['solid_fraction'] = self.solid_fraction
        result['liquid_fraction'] = self.liquid_fraction
        np.save(self.result_name, result)
        return 
        
