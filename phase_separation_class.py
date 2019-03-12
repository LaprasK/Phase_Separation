#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:39:19 2019

@author: zhejunshen
"""
import helpy
import ring_motion
import numpy as np
import matplotlib.pyplot as plt
import tracks
from scipy.spatial import Voronoi, cKDTree as KDTree
import velocity
from collections import defaultdict
import os


def poly_area(corners):
    """calculate area of polygon"""
    area = 0.0
    n = len(corners)
    for i in xrange(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def calculate_area(positions):
    vor = Voronoi(positions)
    regions = (vor.regions[regi] for regi in vor.point_region)
    return np.array([0 if -1 in reg else poly_area(vor.vertices[reg])
                     for reg in regions])
    

def vornoi_liquid(xys, ids, sidelength, hist = False, plot_area = True):
    temp = calculate_area(xys)
    threshold = temp <= 1400
    print(temp)
    if hist:
        ids = np.asarray(ids)
        fig, ax = plt.subplots()
        ax.hist(temp[ids], 50, range=[500, 1500], log=True)
        ax.hist(temp[~ids], 50, range=[500, 1500], log=True)
        plt.show()
    if plot_area:
        xys = np.array(xys)
        ys = xys[threshold][:,1]
        xs = xys[threshold][:,0]        
        f, a = plt.subplots(figsize=(10,10))
        im0 = a.scatter(ys,1024-xs, c = temp[threshold], cmap = 'Paired')
        cax0 = f.add_axes([0.1, 0.9, 0.7, 0.025])
        f.colorbar(im0, cax=cax0, orientation='horizontal')
        plt.show()
    area = temp[~np.array(ids)]
    area = area[area > sidelength**2/0.85]
    number_liquid = len(area)
    total_area = np.sum(area)
    return number_liquid * sidelength ** 2 / total_area
    


class phase_coex:
    def __init__(self, prefix, number_config = 200, config_len = 50, real_particle = 0.2, fps = 2.5, \
                 nearest_neighbor_number = 5,  plot_check = False, solid_den_test = False,\
                 test_layer = 2, vomega = 0.12):
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
        self.solid_den_test = solid_den_test
        self.vomega_criteria = vomega
        self.test_layer = test_layer
        if self.solid_den_test:
            self.solid_density = list()
        self.save_name = parent_direct + '/config_vdata.npy'
        vdata_file = self.save_name
        if os.path.isfile(vdata_file):
            self.config_vdata = np.load(vdata_file).item()
            if len(self.config_vdata.keys()) != number_config:
                self.build_config_vdata()
        else:
            self.config_vdata = dict()
            self.build_config_vdata()
        
    def load_and_process(self):
        self.pdata, self.odata = helpy.load_data(self.prefix, 'p o')
        self.x0, self.y0, self.R = ring_motion.boundary(self.prefix)
        self.side_len = self.R * self.real_particle /4.0
        self.max_dist = self.side_len/1.25
        self.odata['orient'] = (self.odata['orient'] + np.pi)%(2 * np.pi) 
        self.frames = self.pdata['f']
        
        
    def track_config(self, pdata, odata, startframe):
        pfsets = helpy.load_framesets(pdata)
        pftrees = {f: KDTree(helpy.consecutive_fields_view(pfset, 'xy'),
                             leafsize=32) for f, pfset in pfsets.iteritems()}
        trackids = tracks.find_tracks(pdata, maxdist= self.max_dist, giveup = 10, n = 0, stub = 20, \
                                      pftrees = pftrees, pfsets = pfsets, startframe = startframe)
        trackids = tracks.remove_duplicates(trackids, data = pdata)
        return_data = helpy.initialize_tdata(pdata, trackids, odata)
        return_data = helpy.add_self_view(return_data, ('x','y'),'xy')
        return return_data
    
    def single_config_detect(self, v_data):
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
            order_para_mean.append(np.nanmean(product))
            vr_mean.append(np.nanmean(pid_data['vradi']))
            vomega_list.append(np.nanmean(np.abs(pid_data['vomega'])))
        qualify_id = np.asarray(qualify_id)
        order_para_mean = np.asarray(order_para_mean)
        vr_mean = np.asarray(vr_mean)
        vomega_list = np.asarray(vomega_list)
        return qualify_id, order_para_mean, vr_mean, vomega_list
        
        
    def build_config_vdata(self):
        for i in range(0, self.number_config):
            startframe = i * self.config_len
            mask = (self.frames >= startframe) & (self.frames < (i+1)*self.config_len)
            config_pdata, config_odata = self.pdata[mask], self.odata[mask]['orient']
            config_tdata = self.track_config(config_pdata, config_odata, startframe)
            tracksets = helpy.load_tracksets(config_tdata, run_track_orient = True, min_length = self.config_len//2, \
                                             run_repair = 'interp')
            track_prefix = {self.prefix: tracksets}
            v_data = velocity.compile_noise(track_prefix, width=(0.525,), cat = False, side = self.side_len, \
                                        fps = self.fps, ring = True, x0 = self.x0, y0 = self.y0, skip = 1, \
                                        grad = False, start = 0)
            v_data = v_data[self.prefix]
            self.config_vdata[startframe] = v_data
        np.save(self.save_name ,self.config_vdata)
        return
    
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
        self.solid_fraction = list()
        self.xys = dict()
        self.final_id = dict()
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
            if self.solid_den_test:                
                self.solid_density.append(self.get_solid_density(fdata))
            
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
            self.xys[startframe] = xys
            
            ftree = KDTree(xys, leafsize = 16)
            
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
            
            # if you neighbors qualified then you will be solid ,exclude detection error
            qualified_solid = list()
            for pt_id in range(len(xys)):
                dists, ids = ftree.query(xys[pt_id], self.nnn)
                qualified_solid.append(temp_mask[pt_id] or np.sum(temp_mask[ids[1:]]) >= 3)
            
            self.final_id[startframe] = qualified_solid    
            solid_number = np.sum(qualified_solid)
            self.solids.append(solid_number)
            self.liquid_density.append(self.density_calculation(solid_number, len(qualify_id), \
                                                                xys, qualified_solid))
            self.solid_fraction.append(float(solid_number)/len(qualify_id))
            
            if plot_number < self.plot_check:
                xs = helpy.consecutive_fields_view(fdata[startframe][track_mask],'x')
                ys = helpy.consecutive_fields_view(fdata[startframe][track_mask],'y')
                self.plot_check_solid(xs, ys, vr_mean, vr_mask, order_para_mean, \
                                      order_mask,vomega_list, vomega_mask, final_mask,\
                                      qualified_solid)
                plot_number += 1
        return len(qualify_id)
        
    
    def density_calculation(self, number_of_solids, total_number, xys, ids):
        solids_area =  self.real_particle ** 2 * number_of_solids / 0.95347
        fraction = solids_area/self.system_area
        if fraction < 0.8:
            liquid_area = self.system_area - solids_area
            number_liquid = total_number - number_of_solids
            rho_liquid = number_liquid * self.real_particle ** 2/ float(liquid_area)
        else:
            hist = True if fraction > 0.85 else False
            rho_liquid = vornoi_liquid(xys, ids, self.side_len, hist)
        return rho_liquid
        
    
    def plot_check_solid(self,xs,ys, vr_mean, vr_mask, dot_mean_list, order_mask, \
                         vomega_list, vomega_mask, final_mask, qualified_solid):
        #fig_omega, ax_omga = plt.subplots(figsize = (5,5))
        #omga_img = ax_omga.scatter(ys,1024 - xs, c = vomega_list, cmap = 'Paired')
        #fig_omega.colorbar(omga_img)
        """
        fig, ax = plt.subplots(2,2,figsize = (20,20))
        cax0 = fig.add_axes([0.2, 0.9, 0.25, 0.025])
        cax1 = fig.add_axes([0.6, 0.9, 0.25, 0.025])
        cax2 = fig.add_axes([0.2, 0.5, 0.25, 0.025])
        cax3 = fig.add_axes([0.6, 0.5, 0.25, 0.025])
        im0 = ax[0,0].scatter(ys,1024 - xs, c = vr_mean,  cmap='Paired')
        im1 = ax[0,1].scatter(ys,1024 - xs, c = final_mask ,  cmap='Paired')
        im2 = ax[1,0].scatter(ys,1024 - xs, c = vomega_list,  cmap='Paired')
        # need the order paramter criteria
        #im3 = ax[1,1].scatter(ys,1024 - xs, c = final_mask & vomega_mask & order_mask,  cmap='Paired')
        im3 = ax[1,1].scatter(ys,1024 - xs, c = qualified_solid, cmap='Paired')
        fig.colorbar(im0, cax=cax0, orientation='horizontal')
        fig.colorbar(im1, cax=cax1, orientation='horizontal')
        fig.colorbar(im2, cax=cax2, orientation='horizontal')
        fig.colorbar(im3, cax=cax3, orientation='horizontal')
        """
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
            
