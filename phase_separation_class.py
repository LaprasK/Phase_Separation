#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:39:19 2019

@author: zhejunshen
"""
import helpy
import velocity
import ring_motion
import numpy as np
import matplotlib.pyplot as plt
import tracks
from scipy.spatial import cKDTree as KDTree
import velocity


class phase_coex:
    
    def __init__(self, prefix, number_config = 200, config_len = 50, real_particle = 0.2, bins = 200 , plus = 1,\
                nearest_neighbor_number = 5):
        self.prefix = prefix
        self.number_config = number_config
        self.config_len = config_len
        self.real_particle = real_particle
        self.bins = bins
        self.plus = plus
        self.nnn = nearest_neighbor_number
        self.load_and_process()
        
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
        vr_mean = list()
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
        qualify_id = np.asarray(qualify_id)
        return qualify_id, order_para_mean, vr_mean
        
    def phase_detection(self):
        self.solids = []
        self.detect = []
        for i in range(0, self.number_config):
            startframe = i * self.config_len
            mask = (self.frames >= startframe) & (self.frames < (i+1)*self.config_len)
            config_pdata, config_odata = self.pdata[mask], self.odata[mask]['orient']
            config_tdata = self.track_config(config_pdata, config_odata, startframe)
            tracksets = helpy.load_tracksets(config_tdata, run_track_orient = True, min_length = self.config_len//2, \
                                             run_repair = 'interp')
            track_prefix = {self.prefix: tracksets}
            v_data = velocity.compile_noise(track_prefix, width=(0.525,), cat = False, side = self.side_len, \
                                        fps = 5.0, ring = True, x0= self.x0, y0 = self.y0, skip = 1, \
                                        grad = False, start = 0)
            v_data = v_data[self.prefix]
            #pids = v_data.keys()
            qualify_id, order_para_mean, vr_mean = self.single_config_detect(v_data)
            self.detect.append(len(qualify_id))
            
            # order parameter requirement
            order_mask = (order_para_mean >= 1/np.sqrt(2) - 0.12) & (order_para_mean <= 1/np.sqrt(2) + 0.12)
            # vr requirement
            solid_vr_mean = np.mean(vr_mean)
            solid_vr_std = np.std(vr_mean)
            vr_mask = (vr_mean >= solid_vr_mean - solid_vr_std) & (vr_mean <= solid_vr_mean + solid_vr_std)
            
            fdata = helpy.load_framesets(v_data)
            count = 0
            temp_bool = True
            while (len(fdata[startframe + count]['t']) != len(qualify_id)) & (count < 49):
                count += 1
                if count == 49:
                    temp_bool = False
            startframe += count

            fdata_track = fdata[startframe]['t']
            track_mask = list()
            for t in fdata_track:
                track_mask.append(t in qualify_id)
            track_mask = np.asarray(track_mask)
            
            xys = helpy.consecutive_fields_view(fdata[startframe][track_mask], 'xy')
            ftree = KDTree(xys, leafsize = 16)
            
            final_mask = []
            
            for pt_id in range(len(xys)):
                if not vr_mask[pt_id]:
                    final_mask.append(False)
                    continue
                dists, ids = ftree.query(xys[pt_id], self.nnn)
                final_mask.append(np.all(vr_mask[ids]))
            self.solids.append(np.sum(final_mask & order_mask))
    
    
    def plot_check_solid(self):
        return
            