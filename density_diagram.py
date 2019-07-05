c#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:12:36 2019

@author: zhejun
"""

#from phase_separation_class import phase_coex
from machine_learning_phase import phase_coex
import numpy as np 
import matplotlib.pyplot as plt
from ring_motion import Find_Direct
from collections import defaultdict
import os


class phase_diagram:
    """
    path: directory containing different density data result
    for each density, do the phase separation information analysis:
        liquid density, solid density, liquid molecular order, solid molecular order and solid particle fraction
    put them in the phase diagram class.
    """
    def __init__(self, path, plot_check = 0, particle_size = 0.2, load_data = True, single_data = True, vomega = 0.1):
        """
        load_data: combine all information in each density into one dictionary
        single_data: for each density whether load pre-saved data information
        """
        self.path = path
        self.plot_check = plot_check
        #self.solid_density = solid_density
        self.load = load_data
        self.load_single_density_data = single_data
        self.vomega = vomega
        self.density_dict = self.build_density(self.path, self.plot_check)
        self.particle_size = particle_size
        self.total = np.pi*4**2/self.particle_size**2
        
    
    def single_density_load(self, prefix, plot_check = 0 ):
        print(prefix)
        phase = phase_coex(prefix, plot_check= plot_check, vomega = self.vomega)
        parent_direct = os.path.abspath(os.path.join(prefix, os.pardir))
        file_name = os.path.join(parent_direct, 'phase_data.npy')
        if self.load_single_density_data and os.path.isfile(file_name):
            result = np.load(file_name).item()
            liquid = result['liquid_density']
            solid = result['solid_density']
            solid_fraction = result['solid_fraction']
            liquid_order = result['liquid_order']
            solid_order = result['solid_order']
            total_number = int(prefix.split('/')[-2])
            print(total_number)
        else:
            total_number= phase.phase_detection()
            liquid = phase.liquid_density
            liquid_order = phase.liquid_molecular_order
            solid_fraction = phase.solid_fraction
            solid = phase.solid_density
            solid_order = phase.solid_molecular_order
        return {total_number: (liquid, solid, solid_fraction, liquid_order, solid_order, prefix)}  
    
    def build_density(self, path, plot_check = 0):
        prefixs = Find_Direct(path)
        density_file = os.path.join(path, 'density_dict_'+str(self.vomega)+'.npy')
        density_dict = {}
        if os.path.isfile(density_file) and self.load:
            number_of_density = len(prefixs)
            density_dict = np.load(density_file).item()
            if len(density_dict.keys()) < number_of_density:
                # only update the density that is not in the existing dictionary
                exist_prefix = set([epr[-1] for epr in density_dict.values()])
                for prefix in prefixs:
                    if prefix not in exist_prefix:
                        density_dict.update(self.single_density_load(prefix, plot_check = plot_check))
                np.save(density_file, density_dict)
        else:
            density_dict = {}
            for prefix in prefixs:
                print(prefix)
                density_dict.update(self.single_density_load(prefix, plot_check = plot_check))
            np.save(density_file, density_dict)
        return density_dict

    
    
    def plot_single_quantity(self, density, quantity, error, name = '', text_position = (0,0)):
        fig, ax = plt.subplots(figsize = (15,10))
        save_name = os.path.join(self.path, name + '.pdf')
        for d, q, e in zip(density, quantity, error):
            ax.errorbar(d, q, markersize='20', yerr= e,  fmt='o', elinewidth=4, capsize= 6,markeredgewidth=4)
        ax.set_xlabel(r'Area Fraction $\phi$', fontsize = 25)
        ax.set_xlim([0,1])
        ax.set_ylabel(name, fontsize = 25)
        ax.axvspan(0.5, 0.93, facecolor='#2ca02c', alpha=0.2)
        ax.text(text_position[0], text_position[1], "Phase Coexistence", fontsize = 28)
        ax.axvline(0.5, color = 'black', ls = '--', lw= 3)
        #ax.annotate("Phase Coexistence", arrow_position, xytext=(arrow_position[0] - 0.24, arrow_position[1]+0.1),arrowprops=dict(facecolor='black', width=2),fontsize =22)
        ax.tick_params(length=6, width=2, labelsize=20)
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
        fig.savefig(save_name, dpi = 600, bbox_inches = 'tight')
        return
    
    def phase_plot(self):
        self.diagram_data = defaultdict(list)
        sort_key = sorted(self.density_dict.keys())
        quant_name = ['liquid', 'solid', 'solid_fraction', 'liquid_order', 'solid_order']
        for key in sort_key:
            value = self.density_dict[key]
            self.diagram_data['density'].append(key/float(self.total))
            for quant, one_v in zip(quant_name, value[:-1]):
                self.diagram_data[quant].append(np.nanmean(one_v))
                self.diagram_data[quant+'_err'].append(np.nanstd(one_v))

        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['liquid'], \
                                  self.diagram_data['liquid_err'], 'Liquid Density', (0.51,0.51) )
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['solid'], \
                                  self.diagram_data['solid_err'], 'Solid Density', (0.51,0.7    ))
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['solid_fraction'],\
                                  self.diagram_data['solid_fraction_err'], 'Solid Number Fraction', (0.51,0.35))
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['liquid_order'],\
                                  self.diagram_data['liquid_order_err'], 'Liquid Molecular Order', (0.51,0.2))
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['solid_order'],\
                                  self.diagram_data['solid_order_err'], 'Solid Molecular Order', (0.51,0.72))
        np.save(os.path.join(self.path,'diagram_data.npy'),dict(self.diagram_data))