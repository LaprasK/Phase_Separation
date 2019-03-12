#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:12:36 2019

@author: zhejun
"""

from phase_separation_class import phase_coex
import numpy as np 
import matplotlib.pyplot as plt
from ring_motion import Find_Direct
from collections import defaultdict
import os


class phase_diagram:
    def __init__(self, path, plot_check = 0, solid_density = True, particle_size = 0.2, load_data = True, vomega = 0.15):
        self.path = path
        self.plot_check = plot_check
        self.solid_density = solid_density
        self.load = load_data
        self.vomega = vomega
        self.density_dict = self.build_density(self.path, self.plot_check, self.solid_density)
        self.particle_size = particle_size
        self.total = np.pi*4**2/self.particle_size**2
        
    
    def single_density_load(self, prefix, plot_check = 0 ,solid_density = True):
        phase = phase_coex(prefix, plot_check= plot_check, solid_den_test= solid_density, vomega = self.vomega)
        total_number= phase.phase_detection()
        liquid = phase.liquid_density
        solid_fraction = phase.solid_fraction
        solid = list()
        if solid_density:
            solid = phase.solid_density
        return {total_number: (liquid, solid, solid_fraction, prefix)}  
    
    def build_density(self, path, plot_check = 0, solid_density = True):
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
                        density_dict.update(self.single_density_load(prefix, plot_check = plot_check,\
                                                                     solid_density = solid_density))
                np.save(density_file, density_dict)
        else:
            density_dict = {}
            for prefix in prefixs:
                print(prefix)
                density_dict.update(self.single_density_load(prefix, plot_check = plot_check, \
                                                             solid_density = solid_density))
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
        ax.text(0.56, 0.35, "Phase Coexistence", fontsize = 28)
        ax.axvline(0.5, color = 'black', ls = '--', lw= 3)
        #ax.annotate("Phase Coexistence", arrow_position, xytext=(arrow_position[0] - 0.24, arrow_position[1]+0.1),arrowprops=dict(facecolor='black', width=2),fontsize =22)
        ax.tick_params(length=6, width=2, labelsize=20)
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
        fig.savefig(save_name, dpi = 600, bbox_inches = 'tight')
        return
    
    def phase_plot(self):
        self.diagram_data = defaultdict(list)
        sort_key = sorted(self.density_dict.keys())
        for key in sort_key:
            value = self.density_dict[key]
            self.diagram_data['density'].append(key/self.total)
            self.diagram_data['liquid'].append(np.mean(value[0]))
            self.diagram_data['liquid_err'].append(np.std(value[0]))
            self.diagram_data['solid'].append(np.mean(value[1]))
            self.diagram_data['solid_err'].append(np.std(value[1]))
            self.diagram_data['solid_fraction'].append(np.mean(value[2]))
            self.diagram_data['solid_fraction_err'].append(np.std(value[2]))
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['liquid'], \
                                  self.diagram_data['liquid_err'], 'Liquid Density', (0.51,0.51) )
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['solid'], \
                                  self.diagram_data['solid_err'], 'Density Close to Boundary')
        self.plot_single_quantity(self.diagram_data['density'], self.diagram_data['solid_fraction'],\
                                  self.diagram_data['solid_fraction_err'], 'Solid Number Fraction', (0.51,0.05))