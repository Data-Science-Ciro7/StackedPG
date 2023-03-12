#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt

__version__ = '0.2.0'
__author__ = 'Ciro Emmanuel-Martinez'

class StackedPg:
    """Generates a stacked periodogram objects from the files existing in a 'folder' passed
    at initialization.
    The files in the folder must be plain files of individual periodograms, with at least two columns:
    - frequency
    - power
    Any file that cannot be decoded / incorporated will be ignored (and reported)
    
    Parameters
    ----------
    - folder: (mandatory) the work folder holding the individual periodograms.
    - case_name: (optional) an identifier for identifying the case (it will be included in the names of the
        result files).
    - sep: the spacing character in the individual periodogram files.
    - comments: the character that identifies a comment in both the individual periodogram files and in the
    result file.
    - ref_lines: a list of tuples to configure reference vertical lines to ber shown in the plot. Each tuple in
    the list is composed of the frequency value where the line should be drawn, and a string with an identifier
    of the line (use 'None' in the second item in the tuple if you want that particular reference line to be
    omitted from the legend).
    - ref_colors: a list containing the colors for reference lines.
    - ref_styles: a list containing the styles for reference lines.
    Note: defaults are provided for 'colors' and 'styles' if not specified.
    
    Attributes
    ----------
    - folder: the work folder holding the individual periodograms.
    - case_name: (optional) a string identifying the case under analysis. It is also used for file names
    - stacked: the numpy array containing the normalized stacked periodogram, with three columns:
        - frecs
        - AND (multiplication)
        - OR (addition)
    - error_files: a list of the files under which could not be incorporated into the stacked periodogram.
        
    Methods
    -------
    - _calcStacked: calculates the stacked periodograms (called upon instance initialization).
    - plot: plots the calculated stacked periodograms and (optionally) saves them to a file.
    - save: saves the stacked periodograms into a text file.
    """
    
    def __init__(self, folder, case_name=None, header=False, sep=' ', comments="#", ref_lines=[],
                 ref_colors=None, ref_styles=None):
        self.folder = folder
        if case_name is None:
            self.case_name = self.folder
        else:
            self.case_name = case_name
        self.header=header
        self.sep = sep
        self.comments = comments
        self.ref_lines = ref_lines
        if ref_colors is None:
            self.ref_colors = ['red', 'blue', 'green', 'orange']
        else:
            self.ref_colors = ref_colors
        if ref_styles is None:
            self.ref_styles = ['-', '--', '-.', ':']
        else:
            self.ref_styles = ref_styles
        self.error_files = []
        self.stacked = None
        self._calcStacked()
    
    def _calcStacked(self):
        """ Calculates both stacked periodograms: AND and OR"""
        #pg_files = os.listdir(self.folder)
        pg_files = [f for f in os.listdir(self.folder) if os.path.isfile(self.folder + f)]
        frecs = None
        stacked_and = None
        stacked_or = None
        for f in pg_files:
            try:
                new_pg = np.genfromtxt(self.folder + f, comments=self.comments, delimiter=self.sep)
                # Normalize individual PG:
                new_pg[:, 1] = new_pg[:, 1] / np.trapz(y=new_pg[:, 1], x=new_pg[:, 0])
                if frecs is None:
                    frecs = new_pg[:, 0].copy()
                    stacked_and = new_pg[:, 1].copy()
                    stacked_or = new_pg[:, 1].copy()
                else:
                    stacked_and = stacked_and * new_pg[:, 1]
                    stacked_or = stacked_or + new_pg[:, 1]
            except Exception as e:
                self.error_files.append((f, str(e)))
        # Normalize the stacked periodograms:
        stacked_and = stacked_and / np.trapz(y=stacked_and, x=frecs)
        stacked_or = stacked_or / np.trapz(y=stacked_or, x=frecs)
        # Construct the array:
        self.stacked = np.vstack((frecs, stacked_and, stacked_or)).T
        
    def plot(self, showfig=True, savefig=False, combined=False):
        """ Plots the resulting stacked periodograms.
        Optionally, the figure can also be saved to a 'jpg' file.
        The parameter 'combined' is used to generate a single plot or a separate plot"""
        if combined == True:
            plt.figure(figsize=(15.0, 5.0))
            plt.plot(self.stacked[:,0], self.stacked[:,1], label="AND operation")
            plt.plot(self.stacked[:,0], self.stacked[:,2], label="OR operation")
            mark_count = 0
            for refline in self.ref_lines:
                c_idx = mark_count % len(self.ref_colors)
                s_idx = mark_count % len(self.ref_styles)
                plt.axvline(x=refline[0],
                            color=self.ref_colors[c_idx], linestyle=self.ref_styles[s_idx],
                            label=refline[1])
                mark_count += 1
            plt.title(self.case_name + " Stacked periodograms", fontdict = {'fontsize' : 20})
            plt.xlabel("Frequency", fontsize=12)
            plt.ylabel("Normalized power", fontsize=12)
            plt.legend()
            if showfig == True:
                plt.show()
            if savefig == True:
                plt.savefig(self.folder + self.case_name + "_StackedPG_Combined.jpg", format='jpg')
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15.0, 10.0))
            fig.suptitle(self.case_name + " Stacked periodograms", fontsize=20)
            axs[0].plot(self.stacked[:,0], self.stacked[:,1])
            axs[0].set_title("AND operation", fontsize = 16)
            axs[0].set_xlabel("Frequency", fontsize = 12)
            axs[0].set_ylabel("Normalized power", fontsize=16)
            mark_count = 0
            for refline in self.ref_lines:
                c_idx = mark_count % len(self.ref_colors)
                s_idx = mark_count % len(self.ref_styles)
                axs[0].axvline(x=refline[0],
                               color=self.ref_colors[c_idx], linestyle=self.ref_styles[s_idx],
                               label=refline[1])
                mark_count += 1
            axs[1].plot(self.stacked[:,0], self.stacked[:,2])
            #axs[1].set_title("OR operation", fontdict = {'fontsize' : 16})
            axs[1].set_title("OR operation", fontsize=16)
            axs[1].set_xlabel("Frequency", fontsize=12)
            axs[1].set_ylabel("Normalized power", fontsize=12)
            mark_count = 0
            for refline in self.ref_lines:
                c_idx = mark_count % len(self.ref_colors)
                s_idx = mark_count % len(self.ref_styles)
                axs[1].axvline(x=refline[0],
                               color=self.ref_colors[c_idx], linestyle=self.ref_styles[s_idx],
                               label=None)
                mark_count += 1
            fig.tight_layout(pad=1.5, h_pad=1.5)
            
            fig.legend()
            
            if showfig == True:
                fig.show()
            if savefig == True:
                plt.savefig(self.folder + self.case_name + "_StackedPG_Separate.jpg", format='jpg')
                
    def save(self, header=True, sep=' '):
        """ Saves the stacked periodograms, a single file with three columns 
        is generated (frecs, AND, OR)"""
        if header == False:
            header_text = ""
        else:
            header_text = "frec" + sep + "AND" + sep + "OR"
        np.savetxt(self.folder + self.case_name + "_StackedPG.dat", self.stacked,
                   fmt='%.9f', delimiter=sep, header=header_text, comments=self.comments)
