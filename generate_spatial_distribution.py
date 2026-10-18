# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

Generates a labels with coordinates list to be used with polar models


"""

#%% imports

import argparse
import os
import pandas as pd
import numpy as np
import sys


#%% get parameters

parser = argparse.ArgumentParser(description='Coordinate generation')
parser.add_argument('label_file',  help='Input path to label file')
parser.add_argument('dist', choices =['radial', 'focal'],  
                    help='distribution selection: radial or focal')
parser.add_argument('-target_sps', 
                    help='target species for focal distribution only. Use "" for binomials')
parser.add_argument('-target_sps_field', default='SCIENTIFIC_NAME', 
                    help='field containing target species, default SCIENTIFIC_NAME')
parser.add_argument('-radius', default=1, 
                    help='radius of circular distribution, default=1')
parser.add_argument('-arc_pi_frac', default=1, 
                    help='fraction of pi angle for other classes in focal distribution, default=1')
parser.add_argument('-alphabetical', choices =[True, False], default=True, 
                    help='If ordering the target_sps_field alphabetically or keep existing order, default=True')

args = parser.parse_args()

if args.dist == 'focal' and args.target_sps is None:
    parser.error('\n\ntarget species is required for focal distribution')



dataset = args.label_file
distribution = args.dist
target_sps =  args.target_sps
target_sps_field = args.target_sps_field 
radius = args.radius
arc_pi_frac = args.arc_pi_frac
alphabetical = args.alphabetical



#%% distribute classes

df = pd.read_csv(dataset)

classes = len(df)

def generate_radial_dist(num_points, radius=1):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points = [(x[i]*radius, y[i]*radius) for i in range(num_points)]
    return points


def generate_focal_dist(num_points, radius=1, arc_pi_frac=1):
    sector_length = np.pi / arc_pi_frac
    start_angle = (3 * np.pi / 2) - sector_length / 2
    end_angle = (3 * np.pi / 2) + sector_length / 2
    angles = np.linspace(start_angle,  end_angle, num_points-1, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    points = [(x[i]*radius, y[i]*radius) for i in range(num_points-1)]
    points = list([tuple((0,1*radius))]) + points ## add focal element
    return points


if distribution == 'radial':
    coords = generate_radial_dist(classes, radius)
elif distribution == "focal": 
    coords = generate_focal_dist(classes, radius)

assert len(coords) == classes, ("wrong items in coord list" + 
                                str(len(coords)) + 'vs' + str(classes))



#%% pass coord to dataframe 

if not target_sps_field in df.columns:
    print('Target sps field -', target_sps_field, '- is missing\n\nAvailable fields:',
          df.columns, '\n\nProcesss stopped')
    sys.exit()

df.reset_index(drop=False, inplace=True, names='orig_index')

if alphabetical:
    df = df.sort_values(by=target_sps_field, ascending=True)

if distribution == "focal":    
    
    if not target_sps in df[target_sps_field].values:
        print('Target sps -', target_sps, '- is missing\n\nAvailable targets:',
              df[target_sps_field].values, '\n\nProcesss stopped')
        sys.exit()
    else:
        df = df.sort_values(target_sps_field,key=lambda s: s !=target_sps,
                            kind='stable')

df.reset_index(drop=True, inplace=True)
df['coord'] = df.index.map(coords.__getitem__)
df = df.sort_values(by='orig_index')
df.reset_index(drop=True, inplace=True)
df.drop('orig_index', axis=1, inplace=True)
df.reset_index(drop=False, inplace=True, names='index')


#%% output

output_file = (dataset[:-4] + '-' + str(distribution) +'-distrib'+ 
               f"{'-'+target_sps if distribution=='focal' else ''}" 
              f"{'-alphabetical' if alphabetical else ''}" + '.csv')

if os.path.exists(output_file):
    while True:
        answer = input("\n \nThe destination file exists. Overwrite exiting file? Y/N:")
        if answer.lower().startswith("y"):
            df.to_csv(output_file, index=False)
            print("\n \nFile overwritten")
            break;
        else:
            print("\n\nProcesss stopped")
            sys.exit()
else:
    df.to_csv(output_file, index=True)
    print("\n \nFile saved")

print('\n', output_file, '\n')



