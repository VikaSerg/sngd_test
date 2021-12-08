# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:57:36 2021

@author: - V. Sergeyeva 
"""

#%%

# this is to import Techlog measurements and computed variables from one job
# and look into the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import openpyxl


from dlisio import dlis



#%%



# test here for charging directly the .dlis file 
dlis_file, *tail = dlis.load(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\dlis_files\SHSV-227_O.1008165.04.4 ToolSpecific_ConPr_R03_Drill_DV6ME-1818.dlis')

print(dlis_file)
print(tail)

print(dlis_file.describe())

# all code is taken from tutorial here
# https://towardsdatascience.com/loading-well-log-data-from-dlis-using-python-9d48df9a23e2
# and adapted to ecoscope tool case and particular job example

# more detailed API can be found here
# https://dlisio.readthedocs.io/en/stable/dlis/api.html#dlisio.dlis.Channel



#%%



# we check here the dataframe using dlisio library and how to access the variables

# the description and details of the job itself are in ORIGINS command
origin, *origin_tail = dlis_file.origins
print(len(origin_tail))

print(origin.describe())



#%%


# we check here the dataframe using dlisio library and how to access the variables

# Frames within a DLIS file can represent different logging passes or different stages of data
# such as raw well log measurements to petrophysical interpretations or processed data. 
# Each frame has a number of properties.

print('Frames are : ', dlis_file.frames)

for frame in dlis_file.frames:
    print('***************************************** Frame ', frame)
    #print(f'Frame Name: \t\t {frame.name}')
    print(f'Index Type: \t\t {frame.index_type}')
    print(f'Direction: \t\t {frame.direction}')
    print(f'Num of Channels: \t {len(frame.channels)}')
    #print(f'Channel Names: \t\t {str(frame.channels)}')
    #print('\n\n')
    
    # Search through the channels for the index and obtain the units
    print('Channels are: ', frame.channels)
    
    for channel in frame.channels:
        #print('------------- Channel ', channel)
        if channel.name == frame.index:
            depth_units = channel.units
    
    print(f'Depth Interval: \t {frame.index_min} - {frame.index_max} {depth_units}')
    print(f'Depth Spacing: \t\t {frame.spacing} {depth_units}')




#%%



# we check here the dataframe using dlisio library and how to access the variables

#Parameters within the DLIS File
# we have a number of objects associated with the DLIS file. 
# To make them easier to read we can create a short function that creates a pandas dataframe containing the parameters.

def summary_dataframe(object, **kwargs):
    # Create an empty dataframe
    dataset = pd.DataFrame()
    
    # Iterate over each of the keyword arguments
    for i, (key, value) in enumerate(kwargs.items()):
        list_of_values = []
        
        # Iterate over each parameter and get the relevant key
        for item in object:
            # Account for any missing values.
            try:
                x = getattr(item, key)
                list_of_values.append(x)
            except:
                list_of_values.append('')
                continue
        
        # Add a new column to our data frame
        dataset[value] = list_of_values
    
    # Sort the dataframe by column 1 and return it
    return dataset.sort_values(dataset.columns[0])

# The logging parameters can be accessed by calling upon f.parameters. 
# To access the parameters, we can use the attributes name, long_name and values and pass these into the summary function.

dataset_param = summary_dataframe(dlis_file.parameters, name='Name', long_name='Long Name', values='Value')

# Hiding people's names that may be in parameters.
# These two lines can be commented out to show them
mask = dataset_param['Name'].isin(['R8', 'RR1', 'WITN', 'ENGI'])
dataset_param = dataset_param[~mask]

print('Dataset parameters: ')
print(dataset_param)



#%%


# Tools within the DLIS File
# The tools object within the DLIS file contains information relating to the tools that were used to acquire the data. 
# We can get a summary of the tools available be calling upon the summary_dataframe method.

tools = summary_dataframe(dlis_file.tools, name='Name', description='Description')
print('Dataset tools: ')
print(tools)


#%%



# As we are looking to plot rates data, we can look at the parameters for the ECO6 tool. 
# First, we need to grab the object from the dlis and then pass it into the summary_dataframe function.

eco6 = dlis_file.object('TOOL', 'ECO6')
eco6_params = summary_dataframe(eco6.parameters, name='Name', long_name='Long Name', values='Values')
print('eco6_params: ')
print(eco6_params)

# From the returned table, we can view each of the parameters that relate to the tool and the processing of the data.



#%%



# Plotting Data
# Frames and data can be accessed by calling upon the .object() for the file. 
# First, we can assign the frames to variables, which will make things easier when accessing the data within them, 
# especially if the frames contain channels/curves with the same name. 
# The .object() method requires the type of the object being accessed, i.e. 'FRAME' or 'CHANNEL' and its name. 
# In this case, we can refer back to the previous step which contains the channels and the frame names. 
# We can see that the basic logging curves are in one frame and the acoustic data is in another.

frame_60B = dlis_file.object('FRAME','60B')

# for ex. I want to access final bulk density measured by GG
rhob = dlis_file.object('CHANNEL', 'RHOB')
# Print out the properties of the channel/curve
print(f'Name: \t\t{rhob.name}')
print(f'Long Name: \t{rhob.long_name}')
print(f'Units: \t\t{rhob.units}')
print(f'Dimension: \t{rhob.dimension}') # if > 1, then data is an array

ucav = dlis_file.object('CHANNEL', 'UCAV')
# Print out the properties of the channel/curve
print(f'Name: \t\t{ucav.name}')
print(f'Long Name: \t{ucav.long_name}')
print(f'Units: \t\t{ucav.units}')
print(f'Dimension: \t{ucav.dimension}') # if > 1, then data is an array

tdep = dlis_file.object('CHANNEL', 'TDEP', copynr = 0) # here, copynr can be = , 1 or 2
# Print out the properties of the channel/curve
print(f'Name: \t\t{tdep.name}')
print(f'Long Name: \t{tdep.long_name}')
print(f'Units: \t\t{tdep.units}')
print(f'Dimension: \t{tdep.dimension}') # if > 1, then data is an array

natGray = dlis_file.object('CHANNEL', 'GRMA')
# Print out the properties of the channel/curve
print(f'Name: \t\t{natGray.name}')
print(f'Long Name: \t{natGray.long_name}')
print(f'Units: \t\t{natGray.units}')
print(f'Dimension: \t{natGray.dimension}') # if > 1, then data is an array




# Assigning Channels to Variables
# Now that we know how to access the frames and channels of the DLIS file
# we can now assign variable names to the curves that we are looking to plot. 

curves = frame_60B.curves()
#print('Curves are ', curves)

depth_c = curves['TDEP'] * 0.1 / 12 # (ft)
rhob_c = curves['RHOB']
ucav_c = curves['UCAV']
natGray_c = curves['GRMA']
#rhon_c = curves['RHON']
print('****************** Min and max depth values (ft)')
print(f'{depth_c.min()} - {depth_c.max()}')

# To make an initial check on data, we can create a quick log plot of RHOB and UCAV against TDEP
figure = plt.figure()

plt.subplot(3, 1, 1)
plt.plot(depth_c, rhob_c, label = 'RHOB')
#plt.plot(depth_c, rhon_c, label = 'RHON')
plt.xlabel('Depth (ft)')
plt.ylabel('Bulk density g/cm3')
plt.title('RHOB with depth')
plt.grid()
#plt.xlim(1243300.0, 1333220.0)
plt.show()
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.plot(depth_c, ucav_c, label = 'Borehole diameter')
plt.xlabel('Depth (ft)')
plt.ylabel('Diameter (in)')
plt.title('Borehole with depth')
plt.grid()
#plt.xlim(1243300.0, 1333220.0)
plt.show()
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.plot(depth_c, natGray_c, label = 'natural gamma ray signal')
plt.xlabel('Depth (ft)')
plt.ylabel('Gamma ray (gAPI)')
plt.title('Gamma ray with depth')
plt.grid()
plt.show()
plt.legend()
plt.tight_layout()








#%%











#%%


# in the next cells we keep the code for excel and .csv format import to python and dataframe creation

#
#
#



#%%

excel_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_channels_withoutNaNs.xlsx'

csv_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16NFR11_8_channels.csv'
#csv_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_channels_withoutNaNs.csv'

#mont_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\MONT_toAdd_WithoutNaN.xlsx'

dataset_excel = pd.read_excel(excel_file)
dataset_csv = pd.read_csv(csv_file)

#monitor_data = pd.read_excel(mont_file)

var_names_excel = (dataset_excel.columns.values)
var_names_csv = (dataset_csv.columns.values)
#monit_names = (monitor_data.columns.values)

print('**************************************** Data from Excel')
print (dataset_excel)
#print(dataset_excel.head())
print('Variables names:', var_names_excel, '; we have ', var_names_excel.shape[0], ' variables.')

print('**************************************** Data from CSV ')
print (dataset_csv)
#print(dataset_csv.head())
print('Variables names:', var_names_csv, '; we have ', var_names_csv.shape[0], ' variables.')


#print('Monitor names from excel:', monit_names, '; we have ', monit_names.shape[0], ' variables.')




#%%



print('**************************************** Original dataset ')
print (dataset_csv)

# I want to remove all raws where dataset_csv['RHON_EC'] == -9999.0
#dataset_clean = dataset_csv.drop(dataset_csv[dataset_csv["RHON_EC"] == -9999.0].index, inplace=True)
dataset_clean = dataset_csv.drop(dataset_csv[dataset_csv["RHON_EC"] == -9999.0].index)
dataset_clean.reset_index(drop=True, inplace=True)
print('**************************************** Filtered dataset 1 = dataset_clean ')
print ('Dataset without RHON_EC NaN values: ', dataset_clean)

#Say you want to delete all rows with negative values. One liner solution is:-
#df = df[(df > 0).all(axis=1)]

dataset_clean2 = dataset_csv[(dataset_csv != -9999.0).all(axis = 1)]
# eliminate the column WellName
# where the axis number is 0 for rows and 1 for columns
dataset_clean2 = dataset_clean2.drop('WellName', 1)
dataset_clean2.reset_index(drop=True, inplace=True)
print('**************************************** Filtered dataset 2 = dataset_clean2 ')
print ('Dataset without any NaN value: ', dataset_clean2)




#%%


# I work with dataset_clean2 = ALL raws with NaN values were eliminated

depth_min = 11600.0 # ft
depth_max = 13000.0
print("Dataset is reduced to min and max depth values = ", depth_min, depth_max, " ft.")

#dataset_depth = dataset_clean2.drop( dataset_clean2[ (dataset_clean2["TDEP"] > depth_max) or (dataset_clean2["TDEP"] < depth_min) ].index )

# eliminate raws for TDEP > depth max
dataset_depth = dataset_clean2.drop( dataset_clean2[ dataset_clean2["TDEP"] > depth_max ].index )
# eliminate raws for TDEP < depth min
dataset_depth = dataset_depth.drop( dataset_depth[ dataset_depth["TDEP"] < depth_min ].index )

dataset_depth.reset_index(drop=True, inplace=True)
print(dataset_depth)

var_names = (dataset_depth.columns.values)
print('Variables names: ', var_names)

depth = np.array(dataset_depth['TDEP'], dtype="float32")
depth = np.around(depth, decimals = 4)
print('Depth values = ', depth, depth.shape)

#print(dataset_depth)

# we save the dataset in .pkl format, it will be used by the density calculation algorythm

dataset_depth.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16NFR11_8_TDEPlimited.pkl')
print('Data was saved to .pkl!')




#%%


# we extract the columns we need and we assign it to a local var 
# to make manipulations easier

depth = dataset_clean['TDEP']
rho_ng_techlog = dataset_clean['RHON_EC']
#rho_filt = dataset['RHON_EC_withFILTinputs']

#mont_values = monitor_data['MONT']
#print(mont_values)

#print(depth)

figure = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(depth, rho_ng_techlog, label = 'bla bla here')
plt.xlabel('depth (ft)')
plt.ylabel('$\\rho_{ng}$ Techlog (g/cm3)')
plt.title('ng density with depth')
plt.grid()
plt.show()
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
#plt.scatter(rho_ng_techlog, rho_filt, color='red', marker = '+', label = 'something here')
plt.xlabel('$\\rho_{ng}$ Techlog (g/cm3)')
plt.ylabel('$\\rho_{ng}$ FILTERED (g/cm3)')
plt.title('ng densities')
#plt.legend(loc = 'upper left')
plt.grid()
plt.show()
plt.tight_layout()


#%%

# here I save pandas dataframe 
# I will be able to load it to another script without reprocessing again from excel

# we add MONITOR column to the original data
#dataset['MONT'] = mont_values
#print (dataset)

#dataset.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\20_China_WZ6_12_A11_dataset_60B.pkl')
dataset_excel.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16NFR11_8_noNaNs.pkl')

print('Data was saved to .pkl!')



#%%

# we load the .pkl dataframe that we generated from excel
# and we convert pandas dataframe to numpy table

#dataset_load = pd.read_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\20_China_WZ6_12_A11_dataset_60B.pkl')
#dataset_load = pd.read_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16NFR11_8_TDEPlimited.pkl')
dataset_load = pd.read_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16NFR11_8_noNaNs.pkl')
print('Data was loaded from .pkl!')

matrix = np.array(dataset_load)

print('matrix shape = ', matrix.shape)
print('matrix = ', matrix)

dataset_mx = np.transpose(matrix)
# we want variables to be the lines of our matrix
# and columns are the binnimg in depth
n_lines = dataset_mx.shape[0]
n_cols = dataset_mx.shape[1]

print('dataset matrix shape = ', n_lines, ' x ', n_cols)
print('dataset matrix = ', dataset_mx)

# Attention! there can be 'nan' values -> that means that the excel cell was empty
# this may cause bugs and errors when manipulating created numpy arrays

# to check if there are nan values in the matrix
print('Where are nan values? ', np.argwhere(np.isnan(dataset_mx)))


#%%


# we save parameters and calibration values from the job
#params_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\params_values.xlsx'
#calib_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\calib_values.xlsx'

params_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_params.xlsx'
calib_file = r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_calibs.xlsx'

params = pd.read_excel(params_file)
calibs = pd.read_excel(calib_file)

params_names = (params.columns.values)
calibs_names = (calibs.columns.values)

print('Parameters names from excel:', params_names)
print('Calibration names from excel:', calibs_names)

#params.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\20_China_WZ6_12_A11_parameters_60B.pkl')
#calibs.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\20-China_WZ6-12-A11\20_China_WZ6_12_A11_calibrations_60B.pkl')

params.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_params_60B.pkl')
calibs.to_pickle(r'C:\Users\VSergeyeva\Documents\tache_1\sNGD_XRkit_validation\16_NFR11_8\16_NFR11_8_calibs_60B.pkl')

print('Params and calibs were saved to .pkl!')



#%%


# this script can be used to look into raw techlog data variables
# and manipulate them
# dedicated ng-density algorythm will be created into a new script
# it will load pandas dataframe .pkl generated by this script


figure = plt.figure()

plt.plot(dataset_depth['TDEP'], dataset_depth['UCAV'], label = 'bla bla here')
plt.xlabel('depth (ft)')
plt.ylabel('UCAV values (in)')
plt.title('Borehole size with depth')
plt.grid()
plt.show()
plt.legend()
plt.tight_layout()



#%%






#%%







