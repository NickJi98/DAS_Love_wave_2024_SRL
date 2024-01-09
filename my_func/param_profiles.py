#!/usr/bin/env python3

"""
Parameters for selecting profiles (Figures 4 - 6)

Author: Qing Ji
"""

def get_param_profile(xcf_info, case=1):
    
    if case == 1:
        mask1 = (xcf_info['Ind1'] == 135) & (xcf_info['Distance'] <= 632)
        mask2 = (abs(xcf_info['Love']) >= 0.2) & (xcf_info['Rayleigh'] < 0.1) \
            & (xcf_info['Distance'] >= 630)
        sort1 = {"by":'Ind2', "ascending":True}
        sort2 = {"by":'Distance', "ascending":True}
        
    elif case == 2:
        mask1 = (xcf_info['Ind1'] + xcf_info['Ind2'] == 140) \
            & (xcf_info['Ind2'] > 70) & (xcf_info['Ind2'] < 130)
        mask2 = (xcf_info['Ind2'] - xcf_info['Ind1'] == 120) \
            & (xcf_info['Ind2'] >= 135) & (xcf_info['Ind1'] < 70)
        sort1 = {"by":'Ind2', "ascending":True}
        sort2 = {"by":'Ind2', "ascending":True}
        
    elif case == 3:
        mask1 = (xcf_info['Ind1'] + xcf_info['Ind2'] == 260) \
            & (xcf_info['Ind2'] > 130) & (xcf_info['Ind1'] >= 70)
        mask2 = (xcf_info['Ind2'] - xcf_info['Ind1'] == 120) \
            & (xcf_info['Ind2'] >= 135) & (xcf_info['Ind1'] < 70)
        sort1 = {"by":'Ind2', "ascending":True}
        sort2 = {"by":'Ind2', "ascending":False}
        
    elif case == 4:
        mask1 = (xcf_info['Ind1'] == 0) & (xcf_info['Ind2'] <= 65)
        mask2 = (xcf_info['Ind1'] == 135) & (xcf_info['Ind2'] <= 200)
        sort1 = {"by":'Distance', "ascending":True}
        sort2 = {"by":'Distance', "ascending":True}
        
    param_profile = {'mask1': mask1, 'mask2': mask2, 
                     'sort1': sort1, 'sort2': sort2}
    
    return param_profile
