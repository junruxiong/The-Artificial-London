#!/usr/bin/python
# -*- coding:utf-8 -*-

from Simulation import model

model = model.AreaPerson(firm_file='initialize_data/firm.csv',
                         citizen_file='initialize_data/citizen.csv',
                         ward_file='initialize_data/ward.csv',
                         map_file='greater_london_wards.shp')

model.run_model()