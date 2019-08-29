#!/usr/bin/python
# -*- coding:utf-8 -*-

from Simulation import model

model = model.AreaPerson(firm_file='D:/UCL/dissertation/tb/final_data/0821/initialize_data/firm.csv',
                         citizen_file='D:/UCL/dissertation/tb/final_data/0821/initialize_data/citizen.csv',
                         ward_file='D:/UCL/dissertation/tb/final_data/0821/initialize_data/ward.csv',
                         map_file='D:/UCL/dissertation/tb/final_data/0821/地图文件/greater_london_wards.shp')

model.run_model()