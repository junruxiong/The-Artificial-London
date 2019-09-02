# -*- coding: utf-8 -*-

from mesa import Model
from mesa.space import MultiGrid
from mesa_geo.geospace import GeoSpace, GeoAgent
from mesa_geo.geoagent import AgentCreator
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from enum import Enum
import random
import numpy as np
import queue
import pandas as pd
from Simulation import agent
import geopandas as gpd
from shapely.geometry import Point, Polygon
from Simulation import configs
import datetime
import operator
import math
import matplotlib.pyplot as plt


def get_top_npercent_list(list, n):
    olen = len(list)
    begin = int(min(olen - 1, olen - 1 - olen / (100.0 / n)))
    return list[begin:]


def get_low_npercent_list(list, n):
    olen = len(list)
    end = int(max(1, (olen) / (100.0 / n)))
    return list[:end]


class AreaPerson(Model):
    def __init__(self, citizen_file, firm_file, ward_file, map_file):
        super().__init__()
        self.flag_of_debug = True
        self.step_num = 0

        self.citizen_file = citizen_file
        self.firm_file = firm_file
        self.ward_file = ward_file
        self.map_file = map_file

        self.init_from_file()

        self.income_tax_rate = 0.10
        self.business_rate = 0.17
        self.house_rate = configs.HOUSE_RATE  # buy house tax
        self.tax_begin_numer = 20  # Starting from the number of iterations

        self.schedule = RandomActivation(self)
        # self.datacollecto = DataCollector(agent_reporters={"satisfaction": lambda x: x.satisfaction})
        # self.schedule.re
        # self.datacollector = DataCollector(
        #     {"Wolves": lambda m: m.schedule.get_breed_count(Person),
        #      "Sheep": lambda m: m.schedule.get_breed_count(Sheep)})

        # self.grid = GeoSpace(crs={"init": "epsg:3857"})  # epsg:4326

        # self.datacollector = DataCollector

        self.map_df = gpd.read_file(self.map_file)

        self.citizen_num = len(self.citizen_data)
        self.firm_wealth_level = 1e12  # all firms assets lowe 5%，if exceed they can run business
        self.firm_wealth_record = dict()  # key is firm id， value is firm ealth

        # self.aid_to_satisfaction = dict()  # key is area id， value ward satisfaction
        self.aid_to_area = dict()  # key is area id， value is area object
        self.gsscode_to_aid = dict()  # key is gsscode，value is areaid

        self.top1_area_list = []

        self.house_market = dict()  # Sale property。 key is house id，object is house object

        self.total_people_num = 0
        self.total_firm_num = 0

        start_time = datetime.datetime.now()

        self.area_with_max_satisfaction = None  # The area with the highest satisfaction of the current round, used for plotting
        self.area_with_max_avg_housePrice = None
        # initialize ward
        # The gird only puts the area above, and the person and the company are dynamically bound to an area.
        # AC = AgentCreator(agent.Area)
        # areas = AC.from_GeoDataFrame(self.map_df)
        for i in range(self.citizen_num):
            if self.flag_of_debug and i > 100:
                break
            area_name = self.citizen_data['Ward_Name'][i]
            area_code = self.citizen_data['Ward_Code'][i]
            print("init: %d\t%s\t%s" % (i, area_name, area_code))
            citizen = self.citizen_data[self.citizen_data['Ward_Code'] == area_code].reset_index(drop=True)
            firm = self.firm_data[self.firm_data['Ward_Code'] == area_code].reset_index(drop=True)
            ward = self.ward_data[self.ward_data['GSS_CODE'] == area_code].reset_index(drop=True)
            gdf = self.map_df[self.map_df['GSS_CODE'] == area_code].reset_index(drop=True)
            area = agent.Area(self, self.next_id())
            # 给area填充属性
            area.name = area_name
            area.code = area_code
            area.firm_num = firm.loc[0, 'firm_count_Ward']
            area.employee_num = firm.loc[0, 'post_count_Ward']

            area.population = int(citizen.loc[0, 'population'])
            area.earning = citizen.loc[0, 'mean_income']
            area.health = citizen.loc[0, 'health']
            area.satisfaction = citizen.loc[0, 'satisfaction']
            area.wealth = citizen.loc[0, 'assets']
            area.housing_rate = citizen.loc[0, 'own_house_rate']
            area.rent_rate = citizen.loc[0, 'rent_rate']
            area.high_edu_rate = citizen.loc[0, 'high_edu_rate'] / 100.0
            area.middle_edu_rate = citizen.loc[0, 'middle_edu_rate'] / 100.0
            area.low_edu_rate = citizen.loc[0, 'low_edu_rate'] / 100.0

            area.total_space = ward.loc[0, 'total_space']
            area.free_space = ward.loc[0, 'empty_space_rate'] * area.total_space
            area.green_space = ward.loc[0, 'green_space_rate'] * area.total_space
            area.building_space = ward.loc[0, 'public_building_space'] * area.total_space
            area.house_space = ward.loc[0, 'residence_space_rate'] * area.total_space

            area.safety = ward.loc[0, 'nor_safety']
            area.railway = ward.loc[0, 'nor_railway']


            # area.ave_house_price = float(1 / 100.0 * ward.loc[0, 'mean_income'] * \
            #                              random.uniform(configs.HOUSE_PRICE_TO_INCOME * 0.8,
            #                                             configs.HOUSE_PRICE_TO_INCOME * 1.2))
            # mean house price
            area.ave_house_price = ward.loc[0, 'mean_house_price']
            # print("ave house price: %d" % area.ave_house_price)

            self.schedule.add(area)
            self.aid_to_area[area.unique_id] = area
            self.gsscode_to_aid[area_code] = area.unique_id
            # self.grid.add_agents(area)

            # According to the population of ward, place citizen. citizens are dynamic
            for j in range(area.population):
                # person_id = self.citizen_data['区编号'][i] + "_p_" + str(self.next_id())
                x, y = self.get_one_position_by_area_code(area_code)
                person = agent.Person(model=self, pid=self.next_id(),
                                      edu=self.get_random_edu(area_code),
                                      earning=self.get_random_income(area_code),
                                      xpoint=x,
                                      ypoint=y,
                                      house=self.get_random_house(area_code))
                if person.house_state == agent.HouseState.Yes:
                    person.satisfaction += 5.0  # + satisfaction, if citizen own house
                # from mean income，random citizen assets(0.5-1.5)
                person.wealth = random.uniform(0.50 * configs.WEALTH_TO_INCOME,
                                               1.5 * configs.WEALTH_TO_INCOME) * ward.loc[0, 'mean_income'] / 12.0
                person.health = random.uniform(0.9, 1.0) * area.health  # initial health
                # TODO 优化是否有工作初始化逻辑
                person.job_state = random.choice([agent.JobState.Employment, agent.JobState.Unemployment])
                area.people[person.unique_id] = person
                area.pid_to_income[person.unique_id] = person.earning
                person.area = area
                self.schedule.add(person)

            # Place several companies based on the number of companies in the ward
            # firm_num = random.randint(area.population / 50, area.population / 20)
            firm_num = int(firm.loc[0, 'firm_count_Ward'])

            for j in range(firm_num):
                x, y = self.get_one_position_by_area_code(area_code)
                firm = agent.Firm(model=self, fid=self.next_id(),
                                  xpoint=x,
                                  ypoint=y,
                                  wealth=0)
                firm.area = area
                area.firms[firm.unique_id] = firm
                self.firm_wealth_record[firm.unique_id] = firm.wealth
                self.schedule.add(firm)

            # people random match job
            if firm_num > 0:
                firms = list(area.firms.values())
                for p in list(area.people.values()):
                    if p.job_state == agent.JobState.Employment:
                        p.firm = random.choice(firms)
                        p.firm.employees[p.unique_id] = p
            self.total_firm_num += firm_num
            self.total_people_num += area.population

        print("area nums: %d" % self.citizen_num)
        print("people nums: %d" % self.total_people_num)
        print("firm nums: %d" % self.total_firm_num)
        end_time = datetime.datetime.now()
        time_cost = end_time - start_time
        print("time cost: " + str(time_cost).split('.')[0])

    def init_from_file(self):
        self.citizen_data = pd.read_csv(self.citizen_file)
        self.firm_data = pd.read_csv(self.firm_file)
        self.ward_data = pd.read_csv(self.ward_file)

    def get_random_edu(self, area_code):
        d1 = self.citizen_data[self.citizen_data['Ward_Code'] == area_code].reset_index(drop=True)
        high_edu = d1.loc[0, 'high_edu_rate']
        low_edu = d1.loc[0, 'low_edu_rate']
        random_number = 100 * random.random()
        if random_number < high_edu:
            return agent.EduState.High
        elif random_number > 100 - low_edu:
            return agent.EduState.Low
        else:
            return agent.EduState.Middle

    def get_random_income(self, area_code):
        d1 = self.citizen_data[self.citizen_data['Ward_Code'] == area_code].reset_index(drop=True)
        mu = d1.loc[0, 'mean_income']
        sigma2 = mu * 0.2  # The variance is set to 0.2 times the mean
        return np.random.normal(mu, sigma2, 1)[0]

    def get_random_house(self, area_code):
        d1 = self.citizen_data[self.citizen_data['Ward_Code'] == area_code].reset_index(drop=True)
        random_number = 100 * random.random()
        if random_number < d1.loc[0, 'own_house_rate']:
            return agent.HouseState.Yes
        else:
            return agent.HouseState.No

    def get_coords(self, area_code):
        d1 = self.map_df[self.map_df['GSS_CODE'] == area_code].reset_index(drop=True)
        try:
            aa = list(d1.loc[0, 'geometry'].exterior.coords.xy)
            xx = [aa[0][i] for i in range(len(aa[0]))]
            yy = [aa[1][i] for i in range(len(aa[1]))]
        except Exception:
            print('no such district: ' + area_code)
        return xx, yy, d1

    def get_one_position_by_area_code(self, area_code):
        xx, yy, d1 = self.get_coords(area_code)
        max_xx = max(xx)
        min_xx = min(xx)
        max_yy = max(yy)
        min_yy = min(yy)
        # ss = 0
        while True:
            random_x = random.uniform(min_xx, max_xx)
            random_y = random.uniform(min_yy, max_yy)
            if Point(random_x, random_y).within(d1.loc[0, 'geometry']):
                return (random_x, random_y)

    def get_top1_area(self):
        if len(self.top1_area_list) < 1:
            return None
        return self.random.choice(self.top1_area_list)

    # 投资公共设施
    def invest_service(self, area):
        if not area:
            assert area
            return False
        if area.free_space > 0 and len(area.firms) > 0:
            sspace = random.uniform(0.003, 0.01) * area.free_space
            cost = sspace * area.ave_house_price / 85
            firm = random.choice(list(area.firms.values()))
            if firm:
                firm = random.choice(list(area.firms.values()))
                area.wealth -= cost
                area.free_space -= sspace
                firm.wealth += cost
                # print("--->invest success. aid:%d, fid:%d" % (area.unique_id, firm.unique_id))
            return True
        return False

    def add_job(self, area):
        if not area:
            return False
        firms = list(area.firms.values())
        if len(firms) > 0:
            firm = random.choice(firms)
            firm.add_job(True)
        return True

    def save(self):
        pass

    def show1(self):
        self.normalized()
        satisfaction = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.satisfaction
            #else:
            #    val = random.uniform(0.0, 1.0)
            satisfaction.append(val)
        print("---->try to plot-------->")
        column_name = "manyidu_" + str(self.step_num)
        self.map_df.insert(2, column_name, satisfaction, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Satisfaction after day %d " % self.step_num)
        self.map_df.plot(figsize=(15,15), column=column_name, ax=ax, legend=True)
        plt.show()
        # print(self.map_df.head())
        # self.map_df.geometry.plot(column='manyidu')

    def show_firms(self):
        #self.normalized()
        firms = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = len(area.firms)
            # else:
            #    val = random.uniform(0.0, 1.0)
            firms.append(val)
        print("---->try to plot-------->")
        column_name = "show_firms" + str(self.step_num)
        self.map_df.insert(2, column_name, firms, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward number of firms after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Oranges', column=column_name, ax=ax, legend=True)
        plt.show()
        # print(self.map_df.head())
        # self.map_df.geometry.plot(column='manyidu')

    def show_assets(self):
        #self.normalized()
        assets = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.wealth
            # else:
            #    val = random.uniform(0.0, 1.0)
            assets.append(val)
        print("---->try to plot-------->")
        column_name = "show_assets" + str(self.step_num)
        self.map_df.insert(2, column_name, assets, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward mean assets after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Greens', column=column_name, ax=ax, legend=True)
        plt.show()

    def show_citizen(self):
        # self.normalized()
        population = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.population
            # else:
            #    val = random.uniform(0.0, 1.0)
            population.append(val)
        print("---->try to plot-------->")
        column_name = "population" + str(self.step_num)
        self.map_df.insert(2, column_name, population, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward number of citizens after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Reds', column=column_name, ax=ax, legend=True)
        plt.show()

    def show_citizen_density(self):
        #self.normalized()
        citizen_density = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = len(area.population)/area.total_space
            # else:
            #    val = random.uniform(0.0, 1.0)
            citizen_density.append(val)
        print("---->try to plot-------->")
        column_name = "show_citizen_density" + str(self.step_num)
        self.map_df.insert(2, column_name, citizen_density, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("citizen density after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Blues', column=column_name, ax=ax, legend=True)
        plt.show()

    def show_mean_house_price(self):
        # self.normalized()
        mean_commute_dist = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = len(area.employee_num)
            # else:
            #    val = random.uniform(0.0, 1.0)
            mean_commute_dist.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, mean_commute_dist, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward employee_num after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='BuPu',column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()

    def show_income(self):
        # self.normalized()
        mean_income = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.ave_income
            # else:
            #    val = random.uniform(0.0, 1.0)
            mean_income.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, mean_income, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Mean Income after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='plasma', column=column_name, ax=ax, legend=True)
        plt.show()

    def show_building_space(self):
        # self.normalized()
        building_space = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.total_space / area.building_space
            # else:
            #    val = random.uniform(0.0, 1.0)
            building_space.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, building_space, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Public Space Rate after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='RdPu', column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()

    def show_house_space(self):
        # self.normalized()
        house_space = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.total_space / area.house_space
            # else:
            #    val = random.uniform(0.0, 1.0)
            house_space.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, house_space, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward House Space Rate after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='PuRd', column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()

    def show_free_space(self):
        # self.normalized()
        free_space = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.total_space / area.free_space
            # else:
            #    val = random.uniform(0.0, 1.0)
            free_space.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, free_space, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Empty Space Rate day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='PuBuGn', column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()

    def show2(self):
        if not self.area_with_max_satisfaction:
            print("cannot show 2")
            return
        print("try to show2..... %d" % len(self.area_with_max_satisfaction.satisfaction_list))
        aid_to_satisfaction = dict()  # key is area id， value ward satisfaction
        for aid, a in self.aid_to_area.items():
            if a.satisfaction > 0.0:
                aid_to_satisfaction[aid] = a.satisfaction < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_satisfaction)/2)
        for aid, s in sorted(aid_to_satisfaction.items(), key=operator.itemgetter(1)):
            if idx < 5 or len(aid_to_satisfaction) - idx < 6:
                df[self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            print("show2: %d %d"% (idx, len(aid_to_satisfaction)/2))
            if median_idx == idx:
                # print("get median...")
                df["MEDIAN_" + self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            idx += 1
        df = pd.DataFrame(df)
        print(df.head())
        fig, ax = plt.subplots(1, 1)
        ax.set_title("top5 and low5 Ward Satisfaction ")
        ax.set_xlim((0, self.step_num))
        column_name = 'satisfaction'
        df.plot(figsize=(10, 10), ax=ax)
        plt.subplots_adjust(wspace=2)
        plt.show()

    def show3(self):
        if not self.area_with_max_satisfaction:
            print("cannot show 2")
            return
        print("try to show2..... %d" % len(self.area_with_max_satisfaction.satisfaction_list))
        aid_to_assets = dict()
        for aid, a in self.aid_to_area.items():
            aid_to_assets[aid] = a.ave_wealth < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_assets) / 2)
        for aid, s in sorted(aid_to_assets.items(), key=operator.itemgetter(1)):
            if idx < 5 or len(aid_to_assets) - idx < 6:
                df[self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            print("show2: %d %d" % (idx, len(aid_to_assets) / 2))
            if median_idx == idx:
                # print("get median...")
                df["MEDIAN_" + self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            idx += 1
        df = pd.DataFrame(df)
        print(df.head())
        fig, ax = plt.subplots(1, 1)
        ax.set_title("top5 and low5 Ward Mean Assetas ")
        ax.set_xlim((0, self.step_num))
        column_name = 'satisfaction'
        df.plot(figsize=(10, 10), ax=ax)
        plt.subplots_adjust(wspace=2)
        plt.show()

    def commute_line(self):
        if not self.area_with_max_satisfaction:
            print("cannot show 3")
            return
        print("try to show3..... %d" % len(self.area_with_max_satisfaction.satisfaction_list))
        aid_to_commute_line = dict()
        for aid, a in self.aid_to_area.items():
            aid_to_commute_line[aid] = a.ave_distance < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_commute_line) / 2)
        for aid, s in sorted(aid_to_commute_line.items(), key=operator.itemgetter(1)):
            if idx < 5 or len(aid_to_commute_line) - idx < 6:
                df[self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            print("show2: %d %d" % (idx, len(aid_to_commute_line) / 2))
            if median_idx == idx:
                # print("get median...")
                df["MEDIAN_" + self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            idx += 1
        df = pd.DataFrame(df)
        print(df.head())
        fig, ax = plt.subplots(1, 1)
        ax.set_title("top5 and low5 Ward Commute Distance ")
        ax.set_xlim((0, self.step_num))
        column_name = 'satisfaction'
        df.plot(figsize=(10, 10), ax=ax)
        plt.subplots_adjust(wspace=2)
        plt.show()

    def show_public_space(self):
        if not self.area_with_max_satisfaction:
            print("cannot show 3")
            return
        print("try to show3..... %d" % len(self.area_with_max_satisfaction.satisfaction_list))
        aid_to_public_space = dict()  # key is area id， value ward satisfaction
        for aid, a in self.aid_to_area.items():
            aid_to_public_space[aid] = a.building_space < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_public_space) / 2)
        for aid, s in sorted(aid_to_public_space.items(), key=operator.itemgetter(1)):  # Sort the results according to satisfaction, the list can not be sorted again
            if idx < 5 or len(aid_to_public_space) - idx < 6:
                df[self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            print("show2: %d %d" % (idx, len(aid_to_public_space) / 2))
            if median_idx == idx:
                # print("get median...")
                df["MEDIAN_" + self.aid_to_area[aid].name] = pd.Series(self.aid_to_area[aid].satisfaction_list)
            idx += 1
        df = pd.DataFrame(df)
        print(df.head())
        fig, ax = plt.subplots(1, 1)
        ax.set_title("top5 and low5 Public Space ")
        ax.set_xlim((0, self.step_num))
        column_name = 'satisfaction'
        df.plot(figsize=(10, 10), ax=ax)
        plt.subplots_adjust(wspace=2)
        plt.show()

    # TODO
    def normalized(self):
        mx = -99999999
        self.total_people_num = self.total_firm_num = 0
        for _, a in self.aid_to_area.items():
            self.total_people_num += len(a.people)
            self.total_firm_num += len(a.firms)
            a.cscore = 0
            total_income = 0
            total_wealth = 0
            a.population = len(a.people)
            if not a.population:  # Terrible, people in this area are dead.
                a.satisfaction = 0
                if a.satisfaction > 0:
                    mx = a.satisfaction
                continue
            dis_num = 0  # total commuting citizens
            total_distance = 0  # total commuting distance
            for _, p in a.people.items():
                if p.job_state == agent.JobState.Employment:
                    total_income += p.earning
                    if p.firm:
                        x = p.xpoint
                        y = p.ypoint
                        x = p.firm.xpoint - x
                        y = p.firm.ypoint - y
                        total_distance += math.sqrt(x * x + y * y)
                        dis_num += 1
                total_wealth += p.wealth
            a.ave_income = total_income * 1.0 / a.population
            a.ave_wealth = total_wealth * 1.0 / a.population
            a.safety = a.safety
            a.railway = a.railway
            a.ave_distance = total_distance / max(1.0, dis_num)  # avoid divide 0
            a.cscore = a.ave_income * 1
            a.cscore += a.ave_wealth * 0.01
            a.cscore += a.green_space * 100 / a.total_space
            a.cscore += a.building_space * 100 / a.total_space
            a.cscore += a.safety * 100
            a.cscore += a.railway * 100
            if a.cscore > a.ave_distance * 50:
                a.cscore -= (50 * a.ave_distance)
            if a.cscore > mx:
                mx = a.cscore
                self.area_with_max_satisfaction = a
        mx = max(1.0, mx)  # avoid divide 0
        for _, a in self.aid_to_area.items():
            a.satisfaction = a.cscore / mx
            a.collect_data()
            # print("area %d score %.2f" % (a.unique_id, a.satisfaction))

    def step(self):
        self.normalized()
        self.step_num += 1
        print("step %d, firm_num:%d person_num:%d" % (self.step_num,
                                                      self.total_firm_num,
                                                      self.total_people_num))
        self.normalized()
        all_wealth = list(self.firm_wealth_record.values())
        all_wealth.sort()  # sort
        try:
            self.firm_wealth_level = get_low_npercent_list(all_wealth, 5)[-1]  # 后5%
        except:
            print("heheh")

        aid_to_satisfaction = dict()  # key is area id， value ward satisfaction
        for aid, a in self.aid_to_area.items():
            aid_to_satisfaction[aid] = a.satisfaction

        # Update top one percent before the satisfaction of ward
        self.top1_area_list = []

        aids_sort_by_satisfaction = []
        for aid, s in sorted(aid_to_satisfaction.items(), key=operator.itemgetter(1)):  # Sort the results according to satisfaction, the list can not be sorted again
            aids_sort_by_satisfaction.append(aid)

        self.top1_area_list = get_top_npercent_list(aids_sort_by_satisfaction, 1)  # satisfaction top 1%
        low_10_area = get_low_npercent_list(aids_sort_by_satisfaction, 10)  # low 10%
        low_5_area = get_low_npercent_list(aids_sort_by_satisfaction, 5)  # low 5%
        low_1_area = get_low_npercent_list(aids_sort_by_satisfaction, 1)  # low 1%

        area = self.aid_to_area[self.random.choice(low_5_area)]
        all_firms = list(area.firms.values())
        if len(all_firms) > 0:
            firm = self.random.choice(all_firms)
            firm.add_job(fixed=True)

        for a in list(self.aid_to_area.values()):
            a.tax_free_flag = False
        # Update and select 1% of ward after company satisfaction,
        # exempt from consumption tax (excluding real estate transaction tax)
        self.aid_to_area[random.choice(low_1_area)].tax_free_flag = True

        # Random choose a ward with a satisfaction below 5% to add a post
        try_times = 0
        while try_times < 5:
            if self.add_job(self.aid_to_area[random.choice(low_5_area)]):
                break
            try_times += 1

        # invest public facilities
        try_times = 0
        while try_times < 5:
            if self.invest_service(self.aid_to_area[random.choice(low_10_area)]):
                break
            try_times += 1

        self.schedule.step()
        # self.datacollector.collect()

    def run_model(self):
        for i in range(configs.MAX_DAYS):
            start_time = datetime.datetime.now()
            self.step()
            end_time = datetime.datetime.now()
            time_cost = end_time - start_time
            print("in round %d, time cost: %s" % (i + 1, str(time_cost).split('.')[0]))
            for j in range(0,configs.MAX_DAYS,30):


                if i == j*3 - 1:
                    self.show1()
                # if i == j*3:
                #     self.show_firms()
                # if i == j*3 + 1:
                #     self.show_citizen()
                # if i == j*3:
                #     self.show_income()
                # if i == j*3  + 1:
                #     self.show_building_space()
                # if i == j*3  + 2:
                #     self.show_house_space()
                # if i == j*3  + 3:
                #     self.show_free_space()
            #            if i == 30
            if i == 0:
                self.show1()
            # if i == 1:
            #     self.show_firms()
            # if i == 2:
            #     self.show_citizen()
            if i == 29:
                self.show1()
            # if i == 30:
            #     self.show_firms()
            # if i == 31:
            #     self.show_citizen()

            # if i == 30 - 3:
            #     self.show_firms()
            # if i == 30 - 2:
            #     self.show_citizen()
            # if i == 30 - 1:
            #     self.show_mean_house_price()
            # if i == 30:
            #     self.show_income()
            # if i == 30 + 1:
            #     self.show_building_space()
            # if i == 30 + 2:
            #     self.show_house_space()
            # if i == 30 + 3:
            #     self.show_free_space()

            # if i == 179:
            #     self.commute_line()
            # if i == 359:
            #     self.commute_line()
            # if i == 719:
            #     self.commute_line()
            # if i == 1080:
            #     self.commute_line()
            #
            # if i == 180:
            #     self.show_satisfaction()
            # if i == 360:
            #     self.show_satisfaction()
            # if i == 720:
            #     self.show_satisfaction()
            # if i == 1081:
            #     self.show_satisfaction()

            # if i == 181:
            #     self.show_public_space()
            # if i == 361:
            #     self.show_public_space()
            # if i == 721:
            #     self.show_public_space()
            # if i == 1081:
            #     self.show_public_space()
            # if i ==1801:
            #     self.commute_line()
        #self.show1()
        #Print Satisfaction trend of top5+median+low5 ward
        #self.show3()


        #agent.plot_area(self.area_with_max_satisfaction)  # default is person
        #agent.plot_area(self.area_with_max_satisfaction, type="firm")  # draw firm
