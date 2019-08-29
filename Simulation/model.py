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
        self.house_rate = 0.02  # 房地产购置税率
        self.tax_begin_numer = 20  # 从第多少轮迭代起开始

        self.schedule = RandomActivation(self)
        # self.datacollecto = DataCollector(agent_reporters={"satisfaction": lambda x: x.satisfaction})
        # self.schedule.re
        # self.datacollector = DataCollector(
        #     {"Wolves": lambda m: m.schedule.get_breed_count(Person),
        #      "Sheep": lambda m: m.schedule.get_breed_count(Sheep)})

        # self.grid = GeoSpace(crs={"init": "epsg:3857"})  # 默认epsg:4326

        # self.datacollector = DataCollector

        self.map_df = gpd.read_file(self.map_file)

        self.citizen_num = len(self.citizen_data)
        self.firm_wealth_level = 1e12  # 所有公司资产的后5%，超过即可以创建公司
        self.firm_wealth_record = dict()  # key是firm id， value是的firm的wealth

        # self.aid_to_satisfaction = dict()  # key是area id， value是地区满意度
        self.aid_to_area = dict()  # key是area id， value是area object
        self.gsscode_to_aid = dict()  # key是gsscode，value是areaid

        self.top1_area_list = []

        self.house_market = dict()  # 代售的房产。 key是房子的id，object是房子对象

        self.total_people_num = 0
        self.total_firm_num = 0

        start_time = datetime.datetime.now()

        self.area_with_max_satisfaction = None  # 当前轮满意度最高的地区，出图用
        self.area_with_max_avg_housePrice = None
        # 初始化区域
        # gird上面只放area，人和公司和某个area动态绑定
        # AC = AgentCreator(agent.Area)
        # areas = AC.from_GeoDataFrame(self.map_df)
        for i in range(self.citizen_num):
            if self.flag_of_debug and i > 30:
                break
            area_name = self.citizen_data['区名'][i]
            area_code = self.citizen_data['区编号'][i]
            print("init: %d\t%s\t%s" % (i, area_name, area_code))
            citizen = self.citizen_data[self.citizen_data['区编号'] == area_code].reset_index(drop=True)
            firm = self.firm_data[self.firm_data['区编号'] == area_code].reset_index(drop=True)
            ward = self.ward_data[self.ward_data['GSS_CODE'] == area_code].reset_index(drop=True)
            gdf = self.map_df[self.map_df['GSS_CODE'] == area_code].reset_index(drop=True)
            area = agent.Area(self, self.next_id())
            # 给area填充属性
            area.name = area_name
            area.code = area_code
            area.firm_num = firm.loc[0, '小区公司数量']
            area.employee_num = firm.loc[0, '员工数量']

            area.population = int(citizen.loc[0, '人口数量'])
            area.earning = citizen.loc[0, '人均收入']
            area.health = citizen.loc[0, '健康度']
            area.satisfaction = citizen.loc[0, '满意度']
            area.wealth = citizen.loc[0, '资产']
            area.house_rate = citizen.loc[0, '自有房率']
            area.rent_rate = citizen.loc[0, '租房率']
            area.high_edu_rate = citizen.loc[0, '高教育占比'] / 100.0
            area.middle_edu_rate = citizen.loc[0, '中教育占比'] / 100.0
            area.low_edu_rate = citizen.loc[0, '低教育占比'] / 100.0

            area.total_space = ward.loc[0, '总面积/平米']
            area.free_space = ward.loc[0, '空地比例'] * area.total_space
            area.green_space = ward.loc[0, '绿地比例'] * area.total_space
            area.building_space = ward.loc[0, '公共建筑比例'] * area.total_space
            area.house_space = ward.loc[0, '住宅比例'] * area.total_space

            area.safety = ward.loc[0, 'nor_safety']
            area.railway = ward.loc[0, 'nor_railway']

            # 年均收入乘以房价收入比=房屋总价，按照100平算计算一个
            # area.ave_house_price = float(1 / 100.0 * ward.loc[0, '人均收入'] * \
            #                              random.uniform(configs.HOUSE_PRICE_TO_INCOME * 0.8,
            #                                             configs.HOUSE_PRICE_TO_INCOME * 1.2))
            # 平均房价
            area.ave_house_price = ward.loc[0, '平均房价']
            # print("ave house price: %d" % area.ave_house_price)

            self.schedule.add(area)
            self.aid_to_area[area.unique_id] = area
            self.gsscode_to_aid[area_code] = area.unique_id
            # self.grid.add_agents(area)

            # 根据区域的人口数量，放置若干个人。 人是动态的
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
                    person.satisfaction += 5.0  # 有房产满意度增加
                # 根据人均收入，随机出每个人的财富，人居收入*0.5~1.5倍
                person.wealth = random.uniform(0.50 * configs.WEALTH_TO_INCOME,
                                               1.5 * configs.WEALTH_TO_INCOME) * ward.loc[0, '人均收入'] / 12.0
                person.health = random.uniform(0.8, 1.0) * area.health  # 根据区域健康度随机健康
                # TODO 优化是否有工作初始化逻辑
                person.job_state = random.choice([agent.JobState.Employment, agent.JobState.Unemployment])
                area.people[person.unique_id] = person
                area.pid_to_income[person.unique_id] = person.earning
                person.area = area
                self.schedule.add(person)  # 参与调度但是不放在grid上

            # 根据区域的公司数量，放置若干个公司
            # 人口数/50和人口数/20之间，随机一个数作为小区域公司数
            # firm_num = random.randint(area.population / 50, area.population / 20)
            firm_num = int(firm.loc[0, '小区公司数量'])

            for j in range(firm_num):
                x, y = self.get_one_position_by_area_code(area_code)
                firm = agent.Firm(model=self, fid=self.next_id(),
                                  xpoint=x,
                                  ypoint=y,
                                  wealth=0)
                firm.area = area
                area.firms[firm.unique_id] = firm
                self.firm_wealth_record[firm.unique_id] = firm.wealth
                self.schedule.add(firm)  # 参与调度但是不放在grid上

            # 把每个person随机绑定到一个工作上
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
        d1 = self.citizen_data[self.citizen_data['区编号'] == area_code].reset_index(drop=True)
        high_edu = d1.loc[0, '高教育占比']
        low_edu = d1.loc[0, '低教育占比']
        random_number = 100 * random.random()
        if random_number < high_edu:
            return agent.EduState.High
        elif random_number > 100 - low_edu:
            return agent.EduState.Low
        else:
            return agent.EduState.Middle

    def get_random_income(self, area_code):
        d1 = self.citizen_data[self.citizen_data['区编号'] == area_code].reset_index(drop=True)
        mu = d1.loc[0, '人均收入']
        sigma2 = mu * 0.2  # 方差设为0.2倍的均值
        return np.random.normal(mu, sigma2, 1)[0]

    def get_random_house(self, area_code):
        d1 = self.citizen_data[self.citizen_data['区编号'] == area_code].reset_index(drop=True)
        random_number = 100 * random.random()
        if random_number < d1.loc[0, '自有房率']:
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
                val = area.ave_distance
            #else:
            #    val = random.uniform(0.0, 1.0)
            satisfaction.append(val)
        print("---->try to plot-------->")
        column_name = "manyidu_" + str(self.step_num)
        self.map_df.insert(2, column_name, satisfaction, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Mean Commute Distance after day %d " % self.step_num)
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
                val = area.firm_num
            # else:
            #    val = random.uniform(0.0, 1.0)
            firms.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, firms, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward number of firms after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Oranges', column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()
        # print(self.map_df.head())
        # self.map_df.geometry.plot(column='manyidu')

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
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, population, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward number of citizens after day %d " % self.step_num)
        self.map_df.plot(figsize=(15, 15), cmap='Reds', column=column_name, ax=ax, edgecolor='black', legend=True)
        plt.show()

    def show_mean_house_price(self):
        # self.normalized()
        mean_commute_dist = []
        for _, row in self.map_df.iterrows():
            val = 0
            if row['GSS_CODE'] in self.gsscode_to_aid:
                aid = self.gsscode_to_aid[row['GSS_CODE']]
                area = self.aid_to_area[aid]
                val = area.ave_house_price
            # else:
            #    val = random.uniform(0.0, 1.0)
            mean_commute_dist.append(val)
        print("---->try to plot-------->")
        column_name = "mean_commute_dist" + str(self.step_num)
        self.map_df.insert(2, column_name, mean_commute_dist, True)
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Ward Mean House Price after day %d " % self.step_num)
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
                val = area.building_space / area.total_space
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
                val = area.house_space / area.total_space
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
                val = area.free_space / area.total_space
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
        aid_to_satisfaction = dict()  # key是area id， value是该地区满意度
        for aid, a in self.aid_to_area.items():
            if a.satisfaction > 0.0:
                aid_to_satisfaction[aid] = a.satisfaction < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_satisfaction)/2)
        for aid, s in sorted(aid_to_satisfaction.items(), key=operator.itemgetter(1)):  # 按照满意度排序结果，得到的list不能再次排序
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
        aid_to_assets = dict()  # key是area id， value是该地区满意度
        for aid, a in self.aid_to_area.items():
            aid_to_assets[aid] = a.ave_wealth < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_assets) / 2)
        for aid, s in sorted(aid_to_assets.items(), key=operator.itemgetter(1)):  # 按照满意度排序结果，得到的list不能再次排序
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
        aid_to_commute_line = dict()  # key是area id， value是该地区满意度
        for aid, a in self.aid_to_area.items():
            aid_to_commute_line[aid] = a.ave_distance < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_commute_line) / 2)
        for aid, s in sorted(aid_to_commute_line.items(), key=operator.itemgetter(1)):  # 按照满意度排序结果，得到的list不能再次排序
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
        aid_to_public_space = dict()  # key是area id， value是该地区满意度
        for aid, a in self.aid_to_area.items():
            aid_to_public_space[aid] = a.building_space < 1.0
        idx = 0

        df = dict()
        median_idx = int(len(aid_to_public_space) / 2)
        for aid, s in sorted(aid_to_public_space.items(), key=operator.itemgetter(1)):  # 按照满意度排序结果，得到的list不能再次排序
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


    # 归一化。取所有地区中得分最高的，定义为1，其他根据比例折算得分
    # 归一化区域
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
            if not a.population:  # 可怕，这个区域人都挂了
                a.satisfaction = 0
                if a.satisfaction > 0:
                    mx = a.satisfaction
                continue
            dis_num = 0  # 需要通勤的人数
            total_distance = 0  # 总的通勤距离
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
            a.ave_distance = total_distance / max(1, dis_num)  # 防止除以0
            a.cscore = a.ave_income * 1.5
            a.cscore += a.ave_wealth * 2
            a.cscore += a.green_space * 150 / a.total_space  # 绿地比例小于1，得乘一较大的数，不然对总分没影响
            a.cscore += a.building_space * 100 / a.total_space
            a.cscore += a.safety * 1.0
            a.cscore += a.railway * 1.5
            if a.cscore > a.ave_distance * 1.5:
                a.cscore -= (1.5 * a.ave_distance)
            if a.cscore > mx:
                mx = a.cscore
                self.area_with_max_satisfaction = a
        mx = max(1.0, mx)  # 防止除以0
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
        all_wealth.sort()  # 排序
        try:
            self.firm_wealth_level = get_low_npercent_list(all_wealth, 5)[-1]  # 后5%
        except:
            print("heheh")

        aid_to_satisfaction = dict()  # key是area id， value是该地区满意度
        for aid, a in self.aid_to_area.items():
            aid_to_satisfaction[aid] = a.satisfaction

        # 更新满意度前百分之一区域
        self.top1_area_list = []

        aids_sort_by_satisfaction = []
        for aid, s in sorted(aid_to_satisfaction.items(), key=operator.itemgetter(1)):  # 按照满意度排序结果，得到的list不能再次排序
            aids_sort_by_satisfaction.append(aid)

        self.top1_area_list = get_top_npercent_list(aids_sort_by_satisfaction, 1)  # 满意度前1%
        low_10_area = get_low_npercent_list(aids_sort_by_satisfaction, 10)  # 后10%
        low_5_area = get_low_npercent_list(aids_sort_by_satisfaction, 5)  # 后5%
        low_1_area = get_low_npercent_list(aids_sort_by_satisfaction, 1)  # 后1%

        area = self.aid_to_area[self.random.choice(low_5_area)]
        all_firms = list(area.firms.values())
        if len(all_firms) > 0:
            firm = self.random.choice(all_firms)
            firm.add_job(fixed=True)

        for a in list(self.aid_to_area.values()):
            a.tax_free_flag = False
        # 每天更新选择公式满意度后1%的区域个区域，免除消费税(不包括房产交易税
        self.aid_to_area[random.choice(low_1_area)].tax_free_flag = True

        # 随意选一个满意度低于5%的地区增加一个岗位
        # 增加岗位的话，随机选中一个区域，在这个区域里随机选中一个公司，给这个公司增加岗位
        # 如果这个区域没有公司，则增加岗位失败。此时就要再选。
        # 最多选五次，如果岗位还是没有增加成功，就不再增加了
        try_times = 0
        while try_times < 5:
            if self.add_job(self.aid_to_area[random.choice(low_5_area)]):
                break
            try_times += 1

        # 投资公共设施。逻辑和增加岗位一样。最多尝试五次，依然失败则放弃
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
            # for j in range(0,configs.MAX_DAYS,30):
            #     if i == j*3 - 3:
            #         self.show_firms()
            #     if i == j*3 - 2:
            #         self.show_citizen()
            #     if i == j*3 - 1:
            #         self.show_mean_house_price()
            #     if i == j*3:
            #         self.show_income()
            #     if i == j*3  + 1:
            #         self.show_building_space()
            #     if i == j*3  + 2:
            #         self.show_house_space()
            #     if i == j*3  + 3:
            #         self.show_free_space()
            #
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

            if i == 181:
                self.show_public_space()
            if i == 361:
                self.show_public_space()
            if i == 721:
                self.show_public_space()
            if i == 1081:
                self.show_public_space()
            # if i ==1801:
            #     self.commute_line()
        #self.show1()
        # 打印满意度top5+median+low5地区的满意度走势
        #self.show3()

        # 显示最高满意度地区，满意度居中的person/firm的满意度走势图
        #agent.plot_area(self.area_with_max_satisfaction)  # 默认是person
        #agent.plot_area(self.area_with_max_satisfaction, type="firm")  # 绘制firm
