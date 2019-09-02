from mesa import Agent
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from Simulation import model
import random
import math
from Simulation import configs
import operator


def get_distance(x1, y1, x2, y2):
    a = x1 - x2
    b = y1 - y2
    return math.sqrt(a * a + b * b)


class JobState(Enum):
    Unemployment = 0
    Employment = 1
    OwnFirm = 2


class HouseState(Enum):
    No = 0
    Yes = 1


class EduState(Enum):
    Low = 1
    Middle = 2
    High = 3


def is_edu_math(e1, e2):
    if e1 == EduState.High:
        return True
    if e1 == EduState.Middle and e2 != EduState.High:
        return True
    if e1 == EduState.Low and e2 == EduState.Low:
        return True
    return False


# Drawing, showing the satisfaction line chart of the agent.
def plot_agent(agent):
    idperfix = "pid"
    title = "Person"
    print("try to plot agent")
    if isinstance(agent, Person):
        pass
    elif isinstance(agent, Area):
        idperfix = "aid"
        title = "Area"
    elif isinstance(agent, Firm):
        idperfix = "fid"
        title = "Firm"
    else:
        print("ERROR: please check the agent object")
        return

    if len(agent.satisfaction_list) < 1:
        print("cannot plot in %s:%d" % (idperfix, agent.unique_id))
        return
    print("try to plot %s:%d,%d" % (title, agent.unique_id, len(agent.satisfaction_list)))
    df = dict()
    df[agent.unique_id] = pd.Series(agent.satisfaction_list)
    df = pd.DataFrame(df)
    print(df.head())
    fig, ax = plt.subplots(1, 1)
    ax.set_title("aid:%d, %s:%d Satisfaction " % (agent.area.unique_id, idperfix, agent.unique_id))
    df.plot(ax=ax)
    plt.show()


def plot_area(area, type="person"):
    if not area:
        return
    print("here....")
    id_to_satisfaction = dict()
    if type == "person":
        for p in area.people.values():
            if len(p.satisfaction_list) == area.round_num:
                id_to_satisfaction[p.unique_id] = p.satisfaction
    elif type == "firm":
        for f in area.firms.values():
            if len(f.satisfaction_list) == area.round_num:
                id_to_satisfaction[f.unique_id] = f.satisfaction
        pass
    else:
        print("ERROR: unkown type, please check")
    median_idx = len(id_to_satisfaction) / 2
    idx = 0
    agent = None
    for key, val in sorted(id_to_satisfaction.items(), key=operator.itemgetter(1)):
        idx += 1
        if idx >= median_idx:
            # agent = val
            if type == "person":
                agent = area.people[key]
            else:
                agent = area.firms[key]
            break
    if agent:
        plot_agent(agent)


class House():
    def __init__(self, id):
        self.unique_id = id
        self.firm = None  # belongs to which firm？
        self.owner = None  # belongs to which citizen？
        self.area = None  # belongs to which ward，immutable
        self.flag_of_buy = True  #
        self.cost = 0  # cost
        self.price = 0  # price
        self.xpoint = 0
        self.ypoint = 0


class Job():
    def __init__(self):
        self.unique_id = 0
        self.area = None  # immutable
        self.firm = None
        self.edu = None  # education requirement
        self.wage = 0  # wage


class Person(Agent):
    def __init__(self, model, pid,
                 edu=0, earning=0, wealth=0, health=0,
                 job=0, xpoint=0, ypoint=0,
                 satisfaction=0, comfort=0, house=0):
        super().__init__(unique_id=pid, model=model)
        self.model = model
        self.unique_id = pid
        self.pid = pid  # uniqe id
        self.edu = edu
        self.wealth = wealth
        self.earning = earning
        self.health = health
        self.job_state = job
        self.xpoint = xpoint
        self.ypoint = ypoint
        self.satisfaction = satisfaction
        self.comfort = comfort
        self.house_state = house

        self.flag_of_new_person = True

        self.firm = None  # belongs to which firm
        self.area = None  # belongs to which ward

        self.round_num = 0
        self.satisfaction_list = []
        pass

    def consume(self):
        hloss = random.uniform(0.1, 0.15)
        if self.health < 1:
            unitcost = random.randint(8, 45)  # each 0.1hp needs to consume money
            cost = unitcost * hloss * 10
            try:
                firm = self.area.get_one_firm()
                if firm:
                    if self.area.tax_free_flag:
                        firm.wealth += cost
                    else:
                        firm.wealth += cost * (1.0 - configs.BUSINESS_TAX_RATE)
                        self.area.gain_wealth(cost * configs.BUSINESS_TAX_RATE)  # Pay taxes
                    self.wealth -= cost
            except:
                return
        if self.health >= 1:
            self.satisfaction += 1
        elif self.health <= 0.3:
            self.satisfaction -= 1
        return

    def buy_house(self):
        if self.house_state == HouseState.Yes or self.wealth < configs.BUY_HOUSE_BOUND:
            return
        best_house = None
        best_distance = 0
        for _, h in self.model.house_market.items():
            if not best_house:
                if h.price <= self.wealth:
                    best_house = h
                    best_distance = get_distance(self.xpoint, self.ypoint, h.xpoint, h.ypoint)
                continue
            if h.price > self.wealth:
                continue
            distance = get_distance(self.xpoint, self.ypoint, h.xpoint, h.ypoint)
            if distance < 15000:  # choose closest property
                best_distance = distance
                best_house = h

        if best_house:
            # print(self.model)
            # print("house market pop: %d %d %d %d" % (
            #     best_house.unique_id, self.unique_id, self.wealth, len(self.model.house_market)))
            del self.model.house_market[best_house.unique_id]
            best_house.firm.gain_wealth(best_house.price * (1 - configs.HOUSE_RATE))
            best_house.area.gain_wealth(best_house.price * configs.HOUSE_RATE)
            self.wealth -= best_house.price
            self.house_state = HouseState.Yes
            self.xpoint = best_house.xpoint
            self.ypoint = best_house.ypoint
            if self.area != best_house.area:  # migrate by buying house
                self.area.pid_to_income.pop(self.unique_id)
                self.area.people.pop(self.unique_id)
                best_house.area.pid_to_income[self.unique_id] = self.earning
                best_house.area.people[self.unique_id] = self
            self.satisfaction += 10.0
        return

    def find_job(self):
        if self.job_state == JobState.Employment or self.job_state == JobState.OwnFirm:
            return
        best_job = None
        for _, j in self.area.labour_market.items():
            if is_edu_math(self.edu, j.edu):
                if not best_job:
                    best_job = j
                else:
                    if best_job.wage < j.wage:
                        best_job = j
        if best_job:
            self.area.labour_market.pop(best_job.unique_id)
            self.area.pid_to_income[self.unique_id] = best_job.wage
            best_job.firm.employees[self.unique_id] = self
            self.earning = best_job.wage
            self.job_state = JobState.Employment

    def step(self):
        self.round_num += 1
        # print("pid:%d, before step job_state:%d .." % (self.unique_id, self.firm != None))
        # Step 1
        if self.job_state == JobState.Unemployment \
                and self.wealth >= self.model.firm_wealth_level \
                and self.round_num >= configs.THRESHOLD_TO_CREATE_FIRM:  # run a business
            firm = Firm(fid=self.model.next_id(),
                        model=self.model,
                        xpoint=self.xpoint,
                        ypoint=self.ypoint,
                        wealth=0)
            firm.area = self.area
            firm.boss = self
            self.firm = firm
            self.job_state = JobState.OwnFirm
            self.area.firms[firm.unique_id] = firm
            self.model.firm_wealth_record[firm.unique_id] = firm.wealth
            self.model.schedule.add(firm)
            # print("add firm, pid:%d, fid:%d" % (self.unique_id, self.firm.unique_id))
            assert self.firm

        self.find_job()

        # step 2
        if self.job_state == JobState.Employment:
            self.wealth += (self.earning / 30)
        elif self.job_state == JobState.OwnFirm:
            # TODO += assets
            # assert self.firm
            if not self.firm:
                print("pid:%d, gain money" % (self.unique_id))
            nw = self.firm.get_boss_money()
            self.wealth += nw * (1.0 - configs.INCOME_TAX_RATE)
            self.area.gain_wealth(nw * configs.INCOME_TAX_RATE)
            pass

        # consume
        self.consume()

        # step 4 die
        if self.health <= 1e-5 or self.wealth <= -1500:
            if self.health <= 1e5:
                print("pid:%d die for health." % self.unique_id)
            if self.wealth <= -1500:
                print("pid:%d die for wealth." % self.unique_id)
            self.model.schedule.remove(self)
            self.area.people.pop(self.unique_id)
            if self.job_state == JobState.Employment and self.firm:
                self.firm.employees.pop(self.unique_id)
            return

        # step 5 migrate
        if self.house_state == HouseState.No and self.job_state != JobState.OwnFirm:
            low10_percent_income = self.area.get_npencent_income(top_flag=False, n=10)
            top10_percent_income = self.area.get_npencent_income(top_flag=True, n=10)
            if self.earning <= low10_percent_income or self.earning >= top10_percent_income:
                destination = None
                for a in list(self.model.aid_to_area.values()):
                    if a == self.area:  # move out
                        continue
                    if not destination:
                        destination = a
                    else:
                        if math.fabs(a.ave_income - self.earning) < math.fabs(self.earning - destination.ave_income):
                            destination = a
                if destination:
                    self.xpoint, self.ypoint = self.model.get_one_position_by_area_code(a.code)
                    self.area.people.pop(self.unique_id)
                    self.area.pid_to_income.pop(self.unique_id)

                    destination.people[self.unique_id] = self
                    destination.pid_to_income[self.unique_id] = self.earning
                    self.area = destination
                    if self.earning <= low10_percent_income:
                        self.satisfaction -= 3.0
                    else:
                        self.satisfaction += 3.0
                # change job
                if self.firm != None:
                    distance = math.sqrt((self.xpoint - self.firm.xpoint) * (self.xpoint - self.firm.xpoint) + (
                            self.ypoint - self.firm.ypoint) * (self.ypoint - self.firm.ypoint))
                    if distance >= configs.DISTANCE_TRIGGER_TO_CHANGE_JOB:
                        self.firm.employees.pop(self.unique_id)
                        self.job_state = JobState.Unemployment
                        self.firm = None
                        self.find_job()
            pass
        # step 6 buy house
        self.buy_house()

        # step 7 add person
        if self.house_state == HouseState.Yes and self.flag_of_new_person and self.round_num >= configs.THRESHOLD_TO_ADD_PERSON:
            self.flag_of_new_person = False
            person = Person(model=self.model,
                            pid=self.model.next_id(),
                            edu=self.model.get_random_edu(self.area.code),
                            earning=self.model.get_random_income(self.area.code),
                            xpoint=self.xpoint,
                            ypoint=self.ypoint,
                            house=None,
                            health=self.area.health,
                            job=None,
                            satisfaction=self.area.satisfaction,
                            comfort=None)
            self.area.people[person.unique_id] = person
            person.area = self.area
            self.model.schedule.add(person)
        # print("pid:%d, after step job_state:%d .." % (self.unique_id, self.firm != None))
        self.collect_data()
        pass

    def collect_data(self):
        if len(self.satisfaction_list) < self.round_num:
            self.satisfaction_list.append(self.satisfaction)

    def get_wages(self):
        pass

    def update(self):
        pass


class Area(Agent):
    # Initialize to area, then initialize the firm and person according to the area,
    # the person can judge whether to work at a certain firm or have a certain firm according to the rules.
    def __init__(self, model, aid,
                 code=None, name=None, population=None,
                 high_edu=0, middle_edu=0, low_edu=0,
                 earning=None, health=None, job=None,
                 xpoint=None, ypoint=None, satisfaction=None,
                 comfort=None, wealth=None,
                 house_rate=None, rent_rate=None, firm_num=None):
        super().__init__(model=model, unique_id=aid)
        self.model = model
        self.unique_id = aid
        self.aid = aid  # uniqe id
        self.code = code
        self.name = name
        self.population = population
        self.high_edu_rate = high_edu
        self.middle_edu_rate = middle_edu
        self.low_edu_rate = low_edu
        self.employee_num = 0
        self.earning = earning
        self.health = health
        self.job = job
        self.xpoint = xpoint
        self.ypoint = ypoint
        self.satisfaction = satisfaction
        self.cscore = 0  # store normalized score
        self.comfort = comfort
        self.wealth = wealth
        self.house_rate = house_rate
        self.rent_rate = rent_rate
        self.firm_num = firm_num

        self.total_space = 0  # total area
        self.free_space = 0  # usable area
        self.house_space = 0
        self.green_space = 0
        self.building_space = 0

        self.tax_free_flag = False  # Exemption from consumption tax flag

        self.firms = dict()  # key is firm uniq id，value is object in each ward
        self.people = dict()  # key is people unique id，value is object in each ward
        self.pid_to_income = dict()  # people id filt income ，get low 5 percentile

        self.low5_percent_income = []  # low 5% inocme
        self.low10_percent_income = []  # low 10% income
        self.high10_percent_incode = []  # top 0% income
        self.incomes = []
        self.ave_house_price = 0  # mean house price
        self.ave_income = 0  # mean income
        self.ave_wealth = 0  # mean wealth
        self.ave_distance = 0  # mean commuting distacne

        self.labour_market = dict()  # labour market，key is job id，value is job object
        self.dynamic_welath = 0  # 每天增加的财富

        self.satisfaction_list = []
        self.round_num = 0
        pass

    # this function only person and firm use to give asset to government
    def gain_wealth(self, money):
        self.dynamic_welath += money

    def collect_data(self):
        if len(self.satisfaction_list) < self.round_num:
            self.satisfaction_list.append(self.satisfaction)

    def get_one_firm(self):
        return random.choice(list(self.firms.values()))

    def get_npencent_income(self, top_flag=False, n=5):
        if not top_flag and n == 5 and len(self.low5_percent_income) > 0:
            return self.low5_percent_income[-1]
        elif not top_flag and n == 10 and len(self.low10_percent_income) > 0:
            return self.low10_percent_income[-1]
        elif top_flag and n == 10 and len(self.high10_percent_incode) > 0:
            return self.high10_percent_incode[-1]

        if not len(self.incomes):
            self.incomes = list(self.pid_to_income.values())
            self.incomes.sort()
        if not len(self.incomes):
            return self.ave_income
        lst = []
        if top_flag:
            lst = model.get_top_npercent_list(self.incomes, n)
        else:
            lst = model.get_low_npercent_list(self.incomes, n)
        return lst[-1]

    def step(self):
        self.round_num += 1
        # print("area %d step.." % self.unique_id)
        # update assets
        self.wealth += self.dynamic_welath
        self.dynamic_welath = 0

        # More posts. This operation is updated by model

        # Tax reduction. Updated by model

        # 更新低5%收入
        self.incomes = list(self.pid_to_income.values())
        self.incomes.sort()
        self.low5_percent_income = model.get_low_npercent_list(self.incomes, 5)
        self.low10_percent_income = model.get_low_npercent_list(self.incomes, 10)
        self.high10_percent_incode = model.get_top_npercent_list(self.incomes, 10)

        # TODO A random 10-20 fund is issued to unemployed/not holding companies every day.
        #  For each 10, this resident satisfaction is +0.1.
        for p in list(self.people.values()):
            if p.job_state == JobState.Unemployment:
                money = random.randrange(10, 30, 10)#random.randrange(10, 30, 10)
                p.wealth += money
                p.satisfaction += (money / 10) * 0.1

        # Invest in public facilities. Run in model

        pass


class Firm(Agent):
    def __init__(self, model, fid, employee_num=0, xpoint=0, ypoint=0, satisfaction=0, wealth=0):
        super().__init__(model=model, unique_id=fid)
        self.round_num = 0  # record this firm firm iteration，only plus before n iteration
        self.model = model
        self.unique_id = fid
        self.fid = fid  # uniqe id
        self.employee_num = employee_num
        self.xpoint = xpoint
        self.ypoint = ypoint
        self.satisfaction = satisfaction
        self.wealth = wealth
        self.employees = dict()  # staff list. key is person unique id，value is citizen object

        self.area = None  # belongs to which ward, immutable

        self.dynamic_wealth = 0  # change assets everyday
        self.boss = None  # wethear boss

        self.satisfaction_list = []

        pass

    # this function only citizens use to sonsum to firm
    def gain_wealth(self, money):
        self.dynamic_wealth += money

    def get_boss_money(self):
        total_wage_cost = 0
        for e in list(self.employees.values()):
            total_wage_cost += e.earning
        total_wage_cost /= 30
        return (self.dynamic_wealth - total_wage_cost) / 20

    def step(self):
        # print("firm %d step.." % self.unique_id)
        self.round_num += 1

        # 发放工资
        total_wage_cost = 0
        for e, person in self.employees.items():
            total_wage_cost += self.employees[e].earning
            person.wealth += person.earning * (1 - configs.INCOME_TAX_RATE)
        # The first N (30) rounds are paid only, but the company's assets are not deducted,
        # otherwise it will soon go bankrupt
        if self.model.step_num >= configs.FIRM_THRESHOLD:
            self.wealth -= total_wage_cost
        if not self.area.tax_free_flag:
            self.area.gain_wealth(total_wage_cost * configs.INCOME_TAX_RATE)
            # self.wealth -= total_wage_cost * configs.HOUSE_RATE
            pass

        # rent
        self.pay_house_rent()

        # Income is the consumption behavior of the citizens and is defined in the behavior of citizens.

        # fire employee
        self.update_employee()

        # update assets
        self.wealth += self.dynamic_wealth
        self.dynamic_wealth = 0
        self.model.firm_wealth_record[self.unique_id] = self.wealth

        # invest real estate
        self.invest_house()

        # check bankrupt
        self.bankrupt()

        self.collect_data()

    def pay_wages(self):
        if self.area.tax_free_flag:
            pass
        pass

    def pay_house_rent(self):
        rent_money = self.area.ave_house_price / 37.0 / 365 / 6 * len(self.employees)
        if self.model.step_num > configs.FIRM_THRESHOLD:
            self.wealth -= rent_money

    def collect_data(self):
        if len(self.satisfaction_list) < self.round_num:
            self.satisfaction_list.append(self.satisfaction)


    def add_job(self, fixed=False):
        job = Job()
        job.unique_id = self.model.next_id()
        job.firm = self
        job.area = job.firm.area
        # random choice education level
        job.edu = random.choice([EduState.Low, EduState.Middle, EduState.High])
        # According to the mean income data, randomly assign the salary of the post.The higher the degree, the higher the price
        if not fixed:
            wage = self.area.ave_income
            if job.edu == EduState.High:
                wage *= random.uniform(0.9, 1.5)
            elif job.edu == EduState.Middle:
                wage *= random.uniform(0.6, 1.2)
            else:
                wage *= random.uniform(0.5, 0.8)
        else:
            wage = self.area.ave_income
        job.wage = wage
        self.area.labour_market[job.unique_id] = job

    def update_employee(self):
        total_wage = 0
        for p in list(self.employees.values()):
            total_wage += p.earning
        if total_wage >= self.dynamic_wealth:
            self.satisfaction += 1
            self.add_job()
        else:  # reduce a post
            self.satisfaction -= 1
            p = random.choice(list(self.employees.values()))
            self.employees.pop(p.unique_id)
            p.firm = None
            p.job = JobState.Unemployment

    def invest_house(self):
        total_wage = 0
        for e in self.employees:
            total_wage += self.employees[e].earning
        old_wealth = self.wealth
        if self.wealth >= total_wage * 180 and self.wealth > 0:
            old_a = None
            empty_loop1 = 0
            while empty_loop1 < 5:
                a = self.model.get_top1_area()
                if not a:
                    break
                if old_a == a:
                    break
                empty_loop1 += 1
                old_a = a
                a = self.model.aid_to_area[a]
                if a.free_space > 0:
                    money = self.wealth - 180 * total_wage
                    space = money / self.area.ave_house_price / 85  #
                    space = min(space, self.area.free_space / 20.0)
                    # money = space * self.area.ave_house_price
                    empty_loop = 0
                    while space >= 0 and empty_loop < 15 and self.wealth > 0:
                        hspace = random.uniform(42.5, 102)  # invest per house area range
                        if hspace >= space:
                            empty_loop += 1
                            continue
                        empty_loop = 0

                        a.free_space -= hspace
                        space -= hspace

                        house = House(self.model.next_id())
                        house.area = a
                        house.firm = self
                        house.cost = hspace * a.ave_house_price
                        house.price = random.uniform(1.1, 1.135) * house.cost
                        house.xpoint, house.ypoint = self.model.get_one_position_by_area_code(a.code)

                        self.wealth -= house.cost
                        # print("house market add: %d" % house.unique_id)
                        self.model.house_market[house.unique_id] = house
                    #
                    # print("fid:%d, invest house, before wealth:%d, after:%d，"
                    #       "fid:%d, aid:%d, ave house price:%d" % (self.unique_id, old_wealth, self.wealth,
                    #                                               self.unique_id,
                    #                                               self.area.unique_id,
                    #                                               self.area.ave_house_price))

                    break

    def bankrupt(self):
        if self.wealth < 0:
            # print("firm %d brankrupt, money %d" % (self.unique_id, self.wealth))
            # return
            # print()
            # Delete all employees. All employee status reset to unemployed
            es = list(self.employees.keys())
            for emplyee in es:
                # self.model.a
                self.employees[emplyee].job = JobState.Unemployment  # unemployment
                self.employees[emplyee].firm = None
                self.employees.pop(emplyee)
            if self.unique_id not in self.model.firm_wealth_record:
                print("heiheiheihei")
            if self.boss:
                self.boss.job_state = JobState.Unemployment
                self.boss.firm = None
            self.model.firm_wealth_record.pop(self.unique_id)
            self.area.firms.pop(self.unique_id)
            self.model.schedule.remove(self)
            print("fid:%d die for bankrupt." % self.unique_id)
