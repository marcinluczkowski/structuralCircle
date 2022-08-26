""" CALCULATION """ 

mapping_id = []
logs = []

for i in range(len(demand_list)):
    match=False
    for j in range(len(supply_list)):
        if demand_list[i][0] <= supply_list[j][0] and demand_list[i][1] <= supply_list[j][1] and demand_list[i][3] <= supply_list[j][3] and demand_list[i][5] <= supply_list[j][5]:
            match=True
            mapping_id.append(supply_list[j][-1])
            # cut_length.append(supply_list[j][0] - demand_list[i][0])
            break
    if match:
        if plural_assign:
            # shorten the supply element:
            supply_list[j][0] = supply_list[j][0] - demand_list[i][0]
            # sort the supply list
            supply_list = sorted(supply_list, key=lambda x: x[0]) # TODO move this element instead of sorting whole list
            logs.append("#"+str(i)+" Found element #"+str(j)+" and utilized only "+str(supply_list[j][0]/1000)+"m of "+str(demand_list[i][0]/1000)+"m. Demand: L="+str(demand_list[i][0]/1000)+"m, A="+str(demand_list[i][1]/100)+"cm2, I="+str(demand_list[i][3]/10000)+"cm4, H="+str(demand_list[i][5]/10)+"cm.")
        else:
            del supply_list[j]
            logs.append("#"+str(i)+" Found element #"+str(j)+" and utilized fully. Demand: L="+str(demand_list[i][0]/1000)+"m, A="+str(demand_list[i][1]/100)+"cm2, I="+str(demand_list[i][3]/10000)+"cm4, H="+str(demand_list[i][5]/10)+"cm.")
    else:
        mapping_id.append(None)
        logs.append("#"+str(i)+" Not found. Demand: L="+str(demand_list[i][0]/1000)+"m, A="+str(demand_list[i][1]/100)+"cm2, I="+str(demand_list[i][3]/10000)+"cm4, H="+str(demand_list[i][5]/10)+"cm.")
