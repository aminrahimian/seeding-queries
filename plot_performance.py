import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


colors = {10 : 'red', 2 : 'orange', 4 : 'blue'}#, 10 : 'black'

cost_4 = [20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 80, 104, 132]
cost_10 = [10, 30, 40, 50, 60, 80, 100, 130]
cost_2 = [20,22,24,26,28,30,32,34,36,38,40,42,44,50,60,80,110,130]
spread_4 = []
spread_10 = []
spread_2 = []

for c in cost_10:
    spread_10.append(pickle.load(open(
        "./data/fb100-data/pickled_samples/spreading_pickled_samples/k_10/spread_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_query_cost_"
        + str(
            c) + "_vanilla IC_.pkl", 'rb')))

for c in cost_4:
    spread_4.append(pickle.load(open(
        "./data/fb100-data/pickled_samples/spreading_pickled_samples/k_4/spread_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_query_cost_"
        + str(
            c) + "_vanilla IC_.pkl", 'rb')))


for c in cost_2:

    s = []

    s += [np.mean(pickle.load(open(
         "./data/fb100-data/pickled_samples/spreading_pickled_samples/k_2/spread_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_query_cost_"
         + str(c) + "_vanilla IC_.pkl", 'rb')))]

    s += [np.std(pickle.load(open(
        "./data/fb100-data/pickled_samples/spreading_pickled_samples/k_2/spread_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_query_cost_"
        + str(c) + "_vanilla IC_.pkl", 'rb')))]

    spread_2.append(s)

# spread size versus spread query cost

y_4 = [s[0] for s in spread_4]
y_10 = [s[0] for s in spread_10]
y_2 = [s[0] for s in spread_2]
print(y_2)
ci_4 = [1.96 * s[1] / 25000 ** 0.5 for s in spread_4]
ci_10 = [1.96 * s[1] / 25000 ** 0.5 for s in spread_10]
ci_2 = [1.96 * s[1] / 25000 ** 0.5 for s in spread_2]

print('ci_2',ci_2)

fig = plt.figure()

plt.errorbar(cost_2, y_2, ci_2, label="k=2", color = colors[2])
plt.errorbar(cost_4, y_4, ci_4, label="k=4", color = colors[4])
plt.errorbar(cost_10, y_10, ci_10, label="k=10", color = colors[10])


fig.set_figwidth(7)
fig.set_figheight(6)
plt.legend()
plt.xlabel("query cost")
plt.ylabel("spread size")

fig.savefig('./figures/spread_size_vs_spread_query.pdf',
                bbox_inches = 'tight')
plt.close()


revenue_per_spread = 0.01
cost_per_seed = 1
cost_per_query = 0.1
profits_4 = []
profits_10 = []
profits_2 = []
for i in range(len(spread_4)):
    profits_4 += [spread_4[i][0]*revenue_per_spread - cost_4[i]*cost_per_query - 4*cost_per_seed]

for i in range(len(spread_10)):
    profits_10 += [spread_10[i][0]*revenue_per_spread - cost_10[i]*cost_per_query - 10*cost_per_seed]

for i in range(len(spread_2)):
    profits_2 += [spread_2[i][0]*revenue_per_spread - cost_2[i]*cost_per_query - 2*cost_per_seed]


# profit versus spread query cost

print(cost_4)
print(profits_4)


print(cost_10)
print(profits_10)


print(cost_2)
print(profits_2)

fig = plt.figure()

plt.plot(cost_2, profits_2, label="k=2", color = colors[2])
plt.plot(cost_4, profits_4, label="k=4", color = colors[4])
plt.plot(cost_10, profits_10, label="k=10", color = colors[10])

fig.set_figwidth(7)
fig.set_figheight(6)
plt.legend()
plt.xlabel("query cost")
plt.ylabel("profit")

fig.savefig('./figures/profit_vs_spread_query.pdf',
                bbox_inches = 'tight')
plt.close()


# spread size versus edge query cost


T2s = list(range(1,15))

spreads2 = []

for T in T2s:
    spread2 = pickle.load(open(
        './data/fb100-data/pickled_samples/spreading_pickled_samples/k_2/edge_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_T_'
        + str(T) + '_vanilla IC_.pkl', 'rb'))
    spreads2.append((np.mean(spread2), np.std(spread2)))

y2 = [s[0] for s in spreads2]

for s in spreads2:
    print(s)
ci2 = [1.96 * s[1] / 25000 ** 0.5 for s in spreads2]


T4s = list(range(1,15))
spreads4 = []
for T in T4s:
    spread4 = pickle.load(open(
        './data/fb100-data/pickled_samples/spreading_pickled_samples/k_4/edge_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_T_'
        + str(T) + '_vanilla IC_.pkl', 'rb'))
    spreads4.append((np.mean(spread4), np.std(spread4)))

y4 = [s[0] for s in spreads4]

for s in spreads4:
    print(s)
ci4 = [1.96 * s[1] / 25000 ** 0.5 for s in spreads4]

T10s = list(range(1,15))
spreads10 = []
for T in T10s:
    spread10 = pickle.load(open(
        './data/fb100-data/pickled_samples/spreading_pickled_samples/k_10/edge_query/Penn94/spread_size_samples_fb100_edgelist_Penn94_T_'
        + str(T) + '_vanilla IC_.pkl', 'rb'))
    spreads10.append((np.mean(spread10), np.std(spread10)))

y10 = [s[0] for s in spreads10]

for s in spreads10:
    print(s)
ci10 = [1.96 * s[1] / 25000 ** 0.5 for s in spreads10]



fig = plt.figure()
plt.errorbar(T2s, y2, ci2, label="k=2", color = colors[2])
plt.errorbar(T4s, y4, ci4, label="k=4", color = colors[4])
plt.errorbar(T10s, y10, ci10, label="k=10", color = colors[10])

fig.set_figwidth(7)
fig.set_figheight(6)
plt.legend()

plt.xlabel("T")
plt.ylabel("spread size")

fig.savefig('./figures/spread_size_vs_edge_query.pdf',
                bbox_inches = 'tight')
plt.close()