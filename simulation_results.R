#need R > 3.5
library(readr)
library(dplyr)
library(estimatr)
library(ggplot2)
theme_set(theme_bw())
library(scales)
library(RColorBrewer)

setwd("/home/amin/Dropbox (MIT)/Contagion/Seeding/seeding_queries/")

nrho = "nrho_10"
#nrho = "nrho_100"
ed = readr::read_csv(sprintf("data/edge_query_%s.csv", nrho))
names(ed) = tolower(names(ed))
names(ed) = gsub(" ", "_", names(ed))
names(ed) = gsub("_without_leaves", "_wol", names(ed))


k_colors <- c(
  "2" = brewer.pal(8, "Set1")[5],
  "4" = "blue",#brewer.pal(8, "Set1")[2],
  "10" = brewer.pal(8, "Set1")[1]
)

penn_max_nodes = 41536
penn_max_edges = 1362220

ed = ed %>%
  mutate(
    spread_size_percent = 100 * spread_size / penn_max_nodes,
    node_cost_wol_percent = 100 * node_cost_wol / penn_max_nodes,
    node_cost_percent = 100 * node_cost / penn_max_nodes,
    edge_cost_percent = 100 * edge_cost / penn_max_edges
  )



#ed_t0 <- subset(ed,t == 0)
#ed_t0 <- mutate(ed_t0,spread_size_percent_t0 = spread_size_percent)
#ed_t0 <- select (ed_t0,c(spread_size_percent_t0,k,seed_set_id))

#total <- merge(ed,ed_t0,all=TRUE)


eds = ed %>% filter(t<37) %>%
  group_by(k, t, seed_set_id) %>%
  summarise(
    n = n(),
    spread_size_mean = mean(spread_size_percent),
    node_cost_percent = node_cost_percent[1],
    edge_cost_percent = edge_cost_percent[1]
  )



edss = eds %>%
  group_by(k, t) %>%
  summarise(
    n_seed_set_ids = n(),
    spread_size_mean_se = sd(spread_size_mean) / sqrt(n()),
    spread_size_mean = mean(spread_size_mean),
    node_cost_mean = mean(node_cost_percent),
    node_cost_mean_se = sd(node_cost_percent) / sqrt(n()),
    edge_cost_mean = mean(edge_cost_percent),
    edge_cost_mean_se = sd(edge_cost_percent) / sqrt(n())
  )



#k_colors <- scales::seq_gradient_pal("orange","blue", "red")(
#  seq(0, 1, length.out = length(unique(edss$k))) 
#)


# show distribution of spread sizes
ggplot(
  aes(x = t, y = spread_size_percent,
      color = factor(k), group = t),
  data = ed
) +
  facet_wrap( ~ k, labeller = label_both) +
  scale_x_log10() +
  geom_boxplot(outlier.shape = NA) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  )

ggplot(
  aes(x = factor(t), y = spread_size_percent,
      color = factor(k), group = paste(t, sprintf("%02d", k))),
  data = ed
) +
  geom_boxplot(
    #outlier.shape = NA,
    outlier.size = .1, outlier.alpha = .1, outlier.shape = 4,
    lwd = .4,
    width = .4,
    position = position_dodge2(width = 1.3)
    ) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_y_continuous(name = "spread size (%)") +
  scale_x_discrete(name = "T")  +
  theme(legend.position = c(0.9, 0.4))

ggsave(sprintf("figures/edge_queries_spread_size_boxplot_t_%s.pdf", nrho), width = 4.5, height = 3.5)


# mapping between costs

ggplot(
  aes(
    x = edge_cost_percent, 
    y = node_cost_percent,
    color = factor(k),
    shape = factor(k),
    group = k
  ),
  data = eds
) +
  geom_line() +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_shape_discrete(name = "k", guide = guide_legend(reverse = TRUE))

t_to_node_cost_trans_func = function(x) approx(edss$t, edss$node_cost_mean, xout = x)$y
edge_to_node_cost_trans_func = function(x) approx(edss$edge_cost_mean, edss$node_cost_mean, xout = x)$y



ggplot(
  aes(
    x = t, 
    y = spread_size_mean,
    ymin = spread_size_mean - qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    ymax = spread_size_mean + qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    color = factor(k), shape = factor(k), group = k
    ),
  data = edss
) +
  geom_line() +
  geom_pointrange(size = .4) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
    ) +
  scale_shape_discrete(name = "k", guide = guide_legend(reverse = TRUE)) +
  scale_y_continuous(name = "spread size (mean %)") +
  scale_x_continuous(
    name = "T",
    sec.axis = sec_axis(
      ~ t_to_node_cost_trans_func(.),
      name = "nodes queried (mean %)"
      )
  ) +
  theme(legend.position = c(0.9, 0.25))

# ggsave("figures/edge_queries_spread_size_by_t_with_nodes_queried_nrho_100.pdf", width = 4.5, height = 3.5)
ggsave(sprintf("figures/edge_queries_spread_size_by_t_with_nodes_queried_%s.pdf", nrho), width = 4.5, height = 3.5)

ggplot(
  aes(
    x = edge_cost_mean, 
    y = spread_size_mean,
    ymin = spread_size_mean - qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    ymax = spread_size_mean + qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    color = factor(k), shape = factor(k), group = k
  ),
  data = edss
) +
  geom_line() +
  geom_pointrange(size = .4) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_shape_discrete(name = "k", guide = guide_legend(reverse = TRUE)) +
  scale_y_continuous(name = "spread size (mean %)") +
  scale_x_continuous(
    name = "edges queried (mean %)",
    sec.axis = sec_axis(
      ~ edge_to_node_cost_trans_func(.),
      name = "nodes queried (mean %)"
    )
  ) +
  theme(legend.position = c(0.9, 0.25))

# ggsave("figures/edge_queries_spread_size_by_edges_and_nodes_queried_nrho_100.pdf", width = 4.5, height = 3.5)
ggsave(sprintf("figures/edge_queries_spread_size_by_edges_and_nodes_queried_%s.pdf", nrho), width = 4.5, height = 3.5)

##################spread queries

sp = readr::read_csv("data/spread_query.csv")

names(sp) = tolower(names(sp))
names(sp) = gsub(" ", "_", names(sp))
names(sp) = gsub("_without_leaves", "_wol", names(sp))

sp = sp %>%
  mutate(
    spread_size_percent = 100 * spread_size / penn_max_nodes,
    revenue = 0.1 * spread_size,
    cost = 10*k + num_of_spread_query,
  ) 

sp = sp %>%
  mutate(
    profit = revenue - cost
  )

# getting the expected spread sizes and profits

sps = sp %>%
  group_by(k,num_of_spread_query, seed_set_id) %>%
  summarise(
    n = n(),
    spread_size_expected = mean(spread_size_percent),
    profit_expected = mean(profit)
  )

spss = sps %>%
  group_by(k,num_of_spread_query) %>%
  summarise(
    n_seed_set_ids = n(),
    spread_size_mean_se = sd(spread_size_expected) / sqrt(n()),
    spread_size_mean = mean(spread_size_expected),
    profit_mean_se = sd(profit_expected) / sqrt(n()),
    profit_mean = mean(profit_expected)
  )

# show distribution of spread sizes
ggplot(
  aes(x = num_of_spread_query, y = spread_size_percent,
      color = factor(k), group = num_of_spread_query),
  data = sp
) +
  facet_wrap( ~ k, labeller = label_both) +
  scale_x_log10() +
  geom_boxplot(outlier.shape = NA) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  )

ggplot(
  aes(x = factor(num_of_spread_query), y = spread_size_percent,
      color = factor(k), group = paste(num_of_spread_query, sprintf("%02d", k))),
  data = sp
) +
  geom_boxplot(
    #outlier.shape = NA,
    outlier.size = .1, outlier.alpha = .1, outlier.shape = 4,
    lwd = .4,
    width = .4,
    position = position_dodge2(width = 1.3)
  ) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_y_continuous(name = "spread size (%)") +
  scale_x_discrete(name = "query cost")  +
  theme(legend.position = c(0.9, 0.4))



ggsave("figures/spread_queries_spread_size_boxplot_num_of_spread_query.pdf", width = 4.5, height = 3.5)

# mapping between costs

#t_to_node_cost_trans_func = function(x) approx(edss$t, edss$node_cost_mean, xout = x)$y
#edge_to_node_cost_trans_func = function(x) approx(edss$edge_cost_mean, edss$node_cost_mean, xout = x)$y


ggplot(
  aes(
    x = num_of_spread_query, 
    y = spread_size_mean,
    ymin = spread_size_mean - qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    ymax = spread_size_mean + qt(.975, df = n_seed_set_ids) * spread_size_mean_se,
    color = factor(k), shape = factor(k), group = k
  ),
  data = spss
) +
  geom_line() +
  geom_pointrange(size = .4) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_shape_discrete(name = "k", guide = guide_legend(reverse = TRUE)) +
  scale_y_continuous(name = "spread size (mean %)") +
  scale_x_continuous(
    name = "query cost",
    #sec.axis = sec_axis(
    #  ~ t_to_node_cost_trans_func(.),
    #  name = "nodes queried (mean %)"
    #)
  ) +
  theme(legend.position = c(0.9, 0.25))

ggsave("figures/spread_queries_spread_size_by_query_cost.pdf", width = 4.5, height = 3.5)


# plot the profit

ggplot(
  aes(
    x = num_of_spread_query, 
    y = profit_mean,
    ymin = profit_mean - qt(.975, df = n_seed_set_ids) * profit_mean_se,
    ymax = profit_mean + qt(.975, df = n_seed_set_ids) * profit_mean_se,
    color = factor(k), shape = factor(k), group = k
  ),
  data = spss
) +
  geom_line() +
  geom_pointrange(size = .4) +
  scale_color_manual(
    name = "k", values = k_colors,
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_shape_discrete(name = "k", guide = guide_legend(reverse = TRUE)) +
  scale_y_continuous(name = "mean profit") +
  scale_x_continuous(
    name = "query cost",
    #sec.axis = sec_axis(
    #  ~ t_to_node_cost_trans_func(.),
    #  name = "nodes queried (mean %)"
    #)
  ) +
  theme(legend.position = c(0.9, 0.25))

ggsave("figures/spread_queries_profit_by_query_cost.pdf", width = 4.5, height = 3.5)

