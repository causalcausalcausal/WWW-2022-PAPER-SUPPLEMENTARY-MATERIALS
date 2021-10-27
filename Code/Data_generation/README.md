
### Data Generation Folder
This folder contains codes used for synthetic data generation. We adopted the method mentioned in https://github.com/tuye0305/prophet, and created a self-defined causal Directed Acyclic Graph (DAG). In this DAG, causal relationships are represented by directed arrow, from the cause node pointing to effect node.
Under BTS problem setting, such DAG consists 5 different types of node: heterogeneous variables, denoted as X, are generated from underlying causal mechanism with uncertainty effect; unobserved variables, denoted as U, are generated from latent distribution; treatment variables, denoted as T, are generated from independent sampling; outcome variable, denoted as Y, and cost variable, denoted as C are both generated from underlying causal mechanism;
