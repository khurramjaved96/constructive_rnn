low = 2000
high = 2449
command = "./AnimalLearning --config experiment_configs/animal_learning_hybrid_scale_exp.json --run "
query = "INSERT INTO queue (command, priority) VALUES "
for i in range(low, high-1):
    query+="('" + command + str(i) + "'" + "," + "'20'),"

query += "('" + command + str(high-1) + "'" + "," + "'20');"

print(query)