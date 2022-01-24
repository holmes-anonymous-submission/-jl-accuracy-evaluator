import csv,  math, random, labmath

# loading the dataset, keeping only the columns that we need (the first four columns)
# they are age, job, marital, and education
data = []

with open('dataset_1.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    # skip the header
    next(csv_reader)

    for row in csv_reader:
        if len(row) > 4:
            # four columns: age, job, marital, and education
            # age 17--98 divided by 10 and subtracted by 1      =>      9   groups
            # job 1--12 subtracted by 1                         =>      12  groups
            # martial 1--4 subtracted by 1                      =>      4   groups
            # education 1--8 subtracted by 1                    =>      8   groups
            res = []
            res.append(int(row[0]) // 10 - 1)
            res.append(int(row[1]) - 1)
            res.append(int(row[2]) - 1)
            res.append(int(row[3]) - 1)
            data.append(res)

print("total number of records: {}".format(len(data)))

# shuffle the data randomly
random.shuffle(data)

def permute_age(dataset):
    ages = []
    for i in range(len(dataset)):
        ages.append(dataset[i][0])
    random.shuffle(ages)
    new_dataset = []
    for i in range(len(dataset)):
        new_dataset.append([ages[i], dataset[i][1], dataset[i][2], dataset[i][3]])
    return new_dataset

# the first dataset is the first half
# the second dataset is the first half, with the age being permuted within the column
# the public distribution is taken from the public dataset

first_dataset = data[0:int(len(data)/2)]
second_dataset = permute_age(first_dataset)
public_dataset = data

counter_first_dataset = [0] * 3456
counter_second_dataset = [0] * 3456
counter_public_dataset = [0] * 3456

for i in range(len(first_dataset)):
    res = 0
    res = res + first_dataset[i][3]
    res = res * 8
    res = res + first_dataset[i][2]
    res = res * 4
    res = res + first_dataset[i][1]
    res = res * 12
    res = res + first_dataset[i][0]

    counter_first_dataset[res] = counter_first_dataset[res] + 1
    
for i in range(len(second_dataset)):
    res = 0
    res = res + second_dataset[i][3]
    res = res * 8
    res = res + second_dataset[i][2]
    res = res * 4
    res = res + second_dataset[i][1]
    res = res * 12
    res = res + second_dataset[i][0]

    counter_second_dataset[res] = counter_second_dataset[res] + 1

for i in range(len(public_dataset)):
    res = 0
    res = res + public_dataset[i][3]
    res = res * 8
    res = res + public_dataset[i][2]
    res = res * 4
    res = res + public_dataset[i][1]
    res = res * 12
    res = res + public_dataset[i][0]

    counter_public_dataset[res] = counter_public_dataset[res] + 1

# compute critical value for unnormalized Pearson chi-square test statistic
import numpy as np

mean = np.zeros( 3456) 
cov = np.zeros( (3456, 3456)) 
for i in range(3456):
    for j in range(3456):
        if i == j:
            cov[i][j] = 1.0*(counter_public_dataset[i]/len(public_dataset)) * \
                        (1 - counter_public_dataset[i]/len(public_dataset))
        else: 
            cov[i][j] = 1.0*counter_public_dataset[i]/len(public_dataset) *\
                        counter_public_dataset[j]/len(public_dataset)

x = np.random.multivariate_normal(mean, cov, 5000)

samples = []
for i in range(5000):
    s = 0
    for j in range(3456):
        s = s + x[i][j] * x[i][j]
    samples.append(s)
print("critical value: {}".format(np.quantile(samples, 0.95)))

# compute the unnormalized Pearson chi-squared test statistics naively
naive_first_dataset_test_statistics = 0
for i in range(len(counter_first_dataset)):
    naive_first_dataset_test_statistics = naive_first_dataset_test_statistics + \
    (counter_first_dataset[i] - len(first_dataset)/len(public_dataset)*counter_public_dataset[i])*\
    (counter_first_dataset[i] - len(first_dataset)/len(public_dataset)*counter_public_dataset[i])
    
naive_second_dataset_test_statistics = 0
for i in range(len(counter_second_dataset)):
    naive_second_dataset_test_statistics = naive_second_dataset_test_statistics + \
    (counter_second_dataset[i] - len(second_dataset)/len(public_dataset)* counter_public_dataset[i])*\
    (counter_second_dataset[i] - len(second_dataset)/len(public_dataset)*counter_public_dataset[i])
    
print("naive: 1st dataset, test statistics = {}".format(naive_first_dataset_test_statistics/len(first_dataset)))
print("naive: 2nd dataset, test statistics = {}".format(naive_second_dataset_test_statistics/len(second_dataset)))

LARGER_R = 12
SMALLER_R = 6

# initialize the JL by choosing random polynomials
# note that strictly, the polynomials need to go through irreducibility test, here we omit for simplicity
polynomials = []
p = pow(2, 62) - pow(2, 16) + 1
for i in range(LARGER_R):
    polynomial = []
    for j in range(4):
        polynomial.append(random.randint(0, p - 1))
    polynomials.append(polynomial)

# start to compute the r = 12 case
# The r = 6 case will be obtained from r = 12
jl_counter_first_dataset_large = [0] * LARGER_R
jl_counter_second_dataset_large = [0] * LARGER_R
jl_counter_public_dataset_large = [0] * LARGER_R

assert(LARGER_R > SMALLER_R)

# compute the JL counter result (-1 and 1) without normalization (divided by 1/\sqrt{r})
# for the first, second, and the public datasets
for i in range(len(first_dataset)):
    res = 0
    res = res + first_dataset[i][3]
    res = res * 8
    res = res + first_dataset[i][2]
    res = res * 4
    res = res + first_dataset[i][1]
    res = res * 12
    res = res + first_dataset[i][0]

    for j in range(LARGER_R):
        poly_res = (polynomials[j][3] * res * res * res + polynomials[j][2] * res * res + \
                    polynomials[j][1] * res + polynomials[j][0]) % p
        jl_counter_first_dataset_large[j] = jl_counter_first_dataset_large[j] + labmath.jacobi(poly_res, p)

for i in range(len(second_dataset)):
    res = 0
    res = res + second_dataset[i][3]
    res = res * 8
    res = res + second_dataset[i][2]
    res = res * 4
    res = res + second_dataset[i][1]
    res = res * 12
    res = res + second_dataset[i][0]

    for j in range(LARGER_R):
        poly_res = (polynomials[j][3] * res * res * res + polynomials[j][2] * res * res + \
                    polynomials[j][1] * res + polynomials[j][0]) % p
        jl_counter_second_dataset_large[j] = jl_counter_second_dataset_large[j] + labmath.jacobi(poly_res, p)

for i in range(len(public_dataset)):
    res = 0
    res = res + public_dataset[i][3]
    res = res * 8
    res = res + public_dataset[i][2]
    res = res * 4
    res = res + public_dataset[i][1]
    res = res * 12
    res = res + public_dataset[i][0]

    for j in range(LARGER_R):
        poly_res = (polynomials[j][3] * res * res * res + polynomials[j][2] * res * res + \
                    polynomials[j][1] * res + polynomials[j][0]) % p
        jl_counter_public_dataset_large[j] = jl_counter_public_dataset_large[j] + labmath.jacobi(poly_res, p)

# start computing the small ones as well as the normalized results
jl_counter_first_dataset_small_normalized = [0.0] * SMALLER_R
jl_counter_second_dataset_small_normalized = [0.0] * SMALLER_R
jl_counter_public_dataset_small_normalized = [0.0] * SMALLER_R
jl_counter_first_dataset_large_normalized = [0.0] * LARGER_R
jl_counter_second_dataset_large_normalized = [0.0] * LARGER_R
jl_counter_public_dataset_large_normalized = [0.0] * LARGER_R

# the normalization factors are 1/sqrt(6) and 1/sqrt(12)
normalization_factor_small = 1 / math.sqrt(SMALLER_R)
normalization_factor_large = 1 / math.sqrt(LARGER_R)

for i in range(SMALLER_R):
    jl_counter_first_dataset_small_normalized[i] = jl_counter_first_dataset_large[i] * normalization_factor_small
    jl_counter_second_dataset_small_normalized[i] = jl_counter_second_dataset_large[i] * normalization_factor_small
    jl_counter_public_dataset_small_normalized[i] = jl_counter_public_dataset_large[i] * normalization_factor_small / 2

for i in range(LARGER_R):
    jl_counter_first_dataset_large_normalized[i] = jl_counter_first_dataset_large[i] * normalization_factor_large
    jl_counter_second_dataset_large_normalized[i] = jl_counter_second_dataset_large[i] * normalization_factor_large
    jl_counter_public_dataset_large_normalized[i] = jl_counter_public_dataset_large[i] * normalization_factor_large / 2
    # divided by two because the public dataset has 2x more data. The division result should mimic n *p

# compute the test statistics, which will be the sum of squares of the delta
jl_test_statistics_first_dataset_small = 0.0
jl_test_statistics_second_dataset_small = 0.0

jl_test_statistics_first_dataset_large = 0.0
jl_test_statistics_second_dataset_large = 0.0

for i in range(SMALLER_R):
    jl_test_statistics_first_dataset_small = jl_test_statistics_first_dataset_small + \
                                       pow(jl_counter_first_dataset_small_normalized[i] - \
                                           jl_counter_public_dataset_small_normalized[i], 2)
    jl_test_statistics_second_dataset_small = jl_test_statistics_second_dataset_small + \
                                        pow(jl_counter_second_dataset_small_normalized[i] - \
                                            jl_counter_public_dataset_small_normalized[i], 2)

for i in range(LARGER_R):
    jl_test_statistics_first_dataset_large = jl_test_statistics_first_dataset_large + \
                                       pow(jl_counter_first_dataset_large_normalized[i] - \
                                           jl_counter_public_dataset_large_normalized[i], 2)
    jl_test_statistics_second_dataset_large = jl_test_statistics_second_dataset_large + \
                                        pow(jl_counter_second_dataset_large_normalized[i] - \
                                            jl_counter_public_dataset_large_normalized[i], 2)

print("jl: 1st dataset, test statistics = {} for r = {}, and {} for r = {}".format(
    jl_test_statistics_first_dataset_small/len(first_dataset),
    SMALLER_R,
    jl_test_statistics_first_dataset_large/len(first_dataset),
    LARGER_R
))

print("jl: 2nd dataset, test statistics = {} for r = {}, and {} for r = {}".format(
    jl_test_statistics_second_dataset_small/len(second_dataset),
    SMALLER_R,
    jl_test_statistics_second_dataset_large/len(second_dataset),
    LARGER_R
))
